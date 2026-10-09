"""Reuse measured references and evaluate Raw/LCC with unchanged inference."""
import csv
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
METRICS = ('dice', 'jaccard', 'hd95', 'asd')
TRACKED_CASE = 'WSJB9P4JCXUVHBOYFVWL'
PROTOCOL = dict(dataset='LA', labeled_percent=10, seed=42, alpha=0.1,
                iterations=30000, patch_size=[112,112,80], batch_size=4, labeled_bs=2,
                validation_interval=200, primary='LCC Dice', checkpoint_selection='max LCC validation Dice',
                raw_role='secondary; never used to choose checkpoint or primary outcome',
                case_count=20, exclude_cases=False,
                severe_failure_rule='foreground_voxels == 0 OR dice < 0.70 OR hd95 > 40 voxel',
                tracked_case=TRACKED_CASE, additional_seeds_run=False)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def failure_reasons(row):
    reasons = []
    if int(row['foreground_voxels']) == 0: reasons.append('empty_prediction')
    if float(row['dice']) < 0.70: reasons.append('dice_below_0.70')
    if float(row['hd95']) > 40: reasons.append('hd95_above_40_voxel')
    return ';'.join(reasons)


def annotate(row):
    result = dict(row)
    result['severe_failure_reasons'] = failure_reasons(row)
    result['severe_failure'] = bool(result['severe_failure_reasons'])
    result['tracked_case'] = row['case_id'] == TRACKED_CASE
    return result


def validate_coverage(rows, cases, modes):
    for mode in modes:
        for view in ('raw','lcc'):
            selected = [r for r in rows if r['mode']==mode and r['prediction']==view]
            if len(selected)!=20 or {r['case_id'] for r in selected} != set(cases):
                raise ValueError('Missing/duplicate cases: %s/%s' % (mode,view))
            if not all(np.isfinite(float(r[k])) for r in selected for k in METRICS):
                raise ValueError('Nonfinite metrics: %s/%s' % (mode,view))


def reuse_references(cases):
    directory = ROOT/'analysis/results/experiment_A_diagnosis_v2'
    metadata = json.loads((directory/'diagnosis_v2_metadata.json').read_text())
    if metadata['case_ids'] != cases or metadata['patch_size'] != [112,112,80] or metadata['stride'] != [18,18,4]:
        raise ValueError('Cached references do not match the predefined inference protocol')
    if metadata['probability_threshold'] != 0.5 or metadata['connectivity'] != 26:
        raise ValueError('Cached references have different postprocessing')
    for value in metadata['checkpoints'].values():
        if sha256(value['path']) != value['sha256']:
            raise ValueError('Reference checkpoint changed: '+value['path'])
    rows = []
    for original in read_csv(directory/'prediction_raw_vs_lcc.csv'):
        row = dict(original)
        row['mode'] = {'baseline':'none','guidance':'bgs'}[row.pop('model')]
        for key in METRICS: row[key] = float(row[key])
        rows.append(annotate(row))
    validate_coverage(rows,cases,('none','bgs'))
    # Verify the reused primary metrics also reproduce the existing A result CSVs.
    run=ROOT/'Results/experiment_A_bgs_alpha0p1_seed42'
    for mode,filename in [('none','baseline_test.csv'),('bgs','guidance_test.csv')]:
        reference={r['case_id']:r for r in read_csv(run/filename)}
        for row in rows:
            if row['mode']==mode and row['prediction']=='lcc':
                if any(abs(row[k]-float(reference[row['case_id']][k]))>1e-8 for k in METRICS):
                    raise ValueError('Cached LCC metrics differ from completed Experiment A')
    provenance=dict(checkpoints=metadata['checkpoints'],csv=str(directory/'prediction_raw_vs_lcc.csv'),
                    csv_sha256=sha256(directory/'prediction_raw_vs_lcc.csv'),
                    metadata_sha256=sha256(directory/'diagnosis_v2_metadata.json'))
    return rows,provenance


def evaluate_shuffle(checkpoint, iteration, device, data_path, cases):
    from model.vnet import VNet
    from prediction import test_single_case, getLargestCC
    from utils.utils import calculate_metric_percase
    model=VNet(n_channels=1,n_classes=2).to(device).eval()
    kwargs={'weights_only':True} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    model.load_state_dict(torch.load(checkpoint,map_location=device,**kwargs))
    args=SimpleNamespace(dataset='LA',data_path=str(data_path),patch_size=[112,112,80],num_classes=2)
    image_root=Path(data_path)/'2018LA_Seg_TrainingSet'
    if not image_root.is_dir(): image_root=Path(data_path)
    rows=[]
    with torch.no_grad():
        for case in cases:
            with h5py.File(image_root/case/'mri_norm2.h5','r') as f:
                image=f['image'][:]; gt=f['label'][:]
            raw,_=test_single_case(args,model,image,18,4,[112,112,80],2)
            lcc=getLargestCC(raw)
            for view,prediction in [('raw',raw),('lcc',lcc)]:
                values=calculate_metric_percase(torch.from_numpy(gt.astype('float32')),prediction,thr=0.5)
                mask=prediction.astype(bool); target=gt==1
                row=dict(case_id=case,mode='shuffled_bgs',checkpoint_iteration=iteration,prediction=view,
                         **{k:float(v) for k,v in zip(METRICS,values)})
                row.update(foreground_voxels=int(mask.sum()),gt_foreground_voxels=int(target.sum()),
                           tp_voxels=int((mask & target).sum()),fp_voxels=int((mask & ~target).sum()),
                           fn_voxels=int((~mask & target).sum()),connectivity=26)
                rows.append(annotate(row))
            print('Evaluated shuffled_bgs '+case,flush=True)
    validate_coverage(rows,cases,('shuffled_bgs',))
    return rows


def summarize(rows):
    summaries=[]
    for mode in ('none','bgs','shuffled_bgs'):
        for view in ('raw','lcc'):
            group=[r for r in rows if r['mode']==mode and r['prediction']==view]
            if not group: continue
            summaries.append(dict(mode=mode,prediction=view,n_cases=len(group),
                                  checkpoint_iteration=int(group[0]['checkpoint_iteration']),
                                  primary=view=='lcc',severe_failures=sum(bool(failure_reasons(r)) for r in group),
                                  **{k:float(np.mean([float(r[k]) for r in group])) for k in METRICS}))
    return summaries


def write_outputs(output,rows,cases,complete=False):
    output=Path(output)
    validate_coverage(rows,cases,('none','bgs','shuffled_bgs') if complete else ('none','bgs'))
    fields=['case_id','mode','checkpoint_iteration','prediction',*METRICS,'foreground_voxels',
            'gt_foreground_voxels','tp_voxels','fp_voxels','fn_voxels','connectivity',
            'severe_failure','severe_failure_reasons','tracked_case']
    write_csv(output/'per_case_metrics.csv',rows,fields)
    for mode in ('none','bgs','shuffled_bgs'):
        group=[r for r in rows if r['mode']==mode]
        if group: write_csv(output/(mode+'_per_case.csv'),group,fields)
    write_csv(output/'severe_failures.csv',[r for r in rows if failure_reasons(r)],fields)
    write_csv(output/'tracked_case.csv',[r for r in rows if r['case_id']==TRACKED_CASE],fields)
    summary=summarize(rows)
    write_csv(output/'summary.csv',summary)
    primary={r['mode']:r for r in summary if r['prediction']=='lcc'}
    lines=['# Experiment A-Control', '',
           '主要流程预先固定为 LCC；以原 LCC validation Dice 选择最佳 EMA checkpoint。Raw 为辅助评价，不改变选择或结论标准。',
           'LA 10%，seed 42，alpha 0.1，batch 2 labeled + 2 unlabeled，patch 112×112×80，30000 iterations。', '',
           '| 模式 | 预测 | iteration | n | Dice | Jaccard | HD95 | ASD | 严重失效 n |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('| %s | %s | %d | %d | %.6f | %.6f | %.6f | %.6f | %d |' %
                     (r['mode'],r['prediction'],r['checkpoint_iteration'],r['n_cases'],*[r[k] for k in METRICS],r['severe_failures']))
    lines+=['','严重失效：空预测、Dice < 0.70 或 HD95 > 40 voxel；是本实验的描述阈值。所有病例保留在均值中，详见 severe_failures.csv。',
            '既有异常病例 WSJB9P4JCXUVHBOYFVWL 始终单列在 tracked_case.csv；不因此排除它。',
            'LA 原 validation 和最终评价使用同一 test.list 的 20 例，本轮保留原 split 和流程。HD95/ASD 为 voxel；ASD 沿用原单向定义，空预测沿用原距离 100。', '']
    if complete:
        delta=primary['bgs']['dice']-primary['shuffled_bgs']['dice']
        lines+=['LCC Dice 的 BGS−shuffled_bgs 差为 %+.6f（%+.3f 个百分点）。' % (delta,delta*100),
                ('单 seed 的点估计对真实通道对应关系有潜在优势；需要其他 seed 验证，尚不能确立稳定收益。' if delta>0 else
                 '本 seed 的主要指标未显示真实通道对应关系优于随机对应关系。'),
                'BGS 相对 Baseline 的 LCC Dice 差为 %+.6f。' % (primary['bgs']['dice']-primary['none']['dice']),
                '模型是独立训练后比较最终表现，没有跨模型 channel matching。decoder 原有 BatchNorm 耦合保留。',
                '本轮只训练 shuffled_bgs / seed42；不自动运行其他 seed 或 A+。']
    else:
        lines+=['shuffled_bgs 训练进行中。上表仅复用已有 none/bgs 结果，尚无新对照结论。']
    (output/'experiment_report.md').write_text('\n'.join(lines)+'\n')
    return summary
