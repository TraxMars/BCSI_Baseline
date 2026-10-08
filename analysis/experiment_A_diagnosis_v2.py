#!/usr/bin/env python3
"""Experiment A-D2: read-only inference, prediction topology and logged BGS stability."""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from experiment_A_failure_diagnosis import (
    load_model, markdown_table, read_case_metrics, resolve_case_path,
    sha256, write_csv, PROTECTED_SOURCES, METRICS,
)
from prediction import getLargestCC, test_single_case
from utils.utils import calculate_metric_percase

ANOMALY = 'WSJB9P4JCXUVHBOYFVWL'
PATCH = [112, 112, 80]
PHASES = [('early', 0, 10000), ('middle', 10000, 20000), ('late', 20000, 30000)]
TOPK = re.compile(r'BGS iteration (\d+) : diagnostic Top-8 w channels (\[[^\]]*\])')


def config_from_log(path):
    for line in path.read_text().splitlines():
        if 'Namespace(' in line:
            expression = ast.parse(line[line.index('Namespace('):], mode='eval').body
            return {item.arg: ast.literal_eval(item.value) for item in expression.keywords}
    raise RuntimeError('Missing training configuration: ' + str(path))


def snapshot(paths):
    return {str(path.resolve()): sha256(path) for path in sorted(set(paths))}


def component_labels(mask):
    labels = label(mask.astype(bool), connectivity=3)
    sizes = np.bincount(labels.ravel())[1:]
    ids = np.arange(1, len(sizes) + 1)
    order = ids[np.lexsort((ids, -sizes))]
    return labels, sizes, order


def mask_metrics(case, model, variant, mask, gt, checkpoint_iteration):
    values = calculate_metric_percase(torch.from_numpy(gt), mask.astype(np.uint8))
    labels, sizes, order = component_labels(mask)
    tp, fp, fn = int((mask & gt).sum()), int((mask & ~gt).sum()), int((~mask & gt).sum())
    row = dict(case_id=case, model=model, checkpoint_iteration=checkpoint_iteration,
               prediction=variant, **dict(zip(METRICS, map(float, values))),
               tp_voxels=tp, fp_voxels=fp, fn_voxels=fn,
               foreground_voxels=int(mask.sum()), gt_foreground_voxels=int(gt.sum()),
               component_count=len(sizes), connectivity=26,
               largest_component_voxels=int(sizes[order[0]-1]) if len(order) else 0,
               largest_component_fraction=float(sizes.max()/mask.sum()) if mask.any() else 0.0)
    assert abs(row['dice'] - 2*tp/(2*tp+fp+fn)) < 1e-12
    assert abs(row['jaccard'] - tp/(tp+fp+fn)) < 1e-12
    return row


def compare_cases(rows, cases, output):
    indexed = {(r['case_id'], r['model'], r['prediction']): r for r in rows}
    metrics = METRICS + ('tp_voxels', 'fp_voxels', 'fn_voxels', 'foreground_voxels',
                         'component_count', 'largest_component_fraction')
    pairs = []
    for case in cases:
        row = {'case_id': case}
        for metric in metrics:
            for variant in ('raw', 'lcc'):
                baseline = indexed[case, 'baseline', variant][metric]
                guidance = indexed[case, 'guidance', variant][metric]
                row.update({f'baseline_{variant}_{metric}': baseline,
                            f'guidance_{variant}_{metric}': guidance,
                            f'delta_{variant}_{metric}': guidance-baseline})
            raw_delta, lcc_delta = row[f'delta_raw_{metric}'], row[f'delta_lcc_{metric}']
            row[f'lcc_change_in_signed_gap_{metric}'] = lcc_delta-raw_delta
            row[f'lcc_change_in_absolute_gap_{metric}'] = abs(lcc_delta)-abs(raw_delta)
            if metric in METRICS:
                gap_change = abs(lcc_delta)-abs(raw_delta)
                effect = 'widened' if gap_change > 1e-12 else 'narrowed' if gap_change < -1e-12 else 'unchanged'
                row[f'lcc_gap_effect_{metric}'] = ('ranking_reversed_' if raw_delta*lcc_delta < 0 else '')+effect
        pairs.append(row)
    write_csv(output / 'paired_raw_vs_lcc.csv', pairs)
    return indexed, pairs


def longest_run(records, channel):
    best, current = [], []
    for record in records:
        if channel in record['channels']:
            if current and record['iteration'] != current[-1]+500:
                current = []
            current.append(record['iteration'])
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
    return best


def bgs_stability(log_path, output, channels=64):
    content = log_path.read_text()
    configuration = config_from_log(log_path)
    expected = set(range(0, configuration['max_iterations'], 500))
    observations = {}
    for line_number, line_text in enumerate(content.splitlines(), 1):
        match = TOPK.search(line_text)
        if match:
            iteration = int(match[1])
            observations.setdefault(iteration, []).append((ast.literal_eval(match[2]), line_number))
    rows, records, previous = [], [], None
    invalid = []
    for iteration in sorted(expected | set(observations)):
        observed = observations.get(iteration, [])
        values = observed[0][0] if observed else []
        valid = (bool(observed) and len(values) == 8 and len(set(values)) == 8 and
                 all(isinstance(v, int) and 0 <= v < channels for v in values) and
                 all(v == values for v, _ in observed))
        phase = next((name for name, start, end in PHASES if start <= iteration < end), 'outside')
        row = dict(iteration=iteration, phase=phase, expected_500_iteration_record=int(iteration in expected),
                   observed=int(bool(observed)), valid=int(valid), logged_count=len(observed),
                   log_line_numbers=';'.join(str(n) for _, n in observed),
                   top8_indices=json.dumps(values) if observed else '',
                   previous_observed_valid_iteration='', iteration_gap='',
                   adjacent_record_top8_jaccard='', consecutive_500_top8_jaccard='',
                   retained_channels='', entered_channels='', exited_channels='')
        if observed and not valid:
            invalid.append(iteration)
        if valid:
            record = dict(iteration=iteration, phase=phase, channels=values)
            if previous is not None:
                before, current = set(previous['channels']), set(values)
                jaccard = len(before & current)/len(before | current)
                gap = iteration-previous['iteration']
                row.update(previous_observed_valid_iteration=previous['iteration'], iteration_gap=gap,
                           adjacent_record_top8_jaccard=jaccard,
                           consecutive_500_top8_jaccard=jaccard if gap == 500 else '',
                           retained_channels=json.dumps(sorted(before & current)),
                           entered_channels=json.dumps(sorted(current-before)),
                           exited_channels=json.dumps(sorted(before-current)))
            records.append(record)
            previous = record
        rows.append(row)
    write_csv(output / 'bgs_topk_temporal_stability.csv', rows)
    frequencies = []
    for channel in range(channels):
        present = [r for r in records if channel in r['channels']]
        run = longest_run(records, channel)
        row = dict(channel=channel, observed_valid_records=len(records), top8_count=len(present),
                   top8_frequency=len(present)/len(records) if records else '',
                   first_observed_iteration=present[0]['iteration'] if present else '',
                   last_observed_iteration=present[-1]['iteration'] if present else '',
                   longest_consecutive_500_records=len(run),
                   longest_run_start=run[0] if run else '', longest_run_end=run[-1] if run else '')
        for phase, start, end in PHASES:
            phase_records = [r for r in records if start <= r['iteration'] < end]
            count = sum(channel in r['channels'] for r in phase_records)
            row.update({f'{phase}_observed_records': len(phase_records),
                        f'{phase}_top8_count': count,
                        f'{phase}_top8_frequency': count/len(phase_records) if phase_records else ''})
        frequencies.append(row)
    write_csv(output / 'bgs_channel_top8_frequency.csv', frequencies)
    phase_rows = []
    for phase, start, end in PHASES:
        selected = [r for r in records if start <= r['iteration'] < end]
        pairs = [r['consecutive_500_top8_jaccard'] for r in rows
                 if start <= r['iteration'] < end and
                 r['previous_observed_valid_iteration'] != '' and
                 r['previous_observed_valid_iteration'] >= start and
                 r['consecutive_500_top8_jaccard'] != '']
        common = sorted(set.intersection(*(set(r['channels']) for r in selected))) if selected else []
        top_by_frequency = sorted(range(channels), key=lambda c: (
            -sum(c in r['channels'] for r in selected), c))[:8] if selected else []
        phase_rows.append(dict(phase=phase, start=start, end_exclusive=end, actual_records=len(selected),
            expected_records=len(range(start, end, 500)), adjacent_pairs=len(pairs),
            mean_adjacent_jaccard=float(np.mean(pairs)) if pairs else '',
            min_adjacent_jaccard=float(np.min(pairs)) if pairs else '',
            strict_intersection=json.dumps(common), frequency_top8=json.dumps(top_by_frequency)))
    write_csv(output / 'bgs_phase_summary.csv', phase_rows)
    phase_pair_rows = []
    for i, first in enumerate(phase_rows):
        for second in phase_rows[i+1:]:
            a, b = set(json.loads(first['frequency_top8'])), set(json.loads(second['frequency_top8']))
            phase_pair_rows.append(dict(first=first['phase'], second=second['phase'],
                frequency_top8_jaccard=len(a & b)/len(a | b) if a | b else '',
                shared_channels=json.dumps(sorted(a & b))))
    write_csv(output / 'bgs_phase_pair_comparison.csv', phase_pair_rows)
    adjacent = [r['consecutive_500_top8_jaccard'] for r in rows if r['consecutive_500_top8_jaccard'] != '']
    summary = dict(expected_records=len(expected), actual_valid_records=len(records),
        missing_iterations=sorted(expected-set(observations)), invalid_iterations=invalid,
        unexpected_iterations=sorted(set(observations)-expected),
        adjacent_pairs=len(adjacent), adjacent_jaccard_mean=float(np.mean(adjacent)) if adjacent else None,
        adjacent_jaccard_min=float(np.min(adjacent)) if adjacent else None,
        adjacent_jaccard_max=float(np.max(adjacent)) if adjacent else None,
        strict_all_records_intersection=sorted(set.intersection(*(set(r['channels']) for r in records))) if records else [],
        frequent_core_80pct=[r['channel'] for r in frequencies if r['top8_frequency'] != '' and r['top8_frequency'] >= 0.8],
        frequent_core_80pct_each_phase=[r['channel'] for r in frequencies
            if all(r[f'{phase}_top8_frequency'] != '' and r[f'{phase}_top8_frequency'] >= 0.8 for phase, _, _ in PHASES)],
        phases=phase_rows, phase_comparisons=phase_pair_rows)
    (output / 'bgs_stability_summary.json').write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [1, 2]})
    axes[0].plot([r['iteration'] for r in rows if r['consecutive_500_top8_jaccard'] != ''], adjacent, marker='.', linewidth=1)
    axes[0].set(xlabel='Training iteration', ylabel='Adjacent Top-8 Jaccard', ylim=(0, 1.05))
    matrix = np.full((channels, len(rows)), np.nan)
    for i, row in enumerate(rows):
        if row['valid']:
            matrix[:, i] = 0
            matrix[json.loads(row['top8_indices']), i] = 1
    axes[1].imshow(matrix, aspect='auto', origin='lower', interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ticks = list(range(0, len(rows), 10))
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([rows[i]['iteration'] for i in ticks])
    axes[1].set(xlabel='Logged training iteration', ylabel='Experiment A x3 channel')
    fig.tight_layout()
    fig.savefig(output / 'bgs_top8_stability.png', dpi=180)
    plt.close(fig)
    return summary, frequencies


def anomaly_analysis(saved_path, output):
    with np.load(saved_path) as archive:
        gt = archive['gt_mask'].astype(bool)
        raw = {name: archive[f'{name}_raw_prediction'].astype(bool) for name in ('baseline', 'guidance')}
    distance = distance_transform_edt(~gt)
    baseline_labels, _, baseline_order = component_labels(raw['baseline'])
    baseline_first_two = [baseline_labels == i for i in baseline_order[:2]]
    rows, distances, lcc_effects = [], [], []
    added_fp = raw['guidance'] & ~raw['baseline'] & ~gt
    for name in ('baseline', 'guidance'):
        labels, sizes, order = component_labels(raw[name])
        overlaps = np.bincount(labels[gt], minlength=len(sizes)+1)
        added = np.bincount(labels[added_fp], minlength=len(sizes)+1)
        for rank, component_id in enumerate(order, 1):
            component = labels == component_id
            coordinates = np.argwhere(component)
            tp = int(overlaps[component_id])
            volume = int(sizes[component_id-1])
            fp_distance = distance[component & ~gt]
            row = dict(case_id=ANOMALY, model=name, component_id=int(component_id), volume_rank=rank,
                       foreground_voxels=volume, gt_overlap_tp_voxels=tp, fp_voxels=volume-tp,
                       gt_coverage_fraction=tp/int(gt.sum()), component_gt_overlap_fraction=tp/volume,
                       retained_by_lcc=int(rank == 1),
                       guidance_added_fp_voxels=int(added[component_id]) if name == 'guidance' else '',
                       guidance_added_fp_fraction=int(added[component_id])/int(added_fp.sum())
                           if name == 'guidance' and added_fp.any() else '',
                       centroid_voxel=json.dumps(coordinates.mean(axis=0).tolist()),
                       bbox_min_voxel=json.dumps(coordinates.min(axis=0).tolist()),
                       bbox_max_voxel_inclusive=json.dumps(coordinates.max(axis=0).tolist()),
                       fp_distance_to_gt_max_voxels=float(fp_distance.max()) if len(fp_distance) else 0,
                       fp_distance_to_gt_p95_voxels=float(np.percentile(fp_distance, 95)) if len(fp_distance) else 0)
            for threshold in (10, 20, 30):
                row[f'fp_distance_gt_gt{threshold}_voxels'] = int((fp_distance > threshold).sum())
            for rank_index in range(2):
                row[f'baseline_raw_rank{rank_index+1}_overlap_voxels'] = int((component & baseline_first_two[rank_index]).sum()) if len(baseline_first_two) > rank_index else 0
            rows.append(row)
        final = getLargestCC(raw[name]).astype(bool)
        tp_removed = int((raw[name] & ~final & gt).sum())
        fp_removed = int((raw[name] & ~final & ~gt).sum())
        lcc_effects.append(dict(case_id=ANOMALY, model=name, raw_tp=int((raw[name] & gt).sum()),
            raw_fp=int((raw[name] & ~gt).sum()), raw_fn=int((~raw[name] & gt).sum()),
            lcc_tp=int((final & gt).sum()), lcc_fp=int((final & ~gt).sum()), lcc_fn=int((~final & gt).sum()),
            removed_tp_voxels=tp_removed, removed_fp_voxels=fp_removed, added_fn_voxels=tp_removed))
        assert np.all(final <= raw[name])
        for variant, mask in [('raw', raw[name]), ('lcc', final)]:
            values = distance[mask & ~gt]
            distances.append(distance_row(name, variant, values))
    distances.append(distance_row('guidance', 'new_fp_relative_to_baseline_raw', distance[added_fp]))
    write_csv(output / 'anomaly_component_gt_overlap.csv', rows)
    write_csv(output / 'anomaly_lcc_confusion_effect.csv', lcc_effects)
    write_csv(output / 'anomaly_fp_distance_to_gt.csv', distances)
    summary = dict(case_id=ANOMALY, gt_voxels=int(gt.sum()), guidance_added_fp_voxels=int(added_fp.sum()),
        baseline_tp_lost_in_guidance_raw=int((gt & raw['baseline'] & ~raw['guidance']).sum()),
        newly_recovered_tp_in_guidance_raw=int((gt & ~raw['baseline'] & raw['guidance']).sum()),
        components=rows, lcc_effects=lcc_effects, fp_distances=distances)
    (output / 'anomaly_topology_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def distance_row(model, variant, values):
    row = dict(model=model, prediction=variant, fp_voxels=len(values),
        mean_distance_voxels=float(values.mean()) if len(values) else 0,
        median_distance_voxels=float(np.median(values)) if len(values) else 0,
        p95_distance_voxels=float(np.percentile(values, 95)) if len(values) else 0,
        max_distance_voxels=float(values.max()) if len(values) else 0)
    for threshold in (10, 20, 30):
        row[f'fp_farther_than_{threshold}_voxels'] = int((values > threshold).sum())
    return row


def report(output, args, rows, pairs, anomaly, bgs, frequencies, verification):
    mean_gap_raw = float(np.mean([p['delta_raw_dice'] for p in pairs]))
    mean_gap_lcc = float(np.mean([p['delta_lcc_dice'] for p in pairs]))
    lines = ['# Experiment A-D2 — Prediction Topology and BGS Stability Check', '',
        'Baseline EMA 27000 与 Experiment A EMA 29000 在相同 20 个 LA 病例上重新推理。'
        '全部病例（包括预先指定异常病例）进入每项均值和配对统计，没有删除任何病例。', '',
        f'直接实测：全 20 例的平均 Dice 差（Guidance−Baseline）由 raw 的 {mean_gap_raw:+.6f} '
        f'变为 LCC 的 {mean_gap_lcc:+.6f}。'
        + ('因此，LCC 导致这两个 checkpoint 的平均 Dice 排名反转。' if mean_gap_raw*mean_gap_lcc < 0 else
           'LCC 改变了两模型的平均 Dice 差距。')
        + '这说明当前最终评估差距受到预测拓扑与后处理的直接影响；不证明 BGS 排序漂移是训练退化的因果来源。', '',
        '## 方法与保护', '',
        '- 直接调用原 `prediction.test_single_case` 与 `getLargestCC`；patch `[112,112,80]`，'
        'stride `(18,18,4)`，前景概率 `>0.5`，26 邻接连通域。',
        '- 输入直接读取原 `mri_norm2.h5/image`；没有重新归一化、增强或重采样。',
        '- 四个指标直接调用原 `calculate_metric_percase`；HD95/ASD 单位为 voxel（原数据没有 spacing），'
        'ASD 沿用原有预测表面到 GT 的单向定义。空预测沿用原代码的 Dice/Jaccard=0、HD95/ASD=100。',
        '- 原始五个受保护源文件、全部既有 Results 文件与上一版诊断文件，运行前后 SHA256 完全一致。',
        '- 全程 eval / inference_mode；模型参数与 buffers 逐张量不变，没有训练、反向传播或新增 BN 设置。',
        '- 异常病例的重新推理 raw / LCC masks 与上一版已保存 masks 逐 voxel 一致；'
        '连通域 overlap 分析直接读取上一版已保存的 raw masks 和 GT。',
        f'- 实际执行设备：`{args.device}`。完整来源与哈希见 `diagnosis_v2_metadata.json` 和 `verification.json`。', '',
        '## 直接观察到的事实', '', '### 全部 20 例：原始阈值预测与 LCC', '',
        '`delta = guidance − baseline`；Dice/Jaccard 负值、HD95/ASD 正值表示 Guidance 更差。', '']
    aggregates = []
    for variant in ('raw', 'lcc'):
        for model in ('baseline', 'guidance'):
            selected = [r for r in rows if r['model'] == model and r['prediction'] == variant]
            aggregates.append([model, variant] + [f'{np.mean([r[m] for r in selected]):.6f}' for m in METRICS] +
                              [f'{np.mean([r[m] for r in selected]):.1f}' for m in ('fp_voxels', 'fn_voxels', 'foreground_voxels')])
    lines += markdown_table(['模型', '预测', 'Dice', 'Jaccard', 'HD95', 'ASD', 'FP mean', 'FN mean', '预测体积 mean'], aggregates)
    lines += ['', '| 指标 | raw 配对平均 delta | LCC 配对平均 delta | LCC 对 signed gap 的变化 |', '|---|---:|---:|---:|']
    delta_summary = {}
    for metric in METRICS:
        raw_delta = float(np.mean([p[f'delta_raw_{metric}'] for p in pairs]))
        lcc_delta = float(np.mean([p[f'delta_lcc_{metric}'] for p in pairs]))
        delta_summary[metric] = (raw_delta, lcc_delta)
        lines.append(f'| {metric} | {raw_delta:+.6f} | {lcc_delta:+.6f} | {lcc_delta-raw_delta:+.6f} |')
    lines += ['', '逐病例差异与 LCC gap 变化见 `paired_raw_vs_lcc.csv`。绝对 gap 变化与 signed gap 分别保存，'
              '避免把排名反转误称为简单放大/缩小。', '']
    lines += ['逐病例 Dice gap（正值为 Guidance 更高；widened/narrowed 指绝对差距放大/缩小）：', '']
    lines += markdown_table(['病例', 'raw Dice delta', 'LCC Dice delta', 'LCC gap effect'],
        [(p['case_id'], f'{p["delta_raw_dice"]:+.6f}', f'{p["delta_lcc_dice"]:+.6f}',
          p['lcc_gap_effect_dice']) for p in pairs])
    lines += ['']
    for metric in METRICS:
        sign = 1 if metric in ('dice', 'jaccard') else -1
        raw_delta, lcc_delta = delta_summary[metric]
        change = sign*(lcc_delta-raw_delta)
        count = sum(sign*p[f'delta_lcc_{metric}'] < 0 for p in pairs)
        lines += [f'- {metric}：LCC 后 Guidance 劣于 Baseline 的病例 {count}/20；'
                  f'LCC 使 Guidance 的相对表现{"下降" if change < -1e-12 else "上升" if change > 1e-12 else "不变"}。']
    lines += ['', '### 异常病例 WSJB9P4JCXUVHBOYFVWL', '',
              f'GT 前景为 {anomaly["gt_voxels"]} voxels。以下组件按体积降序排名，保留原 label ID；'
              '全部组件明细见 `anomaly_component_gt_overlap.csv`。', '']
    top_rows = [r for r in anomaly['components'] if r['volume_rank'] <= 2]
    lines += markdown_table(['模型', '体积 rank', '组件 ID', '体积', 'GT overlap / TP', 'FP', '覆盖 GT 比例', 'LCC 保留'],
        [(r['model'], r['volume_rank'], r['component_id'], r['foreground_voxels'], r['gt_overlap_tp_voxels'],
          r['fp_voxels'], f'{r["gt_coverage_fraction"]:.6f}', r['retained_by_lcc']) for r in top_rows])
    indexed = {(r['case_id'], r['model'], r['prediction']): r for r in rows}
    lines += ['', '异常病例性能：', '']
    lines += markdown_table(['模型', '预测', 'Dice', 'HD95', 'TP', 'FP', 'FN', '组件数', '最大组件占比'],
        [(name, variant, f'{indexed[ANOMALY,name,variant]["dice"]:.6f}',
          f'{indexed[ANOMALY,name,variant]["hd95"]:.6f}',
          *[int(indexed[ANOMALY,name,variant][m]) for m in ('tp_voxels','fp_voxels','fn_voxels','component_count')],
          f'{indexed[ANOMALY,name,variant]["largest_component_fraction"]:.6f}')
         for name in ('baseline', 'guidance') for variant in ('raw', 'lcc')])
    anomaly_pair = next(p for p in pairs if p['case_id'] == ANOMALY)
    lines += ['', f'异常病例 Dice 差由 raw 的 {anomaly_pair["delta_raw_dice"]:+.6f} '
              f'变为 LCC 的 {anomaly_pair["delta_lcc_dice"]:+.6f}；'
              f'绝对差距增加 {anomaly_pair["lcc_change_in_absolute_gap_dice"]:.6f}。']
    lines += ['', 'LCC 对混淆计数的直接影响：', '']
    lines += markdown_table(['模型', '删除 TP', '删除 FP', '增加 FN'],
        [(r['model'], r['removed_tp_voxels'], r['removed_fp_voxels'], r['added_fn_voxels']) for r in anomaly['lcc_effects']])
    guidance_components = [r for r in anomaly['components'] if r['model'] == 'guidance']
    dominant = max(guidance_components, key=lambda r: r['guidance_added_fp_voxels']) if guidance_components else None
    if dominant:
        lines += ['', f'Guidance 相对 Baseline raw 新增 FP {anomaly["guidance_added_fp_voxels"]} voxels；'
            f'其中 {dominant["guidance_added_fp_voxels"]}（{100*dominant["guidance_added_fp_fraction"]:.2f}%）'
            f'属于 Guidance 体积 rank {dominant["volume_rank"]} / ID {dominant["component_id"]}。',
            f'Guidance raw 同时丢失 Baseline 已命中的 GT voxel {anomaly["baseline_tp_lost_in_guidance_raw"]}，'
            f'新增命中 GT voxel {anomaly["newly_recovered_tp_in_guidance_raw"]}。', '']
        largest = next(r for r in guidance_components if r['volume_rank'] == 1)
        lines += [f'Guidance 最大组件与 Baseline raw 最大、第二大组件的空间交集分别为 '
            f'{largest["baseline_raw_rank1_overlap_voxels"]}、{largest["baseline_raw_rank2_overlap_voxels"]} voxels。'
            '这是两个最终预测的空间对应关系，不能据此重建训练过程中发生过的组件合并事件。', '']
    lines += ['远距离 FP：计算每个 FP 到最近 GT 前景 voxel 的欧氏距离；10/20/30 voxel 是报告用阈值，'
              '不是物理距离或临床标准。', '']
    lines += markdown_table(['模型/FP 集合', 'FP 数', '距离 median', '距离 P95', '距离 max', '>10', '>20', '>30'],
        [(f'{r["model"]}/{r["prediction"]}', r['fp_voxels'], f'{r["median_distance_voxels"]:.3f}',
          f'{r["p95_distance_voxels"]:.3f}', f'{r["max_distance_voxels"]:.3f}',
          *[r[f'fp_farther_than_{t}_voxels'] for t in (10,20,30)]) for r in anomaly['fp_distances']])
    new_distance = anomaly['fp_distances'][-1]
    lines += ['', f'实测新增 FP 中 {new_distance["fp_farther_than_20_voxels"]} 个距 GT 超过 20 voxels，'
              f'最大距离 {new_distance["max_distance_voxels"]:.3f} voxels。', '',
              '### Experiment A 自身的 BGS Top-8 稳定性', '',
              f'- 期望 {bgs["expected_records"]} 条；有效实测 {bgs["actual_valid_records"]} 条。',
              f'- 缺失 iteration：`{bgs["missing_iterations"]}`；无效记录：`{bgs["invalid_iterations"]}`；'
              f'非计划 iteration：`{bgs["unexpected_iterations"]}`。没有补造任何 Top-8。',
              f'- 相邻 500 次实测记录共有 {bgs["adjacent_pairs"]} 对；Jaccard 均值 '
              f'`{bgs["adjacent_jaccard_mean"]}`，最小/最大 '
              f'`{bgs["adjacent_jaccard_min"]}` / `{bgs["adjacent_jaccard_max"]}`。',
              f'- 所有有效记录的严格交集：`{bgs["strict_all_records_intersection"]}`。',
              f'- 全程出现频率 ≥80% 的集合：`{bgs["frequent_core_80pct"]}`；'
              f'每个阶段均 ≥80% 的集合：`{bgs["frequent_core_80pct_each_phase"]}`。', '',
              '早/中/晚阶段固定为 [0,10000)、[10000,20000)、[20000,30000)。'
              '下表 frequency Top-8 按该阶段出现次数排序、平局按 channel ID；不是虚构的瞬时 Top-8。', '']
    lines += markdown_table(['阶段', '实测记录', '阶段内相邻 Jaccard mean', '阶段严格交集', '频率 Top-8'],
        [(r['phase'], r['actual_records'], r['mean_adjacent_jaccard'], r['strict_intersection'], r['frequency_top8']) for r in bgs['phases']])
    lines += ['', '阶段 frequency Top-8 集合对比：', '']
    lines += markdown_table(['阶段 1', '阶段 2', 'Jaccard', '共有 channel'],
        [(r['first'], r['second'], r['frequency_top8_jaccard'], r['shared_channels']) for r in bgs['phase_comparisons']])
    lines += ['', '最高频 channel（全部 64 个 channel 的完整频率见 CSV）：', '']
    lines += markdown_table(['channel', '次数/有效记录', '频率', '最长连续 500 次记录数', '起止 iteration'],
        [(r['channel'], f'{r["top8_count"]}/{r["observed_valid_records"]}', f'{r["top8_frequency"]:.4f}',
          r['longest_consecutive_500_records'], f'{r["longest_run_start"]}–{r["longest_run_end"]}')
         for r in sorted(frequencies, key=lambda r: (-r['top8_count'],r['channel']))[:12]])
    lines += ['', '连续记录只证明这些采样时点持续入选，不证明两次日志之间的 499 次排序完全不变。'
              'Top-8 是 training Student 当前 labeled batch 的 w 排序诊断，不是 EMA Teacher 的推理排序，'
              '也不是 hard Top-K guidance；仅分析 Experiment A 自身，不对齐 Baseline 的 channel index。', '',
              '![BGS Top-8 temporal stability](bgs_top8_stability.png)', '',
              '## 可能的机制解释', '',
              '- 当错误区域属于预测最大连通域时，LCC 会保留它；当 Baseline 的错误区域在次级组件时，'
              'LCC 可能更有效地改善 Baseline，进而扩大最终性能差距。上文的组件 TP/FP 和 gap 变化可直接检验这一描述。',
              '- 新增远距离 FP、最大组件内错误体积和 GT overlap 描述了当前失败形态，'
              '可能与边界选择或训练期优化偏移有关；没有时间连续的同病例预测证据，不能确定形成过程。',
              '- 高频通道核心与外围 Top-8 更替可以同时存在。跨阶段漂移可能来自训练进展或 labeled batch '
              '差异；日志没有固定同一 batch，也没有记录全部 channel 分数及第 8/9 名间隔，不能区分这些解释。', '',
              '## 仍需新训练实验验证的假设', '',
              '- BGS 排序漂移是否导致最终远距离 FP 或 Dice 下降，仍需独立 seed、固定诊断 batch 与反事实训练对照。',
              '- 上一版已观察到的 BN batch 耦合是否是最终退化的因果来源，当前拓扑与排序分析无法证明。',
              '- 是否应修改 guidance、损失、特征交互或 BN，需要独立实验；本阶段没有实现或运行这些变化。', '',
              '## 产物与复现', '',
              '- `prediction_raw_vs_lcc.csv`：20×2×2=80 行，四个指标、TP/FP/FN、体积和连通域统计。',
              '- `paired_raw_vs_lcc.csv`：全部 20 例 raw/LCC 模型差及 gap 变化。',
              '- `anomaly_component_gt_overlap.csv`：两模型所有 raw 组件与 GT overlap、新增 FP 分配及空间交集。',
              '- `anomaly_lcc_confusion_effect.csv` / `anomaly_fp_distance_to_gt.csv`：LCC TP/FP/FN 效应及远距离 FP。',
              '- `bgs_topk_temporal_stability.csv`：计划时点实测/缺失状态、相邻 Jaccard、进入/退出 channel。',
              '- `bgs_channel_top8_frequency.csv`、`bgs_phase_summary.csv`、`bgs_phase_pair_comparison.csv`：全通道和阶段统计。',
              '- `masks/`：20 例 GT、两模型 raw mask 及 LCC mask，供后续独立复核。',
              '- `verification.json`：80 行覆盖、既有指标复现、旧 masks 一致、模型状态不变、保护文件哈希验证。', '',
              '```bash', f'conda run -n SSL python -u analysis/experiment_A_diagnosis_v2.py --device {args.device}', '```', '']
    (output / 'diagnosis_v2_report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, default=ROOT.parent/'Dataset/LA')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'Results/experiment_A_bgs_alpha0p1_seed42')
    parser.add_argument('--previous-diagnosis', type=Path, default=HERE/'results/experiment_A_diagnosis')
    parser.add_argument('--output-dir', type=Path, default=HERE/'results/experiment_A_diagnosis_v2')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    for name in ('data_path', 'run_dir', 'previous_diagnosis', 'output_dir'):
        setattr(args, name, getattr(args,name).resolve())
    if args.output_dir.exists():
        raise FileExistsError('Refusing to overwrite existing output: '+str(args.output_dir))
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; no inference or training started')
    # Bound CPU work for read-only metric calculations on this shared server.
    torch.set_num_threads(2)
    checkpoints = {'baseline': ROOT/'Results/seed_42/result_LA_10l/fold_0/Model_iter_27000.pth',
                   'guidance': args.run_dir/'result_LA_10l/fold_0/Model_iter_29000.pth'}
    log = args.run_dir/'result_LA_10l/fold_0/log.txt'
    for source in (ROOT/'Results/seed_42/result_LA_10l/fold_0/log.txt', log):
        if config_from_log(source)['patch_size'] != PATCH:
            raise RuntimeError('Training patch size does not match original inference')
    cases = (args.data_path/'test.list').read_text().splitlines()
    if len(cases) != 20 or len(set(cases)) != 20 or ANOMALY not in cases:
        raise RuntimeError('Expected the same 20 unique LA cases including the anomaly')
    expected = {name: read_case_metrics(args.run_dir/f'{name}_test.csv') for name in checkpoints}
    if any(set(table) != set(cases) for table in expected.values()):
        raise RuntimeError('Previous evaluation case lists differ from test.list')
    saved = args.previous_diagnosis/f'failure_visualizations/{ANOMALY}_masks.npz'
    protected = [ROOT/name for name in PROTECTED_SOURCES]
    protected += list((ROOT/'Results').rglob('*')) + list(args.previous_diagnosis.rglob('*'))
    protected += [HERE/'experiment_A_failure_diagnosis.py', args.data_path/'test.list']
    protected = [path for path in protected if path.is_file()]
    before = snapshot(protected)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='.'+args.output_dir.name+'.', dir=args.output_dir.parent))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(output/'analysis_log.txt'), logging.StreamHandler(sys.stdout)])
    logging.info('Read-only diagnosis started; staged output=%s', output)
    try:
        bgs, frequencies = bgs_stability(log, output)
        logging.info('BGS records: %s valid, missing=%s; adjacent Jaccard mean=%s',
                     bgs['actual_valid_records'], bgs['missing_iterations'], bgs['adjacent_jaccard_mean'])
        anomaly = anomaly_analysis(saved, output)
        rows, masks, invariant_models = [], {}, []
        (output/'masks').mkdir()
        with np.load(saved) as archive:
            saved_masks = {key: archive[key].astype(bool) for key in archive.files if key != 'image'}
        for name, checkpoint in checkpoints.items():
            model = load_model(checkpoint, args.device)
            original_state = {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}
            # x3 is the output of encoder.block_three, with 64 channels.
            conv_channels = [module.out_channels for module in model.encoder.block_three.modules()
                             if isinstance(module, torch.nn.Conv3d)]
            assert conv_channels and conv_channels[-1] == 64
            for index, case in enumerate(cases, 1):
                with h5py.File(resolve_case_path(args.data_path, case), 'r') as file:
                    image, gt = file['image'][:], file['label'][:].astype(bool)
                if not gt.any():
                    raise RuntimeError('Unexpected empty GT: '+case)
                with torch.inference_mode():
                    raw, probability = test_single_case(args, model, image, stride_xy=18,
                        stride_z=4, patch_size=PATCH, num_classes=2)
                raw = raw.astype(bool)
                assert np.array_equal(raw, probability > 0.5)
                final = getLargestCC(raw).astype(bool)
                assert np.all(final <= raw)
                current = {}
                for variant, mask in [('raw',raw),('lcc',final)]:
                    current[variant] = mask_metrics(case,name,variant,mask,gt,27000 if name == 'baseline' else 29000)
                    rows.append(current[variant])
                for metric in METRICS:
                    if abs(current['lcc'][metric]-expected[name][case][metric]) > 1e-6:
                        raise RuntimeError(f'Original {name} LCC metric not reproduced: {case}/{metric}')
                if case == ANOMALY:
                    assert np.array_equal(gt, saved_masks['gt_mask'])
                    assert np.array_equal(raw, saved_masks[f'{name}_raw_prediction'])
                    assert np.array_equal(final, saved_masks[f'{name}_prediction'])
                if name == 'baseline':
                    masks[case] = dict(gt_mask=gt.astype(np.uint8),baseline_raw_prediction=raw.astype(np.uint8),
                                       baseline_prediction=final.astype(np.uint8))
                else:
                    assert np.array_equal(gt,masks[case]['gt_mask'])
                    np.savez_compressed(output/'masks'/f'{case}_masks.npz', **masks.pop(case),
                        guidance_raw_prediction=raw.astype(np.uint8),guidance_prediction=final.astype(np.uint8))
                logging.info('%s %d/20 %s: raw Dice %.6f, LCC Dice %.6f', name,index,case,
                             current['raw']['dice'], current['lcc']['dice'])
            assert all(torch.equal(original_state[key],value.detach().cpu()) for key,value in model.state_dict().items())
            assert all(parameter.grad is None for parameter in model.parameters())
            invariant_models.append(name)
            del model, original_state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        assert len(rows) == 80 and not masks
        write_csv(output/'prediction_raw_vs_lcc.csv',rows)
        _, pairs = compare_cases(rows,cases,output)
        after = snapshot(protected)
        if before != after:
            raise RuntimeError('An original protected file changed during diagnosis')
        verification = dict(status='passed', prediction_rows=len(rows), cases=len(cases),
            all_20_cases_included=True, original_lcc_metrics_reproduced_within=1e-6,
            saved_anomaly_raw_lcc_gt_voxel_exact=True, models_state_unchanged=invariant_models,
            original_files_hashed=len(before), original_files_unchanged=True,
            training_performed=False, backpropagation_performed=False)
        (output/'verification.json').write_text(json.dumps(verification,indent=2))
        metadata = dict(experiment='Experiment A-D2', completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
            checkpoints={name:dict(path=str(path),sha256=sha256(path)) for name,path in checkpoints.items()},
            device=args.device, python=sys.executable, torch=torch.__version__, patch_size=PATCH,
            stride=[18,18,4], probability_threshold=0.5, connectivity=26,
            input_normalization='Existing mri_norm2.h5 image; no additional normalization',
            case_ids=cases, saved_anomaly_masks=str(saved), saved_anomaly_masks_sha256=sha256(saved),
            bgs_log=str(log), bgs_log_sha256=sha256(log), protected_original_sha256=after,
            analysis_source_sha256=sha256(Path(__file__)))
        (output/'diagnosis_v2_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
        report(output,args,rows,pairs,anomaly,bgs,frequencies,verification)
        logging.info('Verification passed: all 20 cases; 80 prediction rows; %d original files unchanged',len(before))
        os.replace(output,args.output_dir)
    except BaseException:
        logging.exception('Diagnosis did not complete; partial output retained at %s',output)
        raise
    print(json.dumps(dict(status='complete',output_dir=str(args.output_dir)),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
