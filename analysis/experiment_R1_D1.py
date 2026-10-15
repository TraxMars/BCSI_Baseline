"""R1-D1: presealed frozen-model semantic, oracle-direction and representation diagnosis.

Only this analysis creates an x3 leaf for autograd; oracle information never
enters Phase A. All 24 patches/all folds are sealed before Receiver evaluation.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import ast
import copy
import csv
import datetime
import inspect
import io
import json
import logging
import os
import shutil
from pathlib import Path
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import experiment_R1_S0 as s
u, r0, r05 = s.u, s.r0, s.r05

SEED = 42
ALPHA = .02
EPS = 1e-6
LOGIT_ATOL = 2e-5
LOSS_ATOL = 2e-6
FD_ATOL = 2e-5
FD_RTOL = .05
S0 = ROOT / 'analysis/results/experiment_R1_S0/run_20261013_2055'
BASE = ROOT / 'analysis/results/experiment_R1_D1'
CODE = [Path(__file__).resolve(), HERE / 'test_experiment_R1_D1.py']
PROTOCOL = dict(seed=SEED, alpha=ALPHA, mapping='max', budget=.02, cap=.10,
    q='clip(abs(best_fg_cosine-best_bg_cosine)/0.2,0,1); skip Q<=0.001',
    cluster_k_per_class=4, cluster_limit_per_donor_class=512,
    cluster_n_init=10, cluster_max_iter=300, cluster_tol=1e-4,
    local_limit_per_donor_class=64, equal_donor_sampling=True,
    sampling='min(cap,minimum available count across seven donors), no replacement per class',
    random_matching_repetitions=20, bootstrap_repetitions=10000,
    fixed_smoke_and_visuals=[[0,0],[3,0]], finite_difference_patches=[[0,0],[3,0]],
    finite_difference_methods=['true_boundary','nonboundary'], finite_difference_t=[.001,.01],
    finite_difference_centered_always=True, finite_difference_tolerance=dict(atol=FD_ATOL,rtol=FD_RTOL),
    historical_logit_atol=LOGIT_ATOL,historical_loss_atol=LOSS_ATOL,
    pseudo_grid_class='majority of native p>0.5 voxels, foreground iff fraction>=0.5',
    GT_pure_sides='foreground fraction>=.75 or <=.25; mixed excluded from class correctness only',
    gradient_zero='norm exactly zero; undefined cosine stored as null/mask, never fake zero',
    energy_weight='squared actual residual L2 per spatial location',
    phase_separation='all folds/all candidates sealed before any Receiver-role GT read',
    evaluation_order='finish D1-A/B for all 24 patches, then evaluate presealed D1-C candidates; reuse saved Oracle gradient',
    no_independent_validation=True, oracle_never_used_for_candidates=True)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@contextmanager
def computation_precision(strict):
    """Keep native replay and Oracle arithmetic separate; restore on failure."""
    old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    api=hasattr(torch,'set_float32_matmul_precision')
    previous=torch.get_float32_matmul_precision() if api else None
    try:
        if api:torch.set_float32_matmul_precision('highest' if strict else 'high')
        torch.backends.cuda.matmul.allow_tf32=not strict
        torch.backends.cudnn.allow_tf32=not strict
        if strict and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise RuntimeError('Strict FP32 TF32 flags did not take effect')
        yield dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                   float32_matmul_precision_api_supported=api,
                   precision='strict_fp32' if strict else 'historical_tf32')
    finally:
        if api:torch.set_float32_matmul_precision(previous)
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old


def resolve_phase_a_seal(seal, metadata, audit):
    """Only historical SOURCE paths may resolve to authenticated old snapshots.

    Original seal bytes and every candidate path/hash remain untouched. The
    source move is recorded as a post-failure provenance resolution, never a
    replacement or backdated Phase-A seal.
    """
    snapshots={r['original_path']:r for r in audit['source_snapshots']}
    result=copy.deepcopy(seal)
    for item in result['artifacts']:
        path=item['path']
        if path in metadata['code_sha256']:
            snap=snapshots.get(path)
            if not snap or snap['sha256']!=item['sha256'] or r0.sha256(snap['snapshot_path'])!=item['sha256']:
                raise RuntimeError('Cannot authenticate Phase-A source snapshot')
            item['original_source_path']=path
            item['path']=snap['snapshot_path']
    return result


def verify_phase_a_code_unchanged(old_source,new_source):
    names=('phase_a_patch','match_bank','build_banks','save_candidate','load_candidate',
           'create_output','semantic_metrics','alignment','alignment_stats','q_groups','group_case')
    def definitions(source):
        return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(source).body
                if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
    a,b=definitions(old_source),definitions(new_source)
    if any(a[n]!=b[n] for n in names):raise RuntimeError('Resume changed a Phase-A/scientific helper')
    return list(names)


def save_npz(path, **arrays):
    """Lossless FP32, deflate level 1; no modification of historical archives."""
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name, value in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(value), allow_pickle=False)
            z.writestr(name+'.npy', buffer.getvalue())


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def create_output(path):
    p = Path(path).resolve()
    if p.parent != BASE or not p.name.startswith('run_'):
        raise ValueError('Use fresh experiment_R1_D1/run_TIMESTAMP')
    p.mkdir(parents=True, exist_ok=False)
    return p


def protected_manifest():
    result = {}
    for parent, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in ('.git','__pycache__','.pytest_cache')
                         and (Path(parent)/d).resolve() != BASE)
        for name in sorted(files):
            p = Path(parent)/name
            if p.is_file() and p.resolve() not in CODE and p.suffix not in ('.pyc','.pyo'):
                result[str(p.relative_to(ROOT))] = r0.sha256(p)
    return result


class Reader(s.FoldReader):
    def __init__(self, history):
        super().__init__(history)
        self.global_sealed = False

    def read(self, case, patch, field):
        if field == 'label' and case == self.receiver and not self.global_sealed:
            raise RuntimeError('All-fold Phase A seal required before Receiver GT')
        value = super().read(case, patch, field)
        self.events[-1].update(utc=now(), all_fold_sealed=self.global_sealed)
        return value

    def unlock(self, seal):
        if not seal.get('phaseA_complete') or not seal.get('artifacts'):
            raise RuntimeError('Incomplete global seal')
        for a in seal['artifacts']:
            if r0.sha256(a['path']) != a['sha256']:
                raise RuntimeError('Changed sealed candidate: '+a['path'])
        self.global_sealed = True

    def evaluation_fold(self, index, receiver, donors):
        if not self.global_sealed:
            raise RuntimeError('No Receiver evaluation before global seal')
        self.start_fold(index, receiver, donors)
        self.sealed = True


def match_bank(x, bank, classes, mean, scale):
    """No GT API. Best cosine per class; Q measures inter-class margin."""
    if set(np.asarray(classes).tolist()) != {0,1}:
        raise ValueError('Both donor class identities required')
    v = x.reshape(x.shape[0],-1).T.astype(np.float64)
    z = (v-mean)/scale
    b = (bank.astype(np.float64)-mean)/scale
    vn, bn = np.linalg.norm(z,axis=1), np.linalg.norm(b,axis=1)
    if np.any(bn <= EPS):
        raise ValueError('Undefined donor bank cosine norm')
    z /= np.maximum(vn[:,None],EPS)
    b /= bn[:,None]
    sims = np.empty((len(v),2),np.float64)
    winners = np.empty((len(v),2),np.int64)
    for start in range(0,len(v),2048):
        end = min(start+2048,len(v)); score = z[start:end] @ b.T
        for c in (0,1):
            ids = np.flatnonzero(classes==c)
            win = np.argmax(score[:,ids],axis=1)
            winners[start:end,c] = ids[win]
            sims[start:end,c] = score[np.arange(end-start),ids[win]]
    chosen = sims.argmax(1)
    winner = winners[np.arange(len(v)),chosen]
    margin = np.abs(sims[:,0]-sims[:,1])
    q = np.clip(margin/.2,0,1)
    valid = (vn>EPS) & (np.linalg.norm(v,axis=1)>EPS) & (q>s.Q_MIN)
    q[~valid] = 0
    shape = x.shape[1:]
    return dict(target=bank[winner].T.reshape(x.shape).astype(np.float32),
        chosen=chosen.reshape(shape).astype(np.uint8), q=q.reshape(shape).astype(np.float32),
        valid=valid.reshape(shape), winner=winner.reshape(shape),
        fg_similarity=sims[:,0].reshape(shape).astype(np.float32),
        bg_similarity=sims[:,1].reshape(shape).astype(np.float32),
        margin=margin.reshape(shape).astype(np.float32))


def build_banks(xs, libs, donor, receiver, ids, fold):
    if receiver in libs or len(libs)!=7:
        raise ValueError('LOCO contamination')
    pools, locations = {}, {}
    for case, lib in libs.items():
        for c,side in enumerate(('fg','bg')):
            vv,ll = [],[]
            for p in range(3):
                idx = np.flatnonzero(lib['grids'][p]['boundary_'+side])
                vv.append(xs[case,p].reshape(xs[case,p].shape[0],-1).T[idx])
                ll.extend((ids.index(case),p,int(i)) for i in idx)
            pools[case,c] = np.concatenate(vv)
            locations[case,c] = np.array(ll,dtype=np.int64)
    banks,stats = {},[]
    for kind,cap in (('multi',512),('local',64)):
        raw,cls,src = [],[],[]
        for c in (0,1):
            n = min([cap]+[len(pools[case,c]) for case in libs])
            if n<4:
                raise ValueError('Insufficient fixed donor boundary pool; no threshold relaxation')
            chunks=[]
            for case in libs:
                seed=SEED+fold*100003+ids.index(case)*1009+c*97+(31000001 if kind=='multi' else 41000003)
                take=torch.randperm(len(pools[case,c]),generator=s.cpu_generator(seed))[:n].numpy()
                chunks.append(pools[case,c][take])
                if kind=='local': src.extend(locations[case,c][take].tolist())
                stats.append(dict(fold=fold,receiver=receiver,donor=case,representation=kind,donor_class=c,
                    available=len(pools[case,c]),sampled=n,seed=seed))
            vectors=np.concatenate(chunks)
            if kind=='multi':
                z=(vectors.astype(np.float64)-donor['mean'])/donor['scale']
                km=KMeans(n_clusters=4,n_init=10,max_iter=300,tol=1e-4,random_state=SEED+fold*1009+c)
                km.fit(z)
                counts=np.bincount(km.labels_,minlength=4)
                if np.any(counts==0): raise ValueError('Empty fixed KMeans cluster')
                centers=(km.cluster_centers_*donor['scale']+donor['mean']).astype(np.float32)
                raw.extend(centers);cls.extend([c]*4)
                for j in range(4):
                    stats.append(dict(fold=fold,receiver=receiver,representation=kind,donor_class=c,
                        cluster=j,cluster_count=int(counts[j]),iterations=int(km.n_iter_),
                        inertia=float(km.inertia_),prototype_norm=float(np.linalg.norm(centers[j]))))
            else:
                raw.extend(vectors);cls.extend([c]*len(vectors))
        banks[kind]=dict(raw=np.array(raw,np.float32),classes=np.array(cls,np.uint8),
                         source=np.array(src,np.int64) if src else np.empty((0,3),np.int64))
    return banks,stats


def gradient_at_x3(model, features, y, inner, outer):
    if any(p.requires_grad for p in model.parameters()) or model.training:
        raise ValueError('Model must be frozen and eval')
    with torch.enable_grad():
        leaf=features[2].detach().clone().requires_grad_(True)
        ff=list(features);ff[2]=leaf
        loss=u.loss_values(model.decoder(ff),y,inner,outer)['boundary_ce']
        g=torch.autograd.grad(loss,leaf,create_graph=False,retain_graph=False)[0]
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError('Parameter gradient unexpectedly written')
    r0.finite(g.detach().cpu().numpy(),'oracle gradient')
    return g[0].detach().cpu().numpy(),float(loss.detach())


def alignment(g, delta):
    g=g.astype(np.float64);d=delta.astype(np.float64)
    gn=np.linalg.norm(g,axis=0);dn=np.linalg.norm(d,axis=0)
    dot=np.sum(g*d,axis=0)
    valid=(gn>0)&(dn>0)
    a=np.zeros(gn.shape,np.float64)  # storage placeholder, ONLY valid mask is evaluated
    a[valid]=-dot[valid]/(gn[valid]*dn[valid]+EPS)
    return a,valid,dot,gn,dn


def alignment_stats(g,delta,domain):
    a,v,dot,gn,dn=alignment(g,delta);active=v&domain
    n=int(active.sum());energy=dn[active]**2
    return dict(domain_positions=int(domain.sum()),defined_positions=n,
        zero_gradient_positions=int(np.count_nonzero(domain&(gn==0))),
        zero_residual_positions=int(np.count_nonzero(domain&(dn==0))),
        alignment_mean=float(a[active].mean()) if n else None,
        alignment_median=float(np.median(a[active])) if n else None,
        aligned_fraction=float(np.mean(a[active]>0)) if n else None,
        opposed_fraction=float(np.mean(a[active]<0)) if n else None,
        energy_weighted_alignment=float(np.sum(a[active]*energy)/energy.sum()) if n else None,
        directional_derivative=float(dot[domain].sum()),
        undefined_reason=None if n else 'zero_gradient_or_zero_actual_residual_in_domain')


def q_groups(q):
    return [(str(i), (q>=lo)&((q<hi) if hi<1 else (q<=hi)))
            for i,(lo,hi) in enumerate(((0,.25),(.25,.5),(.5,.75),(.75,1.)))]


def mean_or_none(a):
    return float(np.mean(a)) if np.size(a) else None


def semantic_metrics(m,mask,gt,pred0,pred1,ce_weighted_gain):
    fraction=r05.block_count(gt.astype(bool))/64.
    pure=(fraction>=.75)|(fraction<=.25)
    truth=fraction>=.75
    pseudo=r05.block_count(pred0)/64.>=.5
    guessed=m['chosen']==0
    correct=guessed==truth
    teacher_correct=pseudo==truth
    errors=pred0!=gt.astype(bool)
    fixed=errors&(pred1==gt.astype(bool))
    harmed=~errors&(pred1!=gt.astype(bool))
    groups=[]; conflicts=[]; qrows=[]; types=[]
    for name,domain in [('all',np.ones(mask.shape,bool)),('high_D',mask),
                        ('high_D_teacher_correct',mask&pure&teacher_correct),
                        ('high_D_teacher_wrong',mask&pure&~teacher_correct)]:
        evaluate=domain&pure&m['valid']
        groups.append(dict(domain=name,positions=int(domain.sum()),pure_positions=int((domain&pure).sum()),
            mixed_positions=int((domain&~pure).sum()),mixed_fraction=mean_or_none((~pure)[domain]),
            evaluated_positions=int(evaluate.sum()),accuracy=mean_or_none(correct[evaluate]),
            q_mean=mean_or_none(m['q'][domain]),q_saturated_fraction=mean_or_none((m['q']==1)[domain]),
            foreground_accuracy=mean_or_none(correct[evaluate&truth]),
            background_accuracy=mean_or_none(correct[evaluate&~truth]),
            foreground_called_background=int((evaluate&truth&~guessed).sum()),
            background_called_foreground=int((evaluate&~truth&guessed).sum()),
            undefined_reason=None if evaluate.any() else 'no_valid_pure_grid'))
    for name,agree in [('agreement',guessed==pseudo),('conflict',guessed!=pseudo)]:
        d=mask&agree; e=d&pure&m['valid']; lifted=r05.lift_grid(d)
        conflicts.append(dict(domain=name,positions=int(d.sum()),pure_positions=int(e.sum()),
            donor_accuracy=mean_or_none(correct[e]),teacher_accuracy=mean_or_none(teacher_correct[e]),
            original_voxel_error_fraction=mean_or_none(errors[lifted]),
            corrected=int((fixed&lifted).sum()),harmed=int((harmed&lifted).sum())))
    for name,part in q_groups(m['q']):
        d=mask&part;e=d&pure&m['valid'];lift=r05.lift_grid(d)
        qrows.append(dict(group=name,positions=int(d.sum()),pure_positions=int((d&pure).sum()),
            evaluated_positions=int(e.sum()),accuracy=mean_or_none(correct[e]),q_mean=mean_or_none(m['q'][e]),
            original_error_fraction=mean_or_none(errors[lift]),original_correct_fraction=mean_or_none((~errors)[lift]),
            corrected=int((fixed&lift).sum()),harmed=int((harmed&lift).sum()),
            boundary_gain_contribution=float(ce_weighted_gain[lift].sum()),
            undefined_reason=None if e.any() else 'no_valid_pure_grid'))
    for name,tc,mc in [('A',True,True),('B',True,False),('C',False,True),('D',False,False)]:
        d=mask&pure&m['valid']&(teacher_correct==tc)&(correct==mc);lift=r05.lift_grid(d)
        types.append(dict(type=name,positions=int(d.sum()),corrected=int((fixed&lift).sum()),
            harmed=int((harmed&lift).sum()),net=int((fixed&lift).sum()-(harmed&lift).sum()),
            boundary_gain_contribution=float(ce_weighted_gain[lift].sum())))
    e=mask&pure&m['valid']; auroc=None
    if e.any() and np.unique(correct[e]).size==2: auroc=float(roc_auc_score(correct[e],m['q'][e]))
    qrows.append(dict(group='continuous',positions=int(mask.sum()),pure_positions=int(e.sum()),
        auroc_q_correctness=auroc,accuracy=mean_or_none(correct[e]),q_mean=mean_or_none(m['q'][e]),
        q_saturated_fraction=mean_or_none((m['q'][e]==1)),
        brier_descriptive=float(np.mean((m['q'][e]-correct[e])**2)) if e.any() else None,
        undefined_reason=None if auroc is not None else 'single_correctness_class_or_no_valid_pure_grid'))
    return groups,conflicts,qrows,types


def group_case(rows, keys):
    """First repetitions within patch; then patches within case. Never voxel bootstrap."""
    numeric=sorted({k for r in rows for k,v in r.items() if isinstance(v,(int,float,np.number)) and k not in keys+['patch_index','fold','repetition']})
    buckets=defaultdict(list)
    for r in rows: buckets[tuple(r.get(k) for k in ['case_id','patch_index']+keys)].append(r)
    patches=[]
    for key,rr in buckets.items():
        row=dict(zip(['case_id','patch_index']+keys,key))
        for n in numeric: row[n]=s.mean_defined([r.get(n) for r in rr])
        patches.append(row)
    buckets=defaultdict(list)
    for r in patches: buckets[tuple(r.get(k) for k in ['case_id']+keys)].append(r)
    cases=[]
    for key,rr in buckets.items():
        row=dict(zip(['case_id']+keys,key));row['n_patches']=len(rr)
        for n in numeric:
            row[n]=s.mean_defined([r.get(n) for r in rr])
            row[n+'_defined_patches']=sum(r.get(n) is not None for r in rr)
        cases.append(row)
    return cases,numeric


def summarize_tables(tables):
    allcases=[];summary={}
    for table,(rows,keys) in tables.items():
        cases,numeric=group_case(rows,keys)
        allcases.extend(dict(table=table,**r) for r in cases)
        groups=defaultdict(list)
        for r in cases: groups[tuple(r.get(k) for k in keys)].append(r)
        summary[table]=[dict(group=dict(zip(keys,key)),metrics={n:s.summarize([r[n] for r in rr]) for n in numeric}) for key,rr in groups.items()]
    return allcases,summary


def save_candidate(path, logits, x, modified, mask, matching, stats):
    array=logits[0].detach().cpu().numpy()
    pred=(torch.softmax(logits,1)[0,1]>.5).cpu().numpy()
    payload=dict(logits=array,prediction_bits=np.packbits(pred),modified_selected=modified[:,mask],
        mask=mask,selected_class=matching['chosen'],q=matching['q'],valid=matching['valid'],
        fg_similarity=matching['fg_similarity'],bg_similarity=matching['bg_similarity'])
    if 'winner' in matching: payload['winner']=matching['winner']
    save_npz(path,**payload)


@torch.no_grad()
def phase_a_patch(model,image,donor,banks,oldmaps,oldplan,out,ci,patch):
    device=next(model.parameters()).device
    features=model.encoder(torch.from_numpy(image)[None,None].to(device))
    original=model(torch.from_numpy(image)[None,None].to(device))
    baseline=model.decoder(features)
    if not torch.equal(original,baseline): raise AssertionError('Forward replay mismatch')
    x=features[2][0].cpu().numpy()
    if not np.array_equal(x,oldmaps['x3']): raise AssertionError('Historical x3 mismatch')
    p,flips,_=r0.teacher_predictions(model,image,device)
    for a,b in ((p,oldmaps['probability']),(flips,oldmaps['aligned_flip_probabilities'])):
        if not np.array_equal(a,b): raise AssertionError('Historical probabilities mismatch')
    risk=r05.map_to_grid(r05.direction_maps(p,flips)['Dxyz'],x.shape[1:],'max')
    mask=s.risk_mask(risk,ci,patch)
    if not np.array_equal(risk,oldmaps['risk_max']) or not np.array_equal(mask,oldmaps['mask_max']):
        raise AssertionError('Historical risk/mask mismatch')
    selfp,_=s.self_prototypes(x,p)
    protos=dict(donor['prototypes'],self=selfp)
    matches={k:s.match(x,protos[k],donor['mean'],donor['scale']) for k in ('boundary','nonboundary','self')}
    for kind,m in matches.items():
        for key in ('chosen','q','valid','fg_similarity','bg_similarity'):
            if not np.array_equal(m[key],oldmaps[kind+'_'+key]): raise AssertionError('Historical matching differs '+kind+'/'+key)
    true=matches['boundary']
    swap=s.match(x,protos['boundary'][::-1].copy(),donor['mean'],donor['scale'])
    if not np.array_equal(true['q'],swap['q']) or not np.array_equal(true['target'][:,true['q']>0],swap['target'][:,true['q']>0]):
        raise AssertionError('Label-swap identity failed')
    for k,b in banks.items(): matches[k]=match_bank(x,b['raw'],b['classes'],donor['mean'],donor['scale'])
    oldbase=s.read_prediction(oldplan['no_correction',None]['path'])[0]
    if not np.array_equal(baseline[0].cpu().numpy(),oldbase): raise AssertionError('S0 baseline logits mismatch')
    _,true_stats=s.apply_residual(x,true['target'],mask,true['q'],ALPHA)
    specs=[('no_correction','boundary',None,False),('true_boundary','boundary',None,False)]
    for method,kind in [('opposite_match','boundary'),('nonboundary','nonboundary'),('self','self'),('multi','multi'),('local','local')]:
        specs.extend([(method,kind,None,False),(method+'_energy_matched',kind,None,True)])
    specs.extend(('random_match','boundary',rep,False) for rep in range(20))
    rows=[];local_rows=[]
    for method,kind,rep,equal in specs:
        matching=dict(matches[kind]);target=matching['target']
        if method.startswith('opposite_match'):
            matching['chosen']=1-true['chosen'];target=protos['boundary'][matching['chosen']].transpose(3,0,1,2)
        if method=='random_match':
            seed=SEED+ci*100003+patch*9176+rep*101+220000013
            labels=torch.randint(0,2,mask.shape,generator=s.cpu_generator(seed)).numpy().astype(np.uint8)
            matching['chosen']=labels;target=protos['boundary'][labels].transpose(3,0,1,2)
        modified,stats=s.apply_residual(x,target,mask,matching['q'],0 if method=='no_correction' else ALPHA,
            true_stats['residual_l2'] if equal else None)
        logits=model.decoder(s.replay_features(features,modified))
        old=oldplan.get((method,rep));maxdiff=None
        if old:
            oldz=s.read_prediction(old['path'],oldbase)[0]
            maxdiff=float(np.max(np.abs(logits[0].cpu().numpy()-oldz)))
            if maxdiff>LOGIT_ATOL: raise AssertionError('S0 modified logits exceed tolerance '+method)
            for k in ('residual_l2','relative_l2'):
                if abs(stats[k]-old[k])>1e-7: raise AssertionError('S0 actual residual mismatch')
        path=out/(method+'_'+str(rep)+'.npz')
        save_candidate(path,logits,x,modified,mask,matching,stats)
        rows.append(dict(fold=ci,patch_index=patch,method=method,repetition=rep,path=str(path),
            historical_path=old['path'] if old else None,historical_logit_max_difference=maxdiff,**stats))
    m=matches['local']; bank=banks['local']
    for index in np.flatnonzero(mask):
        win=int(m['winner'].ravel()[index]);src=bank['source'][win]
        local_rows.append(dict(fold=ci,patch_index=patch,receiver_grid_index=int(index),source_case_index=int(src[0]),
            source_patch_index=int(src[1]),source_grid_index=int(src[2]),matched_class=int(m['chosen'].ravel()[index]),
            similarity=float(max(m['fg_similarity'].ravel()[index],m['bg_similarity'].ravel()[index])),
            feature_norm=float(np.linalg.norm(bank['raw'][win])),q=float(m['q'].ravel()[index])))
    save_npz(out/'image_only_maps.npz',image=image,x3=x,probability=p,risk=risk,mask=mask,
        **{kind+'_'+key:m[key] for kind,m in matches.items() for key in ('chosen','q','valid','fg_similarity','bg_similarity')})
    return rows,local_rows


def load_candidate(path, x):
    with np.load(path) as f:
        a={k:f[k] for k in f.files}
    changed=x.copy();changed[:,a['mask']]=a['modified_selected']
    pred=np.unpackbits(a['prediction_bits'])[:np.prod(a['logits'].shape[1:])].reshape(a['logits'].shape[1:]).astype(bool)
    matching=dict(chosen=a['selected_class'],q=a['q'],valid=a['valid'],fg_similarity=a['fg_similarity'],bg_similarity=a['bg_similarity'])
    probability=torch.softmax(torch.from_numpy(a['logits']),0)[1].numpy()
    return a['logits'],probability,pred,changed,a['mask'],matching


def fd_check(model,features,y,inner,outer,delta,expected,base_loss):
    rows=[]
    with torch.no_grad():
        for t in (.001,.01):
            losses=[]
            for sign in (1,-1):
                ff=list(features)
                ff[2]=features[2]+sign*t*torch.from_numpy(delta.astype(np.float32))[None].to(features[2].device)
                losses.append(float(u.loss_values(model.decoder(ff),y,inner,outer)['boundary_ce']))
            forward=(losses[0]-base_loss)/t
            center=(losses[0]-losses[1])/(2*t)
            tol=FD_ATOL+FD_RTOL*abs(expected)
            rows.append(dict(t=t,autograd_directional_derivative=expected,forward_difference=forward,
                central_difference=center,forward_absolute_error=abs(forward-expected),
                central_absolute_error=abs(center-expected),tolerance=tol,
                forward_pass=abs(forward-expected)<=tol,central_pass=abs(center-expected)<=tol,
                numerical_note='FP32 decoder/features and CE; FP64 reduction; fixed t, no parameter search'))
    return rows


@torch.no_grad()
def historical_features(model, parent):
    with np.load(parent/'image_only_maps.npz') as z:
        image=z['image'];x=z['x3'];mask=z['mask']
    with computation_precision(False):
        image_tensor=torch.from_numpy(image)[None,None].to(next(model.parameters()).device)
        fs=model.encoder(image_tensor);baseline=model.decoder(fs)
        if not torch.equal(baseline,model(image_tensor)):raise AssertionError('Original/replay mismatch at resume')
    if not np.array_equal(x,fs[2][0].cpu().numpy()):raise AssertionError('Sealed x3 differs at resume')
    z0,p0,pred0,_,_,_=load_candidate(parent/'no_correction_None.npz',x)
    if not np.array_equal(z0,baseline[0].cpu().numpy()):raise AssertionError('Sealed TF32 baseline changed')
    return fs,image,x,mask,z0,p0,pred0


def oracle_pair(model,features,ctx,path):
    device=features[2].device
    y,inner,outer=(ctx[k].to(device) for k in ('y','inner','outer'))
    if path.exists():
        with np.load(path) as a:result={k:a[k] for k in a.files}
        if str(result['precision'].item())!='strict_fp32':raise RuntimeError('Oracle cache has wrong precision')
        return result
    with computation_precision(False):native,native_loss=gradient_at_x3(model,features,y,inner,outer)
    with computation_precision(True) as settings:strict,strict_loss=gradient_at_x3(model,features,y,inner,outer)
    path.parent.mkdir(parents=True,exist_ok=True)
    result=dict(gradient=strict,native_gradient=native,boundary_loss_strict_fp32=np.array(strict_loss),
        boundary_loss_historical_tf32=np.array(native_loss),precision=np.array('strict_fp32'),
        matmul_allow_tf32=np.array(settings['matmul_allow_tf32']),cudnn_allow_tf32=np.array(settings['cudnn_allow_tf32']))
    save_npz(path,**result)
    return result


def strict_preflight(model,reader,records,out,old_fd):
    rows=list(old_fd);cache_root=out/'strict_fp32_oracle'
    for ci,patch in PROTOCOL['finite_difference_patches']:
        case=reader.case_order[ci]
        reader.evaluation_fold(ci,case,[c for c in reader.case_order if c!=case])
        gt=reader.read(case,patch,'label');ctx=s.evaluation_context(gt)
        rr=[r for r in records if r['case_id']==case and r['patch_index']==patch]
        parent=Path(rr[0]['path']).parent
        fs,_,x,_,_,_,_=historical_features(model,parent)
        cache=oracle_pair(model,fs,ctx,cache_root/f'fold_{ci}'/f'patch_{patch}'/'oracle_gradient.npz')
        y,inner,outer=(ctx[k].to(fs[2].device) for k in ('y','inner','outer'))
        for method in PROTOCOL['finite_difference_methods']:
            record=next(r for r in rr if r['method']==method)
            _,_,_,modified,_,_=load_candidate(record['path'],x)
            delta=modified.astype(np.float64)-x.astype(np.float64)
            for strict in (False,True):
                precision='strict_fp32' if strict else 'historical_tf32'
                g=cache['gradient'] if strict else cache['native_gradient']
                base=float(cache['boundary_loss_'+precision])
                d=float(np.sum(g.astype(np.float64)*delta))
                with computation_precision(strict) as settings:
                    checks=fd_check(model,fs,y,inner,outer,delta,d,base)
                rows.extend(dict(case_id=case,fold=ci,patch_index=patch,method=method,repetition=None,
                    precision=precision,record_stage='resume_preflight',baseline_loss=base,
                    strict_flags_verified=not settings['matmul_allow_tf32'] and not settings['cudnn_allow_tf32'] if strict else None,
                    **r) for r in checks)
                logging.info('Precision preflight fold=%d method=%s mode=%s center(.01) error=%.8g pass=%s',
                    ci,method,precision,checks[-1]['central_absolute_error'],checks[-1]['central_pass'])
        r0.write_csv(out/'finite_difference_verification.csv',rows)
        r0.write_csv(out/'data_access_log.csv',reader.events)
    primary=[r for r in rows if r.get('record_stage')=='resume_preflight' and r.get('precision')=='strict_fp32' and r['t']==.01]
    if len(primary)!=4 or not all(r['central_pass'] for r in primary):
        raise RuntimeError('Strict FP32 fixed .01 central checks failed; no full Phase B evaluation')
    r0.write_json(out/'strict_fp32_preflight.json',dict(completed_utc=now(),passed=True,
        primary_checks=4,primary_t=.01,small_t=.001,atol=FD_ATOL,rtol=FD_RTOL,
        all_failures_retained=True,rows=len(rows),no_t_or_tolerance_change=True))
    return rows


def phase_b_strict_patch(model,reader,case,ci,patch,rows,out,tables,history_metrics,stage='AB'):
    if not reader.global_sealed:raise RuntimeError('Candidates not authenticated')
    gt=reader.read(case,patch,'label');ctx=s.evaluation_context(gt)
    parent=Path(rows[0]['path']).parent
    features,image,x,mask,z0,p0,pred0=historical_features(model,parent)
    cache=oracle_pair(model,features,ctx,out/'strict_fp32_oracle'/f'fold_{ci}'/f'patch_{patch}'/'oracle_gradient.npz')
    g=cache['gradient'];native_g=cache['native_gradient']
    strict_lb=float(cache['boundary_loss_strict_fp32'])
    native_gpu_lb=float(cache['boundary_loss_historical_tf32'])
    with torch.no_grad():ce0=F.cross_entropy(torch.from_numpy(z0)[None],ctx['y'],reduction='none')[0].numpy()
    iw,ow=ctx['inner'][0].numpy(),ctx['outer'][0].numpy()
    weights=.5*iw/iw.sum()+.5*ow/ow.sum();native_lb=float((ce0*weights).sum())
    if abs(native_lb-native_gpu_lb)>LOSS_ATOL:raise AssertionError('Native CPU/GPU CE disagrees')
    pure=(ctx['fraction']>=.75)|(ctx['fraction']<=.25)
    teacher_correct=(r05.block_count(pred0)/64.>=.5)==(ctx['fraction']>=.75)
    device=features[2].device;y,inner,outer=(ctx[k].to(device) for k in ('y','inner','outer'))
    if stage=='AB':
        a=g.astype(np.float64).ravel();b=native_g.astype(np.float64).ravel()
        tables.setdefault('precision',[]).append(dict(case_id=case,fold=ci,patch_index=patch,
            strict_baseline_boundary_ce=strict_lb,historical_baseline_boundary_ce=native_lb,
            baseline_precision_difference=strict_lb-native_lb,
            gradient_cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)),
            relative_gradient_difference=float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12)),
            strict_gradient_l2=float(np.linalg.norm(a)),historical_gradient_l2=float(np.linalg.norm(b)),
            strict_gradient_nonzero_fraction=float(np.mean(g!=0)),
            matmul_allow_tf32=bool(cache['matmul_allow_tf32']),cudnn_allow_tf32=bool(cache['cudnn_allow_tf32'])))
    for record in rows:
        method,rep=record['method'],record['repetition']
        if stage=='C' and method=='no_correction':continue
        common=dict(case_id=case,fold=ci,patch_index=patch,method=method,repetition=rep)
        z,p,pred,modified,selected,m=load_candidate(record['path'],x)
        delta=modified.astype(np.float64)-x.astype(np.float64)
        if np.any(delta[:,~selected]!=0):raise AssertionError('Sealed residual escaped mask')
        metrics=s.evaluate_prediction(z,p,pred,gt,pred0,p0,selected,m,ctx)
        metrics['boundary_ce_gain']=native_lb-metrics['boundary_ce']
        old=history_metrics.get((case,patch,method,rep));olddiff=None
        if old:
            olddiff=abs(metrics['boundary_ce']-float(old['boundary_ce']))
            if olddiff>LOSS_ATOL or metrics['corrected']!=int(old['corrected']) or metrics['harmed']!=int(old['harmed']):
                raise AssertionError('Historical TF32 loss or native decisions changed')
        tables['correction'].append(dict(common,**metrics,residual_l2=record['residual_l2'],relative_l2=record['relative_l2'],
            q_mean=record['q_mean'],prediction_precision='historical_tf32',historical_boundary_ce_difference=olddiff))
        with torch.no_grad():ce1=F.cross_entropy(torch.from_numpy(z)[None],ctx['y'],reduction='none')[0].numpy()
        if method!='no_correction':
            sem,con,qr,types=semantic_metrics(m,selected,gt,pred0,pred,(ce0-ce1)*weights)
            for name,values in [('semantic',sem),('conflict',con),('q',qr),('types',types)]:
                tables[name].extend(dict(common,**v) for v in values)
        domains={'all':np.ones(mask.shape,bool),'high_D':mask,'low_D':~mask,
            'teacher_correct_pure':mask&pure&teacher_correct,'teacher_wrong_pure':mask&pure&~teacher_correct,
            'mixed':mask&~pure,'high_Q':mask&(m['q']>=.75),'low_Q':mask&(m['q']<.25),
            'contains_voxel_errors':mask&(r05.block_count(pred0!=gt.astype(bool))>0),
            'no_voxel_errors':mask&(r05.block_count(pred0!=gt.astype(bool))==0)}
        for domain,d in domains.items():
            tables['alignment'].append(dict(common,domain=domain,oracle_precision='strict_fp32',**alignment_stats(g,delta,d)))
        with torch.no_grad(),computation_precision(True):
            # Same precision as G; never overwrite/use these logits as candidates.
            strict_after=float(u.loss_values(model.decoder(s.replay_features(features,modified)),y,inner,outer)['boundary_ce'])
        directional=float(np.sum(g.astype(np.float64)*delta))
        strict_actual=strict_after-strict_lb
        historical_actual=metrics['boundary_ce']-native_lb
        norm=float(np.linalg.norm(delta))
        tables['derivative'].append(dict(common,oracle_precision='strict_fp32',
            predicted_loss_change=directional,actual_loss_change=strict_actual,
            strict_fp32_baseline_loss=strict_lb,strict_fp32_modified_loss=strict_after,
            historical_tf32_baseline_loss=native_lb,historical_tf32_modified_loss=metrics['boundary_ce'],
            historical_actual_loss_change=historical_actual,
            native_gradient_directional_derivative=float(np.sum(native_g.astype(np.float64)*delta)),
            sign_agreement=bool(np.sign(directional)==np.sign(strict_actual)),
            historical_sign_agreement=bool(np.sign(directional)==np.sign(historical_actual)),
            absolute_error=abs(strict_actual-directional),historical_absolute_error=abs(historical_actual-directional),
            sign_above_precision_floor=bool(min(abs(directional),abs(strict_actual))>LOSS_ATOL),
            residual_l2=norm,normalized_directional_derivative=directional/norm if norm else None,
            gradient_l2=float(np.linalg.norm(g.astype(np.float64))),gradient_nonzero_fraction=float(np.mean(g!=0)),
            historical_loss_difference=olddiff,undefined_reason=None if norm else 'zero_residual'))
        if [ci,patch] in PROTOCOL['fixed_smoke_and_visuals'] and method in ('true_boundary','nonboundary','multi','local'):
            a,valid,_,_,_=alignment(g,delta)
            ap=out/'strict_fp32_oracle'/f'fold_{ci}'/f'patch_{patch}'/(method+'_alignment.npz')
            if not ap.exists():save_npz(ap,alignment=a.astype(np.float32),defined=valid)
            visualize(out/'visualizations',case,patch,method,image,gt,p0,p,pred0,pred,mask,m,a,valid)
    patchrows=[r for r in tables['correction'] if r['case_id']==case and r['patch_index']==patch]
    baseline=next(r for r in patchrows if r['method']=='no_correction')
    for r in patchrows:r.update(full_ce_gain=baseline['full_ce']-r['full_ce'],dice_gain=r['dice']-baseline['dice'])
    logging.info('Resumed Phase B %s fold=%d patch=%d complete',stage,ci,patch)


def phase_b_patch(model,reader,case,ci,patch,rows,out,tables,history_metrics,stage='AB'):
    # This entry cannot run until all folds' phase A is sealed.
    if not reader.global_sealed: raise RuntimeError('Global Phase A not sealed')
    gt=reader.read(case,patch,'label')
    parent=Path(rows[0]['path']).parent
    with np.load(parent/'image_only_maps.npz') as z:
        image=z['image'];x=z['x3'];mask=z['mask']
    device=next(model.parameters()).device
    with torch.no_grad():
        features=model.encoder(torch.from_numpy(image)[None,None].to(device))
        baseline=model.decoder(features)
    if not np.array_equal(x,features[2][0].cpu().numpy()): raise AssertionError('Phase B encoder cache mismatch')
    base=next(r for r in rows if r['method']=='no_correction')
    z0,p0,pred0,_,_,_=load_candidate(base['path'],x)
    if not np.array_equal(z0,baseline[0].cpu().numpy()): raise AssertionError('Phase B baseline changed')
    ctx=s.evaluation_context(gt)
    y=ctx['y'].to(device);inner=ctx['inner'].to(device);outer=ctx['outer'].to(device)
    if stage=='AB':
        g,lb=gradient_at_x3(model,features,y,inner,outer)
        save_npz(out/'oracle_gradient.npz',gradient=g,gradient_norm=np.linalg.norm(g.astype(np.float64),axis=0),boundary_loss=lb)
    else:
        with np.load(out/'oracle_gradient.npz') as cache:g=cache['gradient'];lb=float(cache['boundary_loss'])
    logging.info('Oracle fold=%d patch=%d gradient L2=%.8g nonzero=%.6f',ci,patch,np.linalg.norm(g),np.mean(g!=0))
    with torch.no_grad():
        ce0=F.cross_entropy(torch.from_numpy(z0)[None],ctx['y'],reduction='none')[0].numpy()
    iw=ctx['inner'][0].numpy();ow=ctx['outer'][0].numpy()
    weights=.5*iw/iw.sum()+.5*ow/ow.sum()
    cpu_lb=float((ce0*weights).sum())
    if abs(cpu_lb-lb)>LOSS_ATOL: raise AssertionError('CPU/GPU boundary loss mismatch')
    pure=(ctx['fraction']>=.75)|(ctx['fraction']<=.25)
    teacher_correct=(r05.block_count(pred0)/64.>=.5)==(ctx['fraction']>=.75)
    for record in rows:
        method=record['method'];rep=record['repetition']
        if stage=='C' and method=='no_correction':continue
        common=dict(case_id=case,fold=ci,patch_index=patch,method=method,repetition=rep)
        z,p,pred,modified,selected,m=load_candidate(record['path'],x)
        delta=modified.astype(np.float64)-x.astype(np.float64)
        if np.any(delta[:,~selected]!=0): raise AssertionError('Residual outside mask')
        metrics=s.evaluate_prediction(z,p,pred,gt,pred0,p0,selected,m,ctx)
        metrics.update(boundary_ce_gain=cpu_lb-metrics['boundary_ce'])
        old=history_metrics.get((case,patch,method,rep))
        olddiff=None
        if old:
            olddiff=abs(metrics['boundary_ce']-float(old['boundary_ce']))
            if olddiff>LOSS_ATOL: raise AssertionError('Historical actual boundary loss differs')
            if metrics['corrected']!=int(old['corrected']) or metrics['harmed']!=int(old['harmed']):
                raise AssertionError('Historical native binary decisions differ')
        metricrow=dict(common,**metrics,residual_l2=record['residual_l2'],relative_l2=record['relative_l2'],
            q_mean=record['q_mean'],historical_boundary_ce_difference=olddiff)
        tables['correction'].append(metricrow)
        with torch.no_grad(): ce1=F.cross_entropy(torch.from_numpy(z)[None],ctx['y'],reduction='none')[0].numpy()
        weighted_gain=(ce0-ce1)*weights
        if method!='no_correction':
            sem,con,qr,types=semantic_metrics(m,selected,gt,pred0,pred,weighted_gain)
            for name,values in [('semantic',sem),('conflict',con),('q',qr),('types',types)]:
                tables[name].extend(dict(common,**v) for v in values)
        domains={'all':np.ones(mask.shape,bool),'high_D':mask,'low_D':~mask,
            'teacher_correct_pure':mask&pure&teacher_correct,'teacher_wrong_pure':mask&pure&~teacher_correct,
            'mixed':mask&~pure,'high_Q':mask&(m['q']>=.75),'low_Q':mask&(m['q']<.25),
            'contains_voxel_errors':mask&(r05.block_count(pred0!=gt.astype(bool))>0),
            'no_voxel_errors':mask&(r05.block_count(pred0!=gt.astype(bool))==0)}
        for domain,d in domains.items():
            tables['alignment'].append(dict(common,domain=domain,**alignment_stats(g,delta,d)))
        directional=float(np.sum(g.astype(np.float64)*delta))
        actual=metrics['boundary_ce']-cpu_lb
        norm=float(np.linalg.norm(delta))
        tables['derivative'].append(dict(common,predicted_loss_change=directional,actual_loss_change=actual,
            sign_agreement=bool(np.sign(directional)==np.sign(actual)),absolute_error=abs(actual-directional),
            sign_above_precision_floor=bool(min(abs(directional),abs(actual))>LOSS_ATOL),
            residual_l2=norm,normalized_directional_derivative=directional/norm if norm else None,
            gradient_l2=float(np.linalg.norm(g.astype(np.float64))),gradient_nonzero_fraction=float(np.mean(g!=0)),
            historical_loss_difference=olddiff,undefined_reason=None if norm else 'zero_residual'))
        if [ci,patch] in PROTOCOL['finite_difference_patches'] and method in PROTOCOL['finite_difference_methods']:
            fd=fd_check(model,features,y,inner,outer,delta,directional,lb)
            tables['fd'].extend(dict(common,**v) for v in fd)
            r0.write_csv(out.parent.parent/'finite_difference_verification.csv',tables['fd'])
            # The fixed .01 center is the primary implementation check; tiny-t
            # cancellation and forward ReLU crossings remain explicitly reported.
            if not fd[-1]['central_pass']: raise AssertionError('Fixed central finite difference failed; diagnose before interpreting results')
        if [ci,patch] in PROTOCOL['fixed_smoke_and_visuals'] and method in ('true_boundary','nonboundary','multi','local'):
            a,valid,_,_,_=alignment(g,delta)
            save_npz(out/(method+'_alignment.npz'),alignment=a.astype(np.float32),defined=valid)
            visualize(out.parent.parent/'visualizations',case,patch,method,image,gt,p0,p,pred0,pred,mask,m,a,valid)
    # Global losses use baseline metrics to express side effects with one sign.
    matching=[r for r in tables['correction'] if r['case_id']==case and r['patch_index']==patch]
    base_metrics=next(r for r in matching if r['method']=='no_correction')
    for r in matching:
        r['full_ce_gain']=base_metrics['full_ce']-r['full_ce']
        r['dice_gain']=r['dice']-base_metrics['dice']
    logging.info('Phase B %s fold=%d patch=%d complete; all fixed methods retained',stage,ci,patch)


def visualize(directory,case,patch,method,image,gt,p0,p,pred0,pred,mask,m,a,valid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(exist_ok=True)
    z=40;truth=gt.astype(bool)
    corrected=(pred0!=truth)&(pred==truth);harmed=(pred0==truth)&(pred!=truth)
    la=r05.lift_grid(a);lv=r05.lift_grid(valid)
    panels=[('MRI',image,'gray',float(image.min()),float(image.max())),
        ('GT boundary',r0.boundary(truth),'gray',0,1),('Teacher p0',p0,'viridis',0,1),
        ('Risk mask',r05.lift_grid(mask),'gray',0,1),
        ('Match foreground',r05.lift_grid(m['chosen']==0),'gray',0,1),
        ('Q',r05.lift_grid(m['q']),'viridis',0,1),
        ('Alignment (undefined masked)',np.ma.masked_where(~lv,la),'coolwarm',-1,1),
        ('Modified probability',p,'viridis',0,1),('Probability difference',p-p0,'coolwarm',-.05,.05),
        ('Corrected',corrected,'Greens',0,1),('Harmed',harmed,'Reds',0,1),('GT',truth,'gray',0,1)]
    fig,axes=plt.subplots(3,4,figsize=(15,11))
    for ax,(title,arr,cmap,lo,hi) in zip(axes.flat,panels):
        ax.imshow(arr[:,:,z].T,origin='lower',cmap=cmap,vmin=lo,vmax=hi);ax.set_title(title);ax.axis('off')
    fig.suptitle(case+' patch '+str(patch)+' fixed z=40 '+method);fig.tight_layout()
    fig.savefig(directory/(case+'_p'+str(patch)+'_'+method+'.png'),dpi=130);plt.close(fig)


TABLES = {
    'semantic':('semantic_matching_metrics.csv',['method','domain']),
    'conflict':('matching_conflict_analysis.csv',['method','domain']),
    'q':('q_reliability_analysis.csv',['method','group']),
    'types':('matching_error_types.csv',['method','type']),
    'alignment':('residual_gradient_alignment.csv',['method','domain']),
    'derivative':('directional_derivative_analysis.csv',['method']),
    'fd':('finite_difference_verification.csv',['method','t','precision','record_stage']),
    'correction':('correction_comparison.csv',['method'])}


def final_statistics(tables,out):
    cases,summary=summarize_tables({k:(tables[k],keys) for k,(_,keys) in TABLES.items()})
    comparisons=[]
    for table,metric in [('correction','boundary_ce_gain'),('correction','full_ce_gain'),('correction','dice_gain'),
                         ('correction','net_correction'),('semantic','accuracy')]:
        rows=[r for r in cases if r['table']==table and (table!='semantic' or r['domain']=='high_D')]
        for first,second in [('true_boundary','no_correction'),('true_boundary','opposite_match'),
                ('true_boundary','nonboundary'),('true_boundary','nonboundary_energy_matched'),('true_boundary','self'),
                ('multi','true_boundary'),('local','true_boundary'),('multi_energy_matched','true_boundary'),('local_energy_matched','true_boundary')]:
            a={r['case_id']:r.get(metric) for r in rows if r['method']==first}
            b={r['case_id']:r.get(metric) for r in rows if r['method']==second}
            vals=[a[c]-b[c] for c in a if a[c] is not None and b.get(c) is not None]
            if vals: comparisons.append(dict(table=table,metric=metric,first=first,second=second,**s.summarize(vals)))
    summary['paired_comparisons']=comparisons
    summary['statistical_unit']='8 receiver cases; random repetitions -> patch -> case; 10000 case bootstrap'
    r0.write_csv(out/'case_level_metrics.csv',cases)
    r0.write_json(out/'summary_statistics.json',summary)
    r0.write_csv(out/'prototype_representation_comparison.csv',[r for r in cases if r['table'] in ('semantic','correction') and r['method'] in ('true_boundary','multi','local','multi_energy_matched','local_energy_matched')])
    r0.write_csv(out/'boundary_vs_nonboundary_gradient.csv',[r for r in cases if r['table'] in ('alignment','derivative') and r['method'] in ('true_boundary','nonboundary','opposite_match','nonboundary_energy_matched')])
    return summary,cases


def write_report(out,summary,cases,verification):
    def st(table,method,metric,**group):
        for r in summary[table]:
            if r['group']==dict(method=method,**group):return r['metrics'].get(metric,{})
        return {}
    def val(table,method,metric,**group):return st(table,method,metric,**group).get('mean')
    def fmt(v):return '不可定义' if v is None else f'{v:.6g}'
    def ci(x):
        return fmt(x.get('mean'))+' ['+', '.join(fmt(v) for v in (x.get('ci95') or [None,None]))+']'
    methods=['true_boundary','opposite_match','nonboundary','nonboundary_energy_matched','self','random_match','multi','multi_energy_matched','local','local_energy_matched']
    lines=['# Experiment R1-D1：Donor匹配语义、残差方向与局部表示诊断','',
        '## 设计、权限与解释边界','',
        '冻结Baseline EMA iter27000；精确复用8个已见训练病例×3历史patch。主配置固定Dxyz→Max→Top2%、alpha=.02、逐位置相对L2 cap=.10。只修改Decoder输入x3；其他四层特征不变。所有折的候选预测、实际修改特征、配置、源码先SHA256封存，随后才读取当前Receiver角色GT。Donor跨折可换角色；没有把Receiver GT输入候选匹配。',
        '匹配使用Donor-only标准化后的余弦，实际残差在原始特征空间。每类4个KMeans原型，固定n_init10/max_iter300；每Donor每类最多512个、按七病例最小数等量采样。Local NN每Donor每类最多64个等量样本，保留来源case/patch/grid。两类最优相似度差定义Q；全局两均值原型与R1-S0逐图复核。多原型和NN均另有GT无关总残差L2匹配对照，保留自身Q零点和逐位置cap。',
        'Teacher网格类别定义为原分辨率伪标签占比≥0.5；GT纯前景≥.75、纯背景≤.25。混合网格只排除类别正确性，不排除分割损失、纠正误伤和梯度诊断。Q不是校准概率。Q分箱Boundary收益是该分箱覆盖原图体素对全patch平衡Boundary CE的贡献，分母保持全patch内外带数量，不能当作独立干预。',
        'Oracle G仅由临时x3叶张量经autograd.grad获得，不写模型.grad，不进入Phase A。A=〈Delta,-G〉/(||Delta||||G||+1e-6)，零范数位置单独标识。梯度反映整体Boundary CE经Decoder卷积的影响，不等于同网格局部纠正率。梯度余弦中的epsilon会缩小极小幅值位置的绝对值。能量加权采用实际残差范数平方。',
        '精度恢复经用户明确批准：历史TF32有限差分失败记录完整保留。Phase A的768候选、原型、Q、Mask与预测不变；语义、纠正/误伤、Boundary收益、Full CE和Dice均评价封存的历史TF32预测。Oracle梯度、对齐、同精度一阶验证使用严格FP32，matmul与cuDNN allow_tf32=False；本环境PyTorch1.10.2没有set_float32_matmul_precision接口，两个实际可用标志已检查生效。',
        'D1-B中的actual_loss_change明确为严格FP32诊断损失变化，与G使用相同计算精度；historical_actual_loss_change为原TF32实际变化。后者只作为跨精度效果对照，不混入严格FP32有限差分。precision_sensitivity.csv记录每patch两种基准损失和梯度差异。严格FP32临时重放不生成新的选择规则，不替换任何历史预测。',
        '正Gain_B表示Boundary CE降低；方向导数及实际Delta L为负才表示改善。统计先20随机重复平均、再3patch平均、8病例等权，10000病例bootstrap。没有独立验证；病例已参与原训练且LOCO开发集重叠，所有区间与选择均为探索性。没有训练Mean Teacher，也不能推断整例Dice或未见病例收益。','',
        '## D1-A：语义与Q','',
        '|方法|全域纯侧准确率|高D纯侧准确率 [95% CI]|高D Q均值|Q=1比例|Q正确性AUROC|',
        '|---|---:|---|---:|---:|---:|']
    for m in methods:
        lines.append('|'+ '|'.join([m,fmt(val('semantic',m,'accuracy',domain='all')),ci(st('semantic',m,'accuracy',domain='high_D')),
            fmt(val('semantic',m,'q_mean',domain='high_D')),fmt(val('semantic',m,'q_saturated_fraction',domain='high_D')),
            fmt(val('q',m,'auroc_q_correctness',group='continuous'))])+'|')
    lines+=['','固定Q分箱（高D候选域；各病例等权）：','',
            '|方法|Q组|纯侧准确率|候选格点数均值|纠正|误伤|Boundary收益贡献|','|---|---|---:|---:|---:|---:|---:|']
    for m in ['true_boundary','nonboundary','self','multi','local']:
        for q in ('0','1','2','3'):
            lines.append('|'+ '|'.join([m,q]+[fmt(val('q',m,k,group=q)) for k in ['accuracy','positions','corrected','harmed','boundary_gain_contribution']])+'|')
    lines+=['','## D1-B：实际残差与Oracle方向','',
            '|方法|高D平均A|A>0比例|A<0比例|严格FP32一阶Delta L [CI]|严格FP32实际Delta L [CI]|历史TF32实际Delta L [CI]|严格FP32符号一致比例|',
            '|---|---:|---:|---:|---|---|---|---:|']
    for m in methods:
        lines.append('|'+ '|'.join([m]+[fmt(val('alignment',m,k,domain='high_D')) for k in ['alignment_mean','aligned_fraction','opposed_fraction']]+
            [ci(st('derivative',m,'predicted_loss_change')),ci(st('derivative',m,'actual_loss_change')),
             ci(st('derivative',m,'historical_actual_loss_change')),fmt(val('derivative',m,'sign_agreement'))])+'|')
    lines+=['','有限差分使用预先固定case0/3的patch0、True与Non-Boundary残差、t=.001/.01；前向和中心差分全部保存。固定t=.01中心差分为主要正确性检查，容差2e-5+5%×|方向导数|；t=.001的消减误差不用于选择参数。详见finite_difference_verification.csv。','',
            '## D1-C及全局副作用','',
            '|方法|Boundary收益 [95% CI]|获益病例|Full CE收益|Dice变化|纠正|误伤|净纠正|相对特征L2|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        lines.append('|'+ '|'.join([m,ci(st('correction',m,'boundary_ce_gain')),str(st('correction',m,'boundary_ce_gain').get('positive_cases'))]+
            [fmt(val('correction',m,k)) for k in ['full_ce_gain','dice_gain','corrected','harmed','net_correction','relative_l2']])+'|')
    lines+=['','## 病例配对比较','',
            '|指标|第一方法|第二方法|第一减第二均值 [CI]|正差病例|','|---|---|---|---|---:|']
    for r in summary['paired_comparisons']:
        lines.append('|'+ '|'.join([r['metric'],r['first'],r['second'],ci(r),str(r['positive_cases'])])+'|')
    lines+=['','## 全部病例和负结果','',
            '|病例|方法|Boundary收益|净纠正|Dice变化|','|---|---|---:|---:|---:|']
    for r in cases:
        if r['table']=='correction' and r['method'] in ('true_boundary','multi','local','nonboundary'):
            lines.append('|'+ '|'.join([r['case_id'],r['method']]+[fmt(r.get(k)) for k in ['boundary_ce_gain','net_correction','dice_gain']])+'|')
    # Explicit answers use actual estimates rather than generic recommendations.
    true_gain=val('correction','true_boundary','boundary_ce_gain')
    best_local=any((val('correction',m,'boundary_ce_gain') or 0)>0 and
        (val('correction',m,'net_correction') or 0)>0 for m in ('multi','local'))
    lines+=['','## Q1–Q10与机制判断','',
        'Q1：True高D Q均值为'+fmt(val('semantic','true_boundary','q_mean',domain='high_D'))+
        '，Q饱和比例'+fmt(val('semantic','true_boundary','q_saturated_fraction',domain='high_D'))+
        '，对应纯侧准确率'+fmt(val('semantic','true_boundary','accuracy',domain='high_D'))+
        '，Q正确性AUROC '+fmt(val('q','true_boundary','auroc_q_correctness',group='continuous'))+
        '。不能把大Q当作高可信概率；固定分箱显示其单调性和错误集中情况。',
        'Q2：在高D且Donor/Teacher冲突的纯侧格点，True Donor准确率 '+fmt(val('conflict','true_boundary','donor_accuracy',domain='conflict'))+
        '、Teacher准确率 '+fmt(val('conflict','true_boundary','teacher_accuracy',domain='conflict'))+
        '。两者冲突不代表Donor应覆盖Teacher；混合格点排除数量见semantic CSV。B类误伤均值 '+fmt(val('types','true_boundary','harmed',type='B'))+
        '，C类纠正均值 '+fmt(val('types','true_boundary','corrected',type='C'))+'。',
        'Q3：True实际残差与负梯度方向一致比例 '+fmt(val('alignment','true_boundary','aligned_fraction',domain='high_D'))+
        '、相反比例 '+fmt(val('alignment','true_boundary','opposed_fraction',domain='high_D'))+
        '；整体一阶Delta L '+fmt(val('derivative','true_boundary','predicted_loss_change'))+
        '，实际Delta L '+fmt(val('derivative','true_boundary','actual_loss_change'))+'。均值余弦与全局方向导数需同时解释。',
        'Q4：Reverse一阶Delta L '+fmt(val('derivative','opposite_match','predicted_loss_change'))+
        '、Non-Boundary '+fmt(val('derivative','nonboundary','predicted_loss_change'))+
        '；实际变化分别 '+fmt(val('derivative','opposite_match','actual_loss_change'))+' / '+fmt(val('derivative','nonboundary','actual_loss_change'))+
        '。结合等能量对照判断方向是否仍有优势，不能把更小CE等同更好语义；类别概率校准只属可能解释，未被独立因果分解。',
        'Q5：True一阶与实际变化符号一致比例 '+fmt(val('derivative','true_boundary','sign_agreement'))+
        '，绝对误差均值 '+fmt(val('derivative','true_boundary','absolute_error'))+
        '，以上均为严格FP32同精度比较。与历史TF32实际变化的符号一致比例 '+fmt(val('derivative','true_boundary','historical_sign_agreement'))+
        '，这是跨精度的机制解释参考。原TF32小t验证失败不被改称通过；固定t=.01的严格FP32中心差分先完成4组验证才继续完整评价。t=.001全部失败/通过情况保留，不改变容差。有限alpha差异可含ReLU跨区、非线性与数值误差，不能逐个唯一归因。',
        'Q6：多原型/Local NN高D纯侧准确率 '+fmt(val('semantic','multi','accuracy',domain='high_D'))+' / '+fmt(val('semantic','local','accuracy',domain='high_D'))+
        '，对照Global '+fmt(val('semantic','true_boundary','accuracy',domain='high_D'))+'；这是固定库规模的局部表示诊断，不是调参得到的最佳结果。',
        'Q7：多原型/Local NN Boundary收益 '+fmt(val('correction','multi','boundary_ce_gain'))+' / '+fmt(val('correction','local','boundary_ce_gain'))+
        '，误伤 '+fmt(val('correction','multi','harmed'))+' / '+fmt(val('correction','local','harmed'))+
        '。准确率、Q或相似度提升不足以证明Decoder修正有效，需同时看净纠正和全图副作用。',
        'Q8：选择E（多种问题共同存在，暂不能唯一归因）。语义正确性、Oracle方向和表示改动的证据分别见上表；这里没有独立操纵“正确语义但固定同能量方向”与“错误语义”等因果因素，不能单独认定A/B/C/D中的某项为全部原因。',
        ('Q9：局部表示出现值得复核的信号，但优先研究修正方向及拒绝条件；不在本8例上继续搜索Q/K/alpha。' if best_local else
         'Q9：优先研究修正方向，暂停直接把全局/局部原型插值升级为训练方法。若继续匹配研究，应在独立控制中验证方向及误伤；不通过本批Oracle挑选补偿。'),
        'Q10：尚不具备进入完整Mean Teacher训练的证据。'+('主True Boundary收益为负。' if true_gain<0 else '即使某固定候选为正，也缺少独立验证。')+
        '本次8个已见病例、共享Donor开发集和冻结checkpoint分析不证明半监督训练或泛化改善。','',
        '## 验证、失败情况与可复核路径','',
        '所有实验数值、随机重复和失败病例保留。不可定义指标为空并带原因；零范数余弦以defined mask保存。测试、封存、权限日志、历史复现差异与模型保护详见verification.json、test_results.json、phaseA_seal.json、data_access_log.csv。预先固定case0/3 patch0 z40图像位于visualizations/。',
        '源码：analysis/experiment_R1_D1.py；测试：analysis/test_experiment_R1_D1.py；本目录为本地可复核结果，未自动推送GitHub。']
    (out/'experiment_report.md').write_text('\n'.join(lines)+'\n')
    start=lines.index('## Q1–Q10与机制判断')
    (out/'mechanism_diagnosis.md').write_text('# R1-D1机制诊断\n\n'+'\n'.join(lines[start:])+'\n')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:7')
    p.add_argument('--mode',choices=['smoke','full'],default='full')
    p.add_argument('--output_dir',default=str(BASE/datetime.datetime.now(r0.TZ).strftime('run_%Y%m%d_%H%M%S')))
    p.add_argument('--cpu_threads',type=int,default=4)
    p.add_argument('--resume_phaseB',action='store_true',help='Use authenticated existing candidates; never run Phase A')
    return p.parse_args(argv)


def resume_phase_b(args):
    out=Path(args.output_dir).resolve()
    if out.parent!=BASE or not (out/'phaseA_seal.json').is_file():raise ValueError('Resume requires original sealed run')
    metadata=json.loads((out/'experiment_metadata.json').read_text())
    if metadata['status']=='complete':raise RuntimeError('Already complete; refuse duplicate evaluation')
    pid=metadata.get('resume_pid')
    if pid and metadata['status'].startswith('phaseB'):
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Existing Phase B process is alive; refuse duplicate')
    stamp=datetime.datetime.now(r0.TZ).strftime('%Y%m%d_%H%M%S')
    audit_dir=out/'resume_audit'/stamp;audit_dir.mkdir(parents=True,exist_ok=False)
    for name in ('experiment_metadata.json','verification.json','test_results.json','finite_difference_verification.csv',
                 'experiment_report.md','mechanism_diagnosis.md','data_access_log.csv'):
        if (out/name).exists():shutil.copy2(out/name,audit_dir/name)
    old_audit=json.loads((out/'failure_provenance/post_failure_audit.json').read_text())
    original_seal=json.loads((out/'phaseA_seal.json').read_text())
    seal_sha=r0.sha256(out/'phaseA_seal.json')
    if seal_sha!=old_audit['parent_phaseA_seal_sha256']:raise RuntimeError('Original Phase-A seal changed')
    resolved=resolve_phase_a_seal(original_seal,metadata,old_audit)
    old_snapshot=next(r['snapshot_path'] for r in old_audit['source_snapshots'] if r['original_path']==str(CODE[0]))
    unchanged_helpers=verify_phase_a_code_unchanged(Path(old_snapshot).read_text(),CODE[0].read_text())
    plan=json.loads((out/'candidate_plan.json').read_text());records=plan['records'];ids=metadata['case_order']
    if plan['protocol']!=metadata['protocol'] or len(records)!=768 or original_seal['candidate_count']!=768:
        raise RuntimeError('Existing candidate protocol/count mismatch')
    keys={(r['case_id'],r['patch_index'],r['method'],r['repetition']) for r in records}
    if len(keys)!=768 or {(r['case_id'],r['patch_index']) for r in records}!={(c,p) for c in ids for p in range(3)}:
        raise RuntimeError('Incomplete/duplicate candidate plan')
    histories=json.loads((S0/'experiment_metadata.json').read_text())
    history,source_ids,checkpoint,_,_=s.load_references(argparse.Namespace(**histories['configuration']))
    if ids!=source_ids or checkpoint!=metadata['checkpoint'] or history['sampling']['patch_coordinates']!=metadata['crop_coordinates']:
        raise RuntimeError('Resume source/checkpoint/crop mismatch')
    reader=Reader(history);reader.case_order=ids
    current_hashes={str(p):r0.sha256(p) for p in CODE}
    for p in CODE:
        shutil.copy2(p,audit_dir/('resume_'+p.name))
        resolved['artifacts'].append(s.artifact(audit_dir/('resume_'+p.name),'resume_execution_source'))
    audit=dict(created_utc=now(),reason='User authorized strict FP32 Oracle/FD only; restore Phase B from unchanged candidates',
        prior_source_sha256=metadata['code_sha256'],resume_source_sha256=current_hashes,
        original_phaseA_seal_sha256=seal_sha,phaseA_protocol_unchanged=True,phaseA_helpers_AST_unchanged=unchanged_helpers,
        source_resolution='Only original source entries resolve to authenticated failure snapshots; all candidate paths/hashes unchanged',
        no_phaseA_execution=True,strict_oracle=True,historical_predictions_and_actual_losses='original TF32 archives',
        fd_t_and_tolerance_unchanged=True,resume_after_GT_first_read=True)
    r0.write_json(audit_dir/'resume_protocol_audit.json',audit)
    r0.write_json(audit_dir/'authenticated_evaluation_seal.json',resolved)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'analysis_log.txt'),logging.StreamHandler()])
    logging.info('Authenticated Phase-B-only resume; original Phase A stays immutable')
    metadata.update(status='phaseB_resume_verification',resume_pid=os.getpid(),resume_started_utc=now(),
        resume_source_sha256=current_hashes,resume_audit_path=str(audit_dir),oracle_precision='strict_fp32',
        prediction_precision='historical_tf32',resume_authorization='Explicit user approval for strict FP32 Oracle/finite difference',
        phaseA_source_snapshots=old_audit['source_snapshots'])
    r0.write_json(out/'experiment_metadata.json',metadata)
    try:
        reader.unlock(resolved)  # Full byte hashes, including unchanged predictions, before GT.
        logging.info('All original Phase-A artifacts authenticated; checking protected historical files')
        before=protected_manifest()
        if before!=metadata['protected_file_sha256_before']:raise AssertionError('Historical project files differ from original inventory')
        test=subprocess.run([sys.executable,'-B',str(CODE[1])],capture_output=True,text=True,cwd=ROOT)
        r0.write_json(out/'test_results.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr,
            precision_resume_tests=True,previous_tests_preserved=str(audit_dir/'test_results.json')))
        if test.returncode:raise RuntimeError('Updated precision/resume tests failed')
        model=u.load_frozen(checkpoint['path'],args.device);state=u.snapshot_state(model)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        with computation_precision(True) as settings:pass
        metadata['strict_fp32_settings']=settings;r0.write_json(out/'experiment_metadata.json',metadata)
        reader.events=read_csv(out/'data_access_log.csv')
        # The failed process did not flush its first Phase-B GT event. Recover
        # the known event, with no invented exact read timestamp.
        if not any(e.get('event_source')=='original_failure_log_reconstruction' for e in reader.events):
            reader.events.append(dict(event_index=len(reader.events),fold=0,case_id=ids[0],patch_index=0,
                dataset='label',role='receiver',phase='B',fold_sealed=True,all_fold_sealed=True,
                utc=None,event_source='original_failure_log_reconstruction',
                evidence='Oracle fold=0 patch=0 log 2026-10-14 17:51:45.996 after authenticated global unlock; exact GT read time not logged'))
        old_fd=[]
        # Always retain the ORIGINAL failed record, not a previous resume's table.
        original_fd=out/'resume_audit'
        original_csv=sorted(original_fd.glob('*/finite_difference_verification.csv'))[0]
        for row in read_csv(original_csv):
            d=dict(row,precision='historical_tf32',record_stage='original_failed_attempt')
            for k in ('t','autograd_directional_derivative','forward_difference','central_difference','forward_absolute_error','central_absolute_error','tolerance'):
                d[k]=float(d[k])
            for k in ('central_pass','forward_pass'):d[k]=d[k]=='True'
            old_fd.append(d)
        # Complete all FOUR predetermined strict checks before science tables.
        fd_rows=strict_preflight(model,reader,records,out,old_fd)
        metadata.update(status='phaseB',strict_preflight_passed=True,strict_preflight_completed_utc=now())
        r0.write_json(out/'experiment_metadata.json',metadata)
        state_path=out/'phaseB_resume_state.json'
        if state_path.exists():
            progress=json.loads(state_path.read_text());tables=progress['tables'];done=set(progress['completed_jobs'])
            tables['fd']=fd_rows
        else:tables={k:[] for k in TABLES};tables['fd']=fd_rows;done=set()
        hm={}
        for r in read_csv(S0/'patch_level_metrics.csv'):
            if r['mapping']=='max' and float(r['alpha'])==ALPHA:
                hm[r['case_id'],int(r['patch_index']),r['method'],int(r['repetition']) if r['repetition'] else None]=r
        for stage in ('AB','C'):
            logging.info('Strict Oracle evaluation stage %s; native TF32 prediction metrics',stage)
            for ci,case in enumerate(ids):
                reader.evaluation_fold(ci,case,[c for c in ids if c!=case])
                for patch in range(3):
                    job=f'{stage}/{ci}/{patch}'
                    if job in done:continue
                    rr=[r for r in records if r['case_id']==case and r['patch_index']==patch and
                        (r['method']=='no_correction' or r['method'].startswith(('multi','local'))==(stage=='C'))]
                    phase_b_strict_patch(model,reader,case,ci,patch,rr,out,tables,hm,stage)
                    done.add(job)
                    # Atomic progress prevents repeat inclusion on later SSH loss.
                    temp=out/'phaseB_resume_state.tmp.json'
                    r0.write_json(temp,dict(tables=tables,completed_jobs=sorted(done),original_seal_sha256=seal_sha))
                    os.replace(temp,state_path)
                    for k,(name,_) in TABLES.items():r0.write_csv(out/name,tables[k])
                    r0.write_csv(out/'precision_sensitivity.csv',tables.get('precision',[]))
                    r0.write_csv(out/'data_access_log.csv',reader.events)
        summary,cases=final_statistics(tables,out)
        summary.update(prediction_precision='historical_tf32',oracle_precision='strict_fp32',
            original_TF32_finite_difference_failure_retained=True,
            strict_precision_sensitivity=[dict(case_id=r['case_id'],**{k:v for k,v in r.items() if k!='case_id'}) for r in tables['precision']])
        r0.write_json(out/'summary_statistics.json',summary)
        verify=u.verify_state(model,state)
        logging.info('Final model state passed; verifying historical SHA256 and immutable Phase A')
        after=protected_manifest()
        if after!=before:raise AssertionError('Protected historical files changed during Phase B')
        reader.unlock(resolved)
        if r0.sha256(out/'phaseA_seal.json')!=seal_sha:raise AssertionError('Original seal file changed')
        for table in tables.values():
            for row in table:
                for name,value in row.items():
                    if isinstance(value,(float,np.floating)) and not np.isfinite(value):raise AssertionError('Nonfinite '+name)
        prim=[r for r in tables['fd'] if r.get('record_stage')=='resume_preflight' and r.get('precision')=='strict_fp32' and r['t']==.01]
        verify.update(status='complete',candidate_count=768,all_8_cases_24patches=True,
            protected_file_sha256_unchanged=True,protected_file_count=len(before),original_phaseA_seal_sha256=seal_sha,
            phaseA_candidates_and_predictions_unchanged=True,phaseA_source_resolution_documented=True,
            phaseA_helpers_AST_unchanged=True,no_phaseA_rerun=True,
            strict_fp32_settings=settings,oracle_only_x3_leaf=True,no_parameter_gradients=True,
            original_TF32_fd_failures_retained=True,strict_fp32_primary_checks=4,
            strict_fp32_primary_all_pass=len(prim)==4 and all(r['central_pass'] for r in prim),
            fixed_t_and_tolerance_unchanged=True,precision_never_mixed_for_finite_differences=True,
            predictions_and_actual_segmentation_metrics='sealed historical_tf32',
            all_GT_evaluation_after_authenticated_seal=True,case_statistical_unit=True,completed_jobs=len(done),
            no_test_GT=True,no_training_or_parameter_or_EMA_update=True,no_nonfinite_outputs=True)
        r0.write_json(out/'verification.json',verify)
        write_report(out,summary,cases,verify)
        metadata.update(status='complete',completed_utc=now(),protected_file_sha256_after=after,
            source_sha256_before_resume=metadata['code_sha256'],source_sha256_after_resume=current_hashes,
            resume_completed_jobs=len(done),phaseA_seal_unchanged=True)
        r0.write_json(out/'experiment_metadata.json',metadata)
        logging.info('R1-D1 full diagnosis complete; same original run %s',out)
    except Exception as exc:
        logging.exception('Resume stopped; candidates and scientific thresholds preserved')
        metadata.update(status='phaseB_resume_failed',resume_error=repr(exc),resume_failed_utc=now())
        r0.write_json(out/'experiment_metadata.json',metadata)
        raise


def main(argv=None):
    args=parse_args(argv)
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if args.device.startswith('cuda'): torch.ones(1,device=args.device)
    if args.resume_phaseB:return resume_phase_b(args)
    oldmeta=json.loads((S0/'experiment_metadata.json').read_text())
    oldverify=json.loads((S0/'verification.json').read_text())
    if oldmeta['status']!='complete' or not oldverify['all_8_cases_24patches']:raise ValueError('S0 incomplete')
    for path,h in oldmeta['code_sha256'].items():
        if r0.sha256(path)!=h:raise ValueError('S0 source changed since formal run')
    refargs=argparse.Namespace(**oldmeta['configuration'])
    history,ids,checkpoint,references,refhashes=s.load_references(refargs)
    if oldmeta['case_order']!=ids or oldmeta['crop_coordinates']!=history['sampling']['patch_coordinates'] or oldmeta['checkpoint']!=checkpoint:
        raise ValueError('S0 historical identity mismatch')
    out=create_output(args.output_dir)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'analysis_log.txt'),logging.StreamHandler()])
    metadata=dict(status='initializing',started_utc=now(),configuration=vars(args),protocol=PROTOCOL,
        checkpoint=checkpoint,case_order=ids,crop_coordinates=oldmeta['crop_coordinates'],data_sources=oldmeta['data_sources'],
        code_sha256={str(p):r0.sha256(p) for p in CODE},reference_sha256=refhashes,
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,device=args.device))
    r0.write_json(out/'experiment_metadata.json',metadata)
    try:
        logging.info('Read-only SHA256 inventory of protected existing files')
        before=protected_manifest();metadata['protected_file_sha256_before']=before
        r0.write_json(out/'experiment_metadata.json',metadata)
        test=subprocess.run([sys.executable,'-B',str(CODE[1])],capture_output=True,text=True,cwd=ROOT)
        r0.write_json(out/'test_results.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr))
        if test.returncode:raise RuntimeError('Unit tests failed')
        model=u.load_frozen(checkpoint['path'],args.device);state=u.snapshot_state(model)
        reader=Reader(history);images={};xs={};maps={};plans={};historical_artifacts={}
        for f in sorted(S0.glob('fold_*/phaseA_seal.json')):
            seal=json.loads(f.read_text())
            if seal['receiver_case'] in seal['donor_cases']:raise ValueError('Historical LOCO contaminated')
            for a in seal['artifacts']:historical_artifacts[a['path']]=a['sha256']
        def checked(path):
            path=Path(path)
            if str(path) in historical_artifacts and r0.sha256(path)!=historical_artifacts[str(path)]:
                raise ValueError('S0 sealed file modified '+str(path))
            return path
        for ci,case in enumerate(ids):
            plan=json.loads(checked(S0/f'fold_{ci}/phaseA_plan.json').read_text())
            for patch in range(3):
                path=S0/f'fold_{ci}/phaseA'/(case+f'_p{patch}')/'matching_maps.npz'
                with np.load(checked(path)) as f:maps[case,patch]={k:f[k] for k in f.files}
                images[case,patch]=reader.read(case,patch,'image')
                if not np.array_equal(images[case,patch],maps[case,patch]['image']):raise ValueError('Historical crop image changed')
                xs[case,patch]=maps[case,patch]['x3']
                plans[case,patch]={(r['method'],r['repetition']):r for r in plan['records'] if r['patch_index']==patch and r['mapping']=='max' and r['alpha']==ALPHA}
        artifacts=[];allrecords=[];bankstats=[];nnrows=[]
        folds=(0,3) if args.mode=='smoke' else range(8)
        metadata['status']='phaseA';r0.write_json(out/'experiment_metadata.json',metadata)
        for ci in folds:
            case=ids[ci];reader.start_fold(ci,case,[c for c in ids if c!=case])
            libs,donor=s.fold_prototypes(reader,xs,ids,case)
            with np.load(checked(S0/f'fold_{ci}/donor_prototypes.npz')) as old:
                for k,v in dict(mean=donor['mean'],scale=donor['scale'],**donor['prototypes']).items():
                    if not np.array_equal(v,old[k]):raise AssertionError('Recomputed donor/scaler differs '+k)
            banks,stats=build_banks(xs,libs,donor,case,ids,ci);bankstats.extend(stats)
            fd=out/f'fold_{ci}';fd.mkdir()
            save_npz(fd/'banks.npz',mean=donor['mean'],scale=donor['scale'],**donor['prototypes'],
                **{k+'_'+j:v for k,b in banks.items() for j,v in b.items()})
            for patch in ((0,) if args.mode=='smoke' else range(3)):
                pd=fd/f'patch_{patch}';pd.mkdir()
                # Exact historical checks occur without Receiver labels.
                rows,nr=phase_a_patch(model,images[case,patch],donor,banks,maps[case,patch],plans[case,patch],pd,ci,patch)
                for r in rows:r['case_id']=case
                allrecords.extend(rows);nnrows.extend(dict(case_id=case,source_case_id=ids[r['source_case_index']],**r) for r in nr)
                logging.info('Phase A fold=%d patch=%d %d candidates complete; Receiver GT locked',ci,patch,len(rows))
            r0.write_csv(out/'data_access_log.csv',reader.events)
        if len(allrecords)!=(2 if args.mode=='smoke' else 24)*32:raise AssertionError('Incomplete fixed candidate plan')
        r0.write_csv(out/'multi_prototype_statistics.csv',bankstats)
        r0.write_csv(out/'local_nn_matching_statistics.csv',nnrows)
        r0.write_csv(out/'feature_perturbation_statistics.csv',allrecords)
        r0.write_json(out/'candidate_plan.json',dict(protocol=PROTOCOL,records=allrecords))
        for f in sorted(out.glob('fold_*')):
            for p in sorted(f.rglob('*.npz')):artifacts.append(s.artifact(p,'candidate'))
        for p in [out/'candidate_plan.json',out/'multi_prototype_statistics.csv',out/'local_nn_matching_statistics.csv']+CODE:
            artifacts.append(s.artifact(p,'protocol_or_source'))
        seal=dict(created_utc=now(),phaseA_complete=True,artifacts=artifacts,receiver_GT_reads=0,
            fold_count=len(folds),patch_count=2 if args.mode=='smoke' else 24,candidate_count=len(allrecords))
        r0.write_json(out/'phaseA_seal.json',seal)
        if args.mode=='smoke':
            verify=u.verify_state(model,state)
            verify.update(no_receiver_GT_reads=not any(e['role']=='receiver' and e['dataset']=='label' for e in reader.events),
                historical_predictions_masks_matching_reproduced=True,candidate_count=len(allrecords))
            after=protected_manifest()
            if before!=after:raise AssertionError('Protected files changed')
            verify['protected_file_sha256_unchanged']=True
            r0.write_json(out/'verification.json',verify)
            metadata.update(status='smoke_complete',completed_utc=now());r0.write_json(out/'experiment_metadata.json',metadata)
            (out/'experiment_report.md').write_text('# R1-D1工程冒烟\n\n固定case0/3 patch0，仅Phase A；64个候选已封存。没有读取Receiver GT、没有评价收益或调整参数。\n')
            return
        reader.unlock(seal)
        metadata.update(status='phaseB',receiver_unlock_utc=now());r0.write_json(out/'experiment_metadata.json',metadata)
        tables={k:[] for k in TABLES}
        hm={}
        for r in read_csv(S0/'patch_level_metrics.csv'):
            if r['mapping']=='max' and float(r['alpha'])==ALPHA:
                hm[r['case_id'],int(r['patch_index']),r['method'],int(r['repetition']) if r['repetition'] else None]=r
        for stage in ('AB','C'):
            logging.info('Starting fixed Phase B stage %s',stage)
            for ci,case in enumerate(ids):
                reader.evaluation_fold(ci,case,[c for c in ids if c!=case])
                for patch in range(3):
                    rr=[r for r in allrecords if r['case_id']==case and r['patch_index']==patch and
                        (r['method']=='no_correction' or r['method'].startswith(('multi','local'))==(stage=='C'))]
                    pd=out/f'fold_{ci}'/f'patch_{patch}'
                    phase_b_patch(model,reader,case,ci,patch,rr,pd,tables,hm,stage)
                    for k,(name,_) in TABLES.items():r0.write_csv(out/name,tables[k])
                    r0.write_csv(out/'data_access_log.csv',reader.events)
        summary,cases=final_statistics(tables,out)
        verify=u.verify_state(model,state)
        logging.info('Final read-only protection hashes and sealed-candidate recheck')
        after=protected_manifest()
        if before!=after:raise AssertionError('Historical files changed')
        reader.unlock(seal)
        # No silent nonfinite JSON/CSV output; explicit None is allowed.
        for k,rows in tables.items():
            for row in rows:
                for name,v in row.items():
                    if isinstance(v,(float,np.floating)) and not np.isfinite(v):raise AssertionError('Nonfinite '+k+'/'+name)
        verify.update(protected_file_sha256_unchanged=True,protected_file_count=len(before),
            all_candidates_sealed_before_receiver_GT=True,receiver_GT_reads=sum(e['role']=='receiver' and e['dataset']=='label' for e in reader.events),
            historical_forward_masks_matching_and_loss_within_fixed_tolerances=True,
            oracle_only_x3_leaf=True,no_parameter_gradients=all(p.grad is None for p in model.parameters()),
            fixed_fd_primary_all_pass=all(r['central_pass'] for r in tables['fd'] if r['t']==.01),
            fd_all_results_retained=True,statistical_unit='case',no_nonfinite_outputs=True,
            candidate_count=len(allrecords),all_8_cases_24patches=True,no_test_GT=True,no_training=True)
        r0.write_json(out/'verification.json',verify)
        write_report(out,summary,cases,verify)
        metadata.update(status='complete',completed_utc=now(),protected_file_sha256_after=after)
        r0.write_json(out/'experiment_metadata.json',metadata)
        logging.info('R1-D1 complete: %s',out)
    except Exception as exc:
        logging.exception('R1-D1 stopped; no scientific thresholds changed')
        metadata.update(status='failed',error=repr(exc),failed_utc=now())
        r0.write_json(out/'experiment_metadata.json',metadata)
        raise


if __name__=='__main__':main()
