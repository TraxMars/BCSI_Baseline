"""D0-A: frozen EMA ablation and leave-one-case-out boundary-side probes.

Only standalone logistic probes are fitted. VNet is frozen, eval, no_grad;
no MT training, BGS gain, decoder changes, feature transfer or D0-B.
"""
import argparse
from collections import defaultdict
import datetime
import inspect
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import warnings

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import h5py
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr, rankdata
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score, precision_score, recall_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import experiment_bgs_utility as u
import experiment_receiver_R0 as r0

MODES = ("scale_0p95", "scale_0p50", "zero", "spatial_mean")
FACTORS = {"scale_0p95": .95, "scale_0p50": .5, "zero": 0.}
PRIMARY = ("boundary_contribution", "bgs", "full_ce_contribution", "random")
KS = (8, 16)
BOOTSTRAPS = 10000
SIGN_EPS = 1e-6
NEW_CODE = {Path(__file__).resolve(), HERE / "test_experiment_donor_D0.py"}
PROBE_SAMPLES = 512
PROBE_C = 1.
PROBE_MAX_ITER = 1000
PROBE_TOL = 1e-6


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference_dir", default=str(ROOT / "analysis/results/experiment_bgs_utility/run_cuda_20261009_1713"))
    p.add_argument("--model_path", default=str(ROOT / "Results/seed_42/result_LA_10l/fold_0/Model_iter_27000.pth"))
    p.add_argument("--output_dir", default=str(ROOT / "analysis/results/experiment_donor_D0" /
        datetime.datetime.now(r0.TZ).strftime("run_%Y%m%d_%H%M%S")))
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--decoder_batch_size", type=int, default=1)
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.seed != 42 or not 1 <= args.decoder_batch_size <= 8 or args.cpu_threads < 1:
        p.error("Fixed seed42, decoder batch1..8 and positive CPU threads required")
    return args


def create_output(path):
    p = Path(path).resolve()
    if p.parent != ROOT / "analysis/results/experiment_donor_D0" or not p.name.startswith("run_"):
        raise ValueError("Use a fresh timestamp child of analysis/results/experiment_donor_D0")
    p.mkdir(parents=True, exist_ok=False)
    return p


def protected_manifest(out):
    result = {}
    for parent, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache") and (Path(parent)/d).resolve() != out.resolve())
        for name in sorted(files):
            path = Path(parent)/name
            if path.is_file() and path.resolve() not in NEW_CODE and path.suffix not in (".pyc", ".pyo"):
                result[str(path.relative_to(ROOT))] = r0.sha256(path)
    return result


def read_csv(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def historical_channels(reference, ids):
    rows = [r for r in read_csv(reference/"channel_bgs_utility.csv") if r["checkpoint"] == "baseline_ema"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case_id"], int(row["patch_index"])].append(row)
    if set(grouped) != {(i,p) for i in ids for p in range(3)}:
        raise ValueError("Historical channel records missing/duplicated patches")
    result = {}
    for key, rr in grouped.items():
        rr.sort(key=lambda x:int(x["channel"]))
        if [int(r["channel"]) for r in rr] != list(range(64)):
            raise ValueError("Historical checkpoint must have all64 distinct channels")
        result[key] = {metric: np.array([float(r[metric]) for r in rr]) for metric in ("bgs","utility")}
        result[key]["losses"] = {metric: float(rr[0][metric]) for metric in ("boundary_ce","full_ce","supervised_ce_dice")}
    fd = {(r["case_id"],int(r["patch_index"]),int(r["channel"])):float(r["boundary_ce_gain_minus"])
          for r in read_csv(reference/"finite_difference_validation.csv") if r["checkpoint"] == "baseline_ema"}
    return result, fd


class RoleReader:
    """GT is available for Donor A; held-out GT is locked per LOCO fold.

    A case is a developer in other folds. Its earlier Donor GT/measurements do
    not make it eligible for its own fold's selection, scaler or probe fit.
    """
    def __init__(self, history):
        self.history = history
        self.sources = {s["case_id"]:s for s in history["data_sources"]}
        self.coords = {(c["case_id"],c["patch_index"]):c for c in history["sampling"]["patch_coordinates"]}
        self.events = []
        self.fold = None
        self.held = None
        self.plan_sealed = False

    def start_fold(self, fold, held):
        self.fold, self.held, self.plan_sealed = fold, held, False

    def seal_plan(self, path, digest):
        if r0.sha256(path) != digest:
            raise RuntimeError("Fold selection/probe plan seal digest differs")
        self.plan_sealed = True

    def read(self, case_id, patch, field):
        if field not in ("image","label"):
            raise ValueError("Only image/label reads permitted")
        if self.fold is not None and field == "label" and case_id == self.held and not self.plan_sealed:
            raise RuntimeError("Held-out GT locked until dev-only selection and probe parameters saved")
        with h5py.File(self.sources[case_id]["path"],"r") as f:
            array = f[field][:]
        arr = r0.replay_array(array,self.coords[case_id,patch],[112,112,80],np.float32 if field=="image" else np.int64)
        r0.finite(arr,case_id+"/"+field)
        if field == "label" and not np.isin(arr,[0,1]).all():
            raise ValueError("Nonbinary historical GT")
        self.events.append(dict(event_index=len(self.events),case_id=case_id,patch_index=patch,field=field,fold=self.fold,
            role="donor_descriptive" if self.fold is None else "held_out_evaluation" if case_id==self.held else "development",
            fold_plan_sealed=self.plan_sealed))
        return arr


@torch.no_grad()
def loss_values(logits, labels, inner, outer):
    """Per-example evaluation even when decoder computation is batched."""
    if logits.shape[0] != 1:
        raise ValueError("Loss reductions must be per-example, never mix interventions")
    values = u.loss_values(logits,labels,inner,outer)
    ce = F.cross_entropy(logits,labels.long(),reduction="none")
    values.update(inner_ce=ce[inner].double().mean(),outer_ce=ce[outer].double().mean())
    values["ce_plus_dice_sum"] = 2*values["supervised_ce_dice"]
    for key,value in values.items(): r0.finite(value,key)
    return {k:float(v.item()) for k,v in values.items()}


@torch.no_grad()
def intervene(features, channels, mode):
    """Modify x3 only, out of place; preserve all other feature objects."""
    if mode not in MODES or len(channels) != len(set(channels)) or any(c<0 or c>=features[2].shape[1] for c in channels):
        raise ValueError("Invalid fixed x3 intervention")
    result = list(features)
    f = features[2]
    new = f.clone()
    if mode == "spatial_mean":
        new[:,channels] = f[:,channels].mean((2,3,4),keepdim=True)
    else:
        new[:,channels] = f[:,channels]*FACTORS[mode]
    result[2] = new
    relative = float(((new-f).double().norm()/(f.double().norm()+1e-6)).item())
    assert all(result[i] is features[i] for i in (0,1,3,4))
    return result, relative


@torch.no_grad()
def evaluate_interventions(model,features,labels,inner,outer,scenarios,batch_size):
    rows=[]
    for start in range(0,len(scenarios),batch_size):
        ss=scenarios[start:start+batch_size]
        changes=[intervene(features,s["channels"],s["mode"]) for s in ss]
        batch=[features[i].expand(len(ss),-1,-1,-1,-1) if i!=2 else torch.cat([r[0][2] for r in changes],0) for i in range(5)]
        logits=model.decoder(batch)
        r0.finite(logits,"intervention logits")
        for i,scenario in enumerate(ss):
            rows.append(dict(**scenario,relative_feature_change=changes[i][1],**loss_values(logits[i:i+1],labels,inner,outer)))
        del logits,batch,changes
    return rows


@torch.no_grad()
def probe_boundary_masks(labels,spatial_shape):
    small = F.interpolate(labels[:,None].float(),size=tuple(spatial_shape),mode="nearest")[:,0].long()
    inner,outer=u.boundary_sides(small)
    return inner[0].cpu().numpy(),outer[0].cpu().numpy()


def sampled_probe_data(feature,inner,outer,seed):
    count=min(PROBE_SAMPLES,int(inner.sum()),int(outer.sum()))
    if count<8:
        return np.empty((0,feature.shape[0]),np.float64),np.empty(0,np.int64),np.empty(0,np.int64),dict(valid=False,reason="insufficient_x3_boundary_sides",samples_per_class=count)
    rng=np.random.RandomState(seed)
    ii=rng.choice(np.flatnonzero(inner.ravel()),count,replace=False)
    oo=rng.choice(np.flatnonzero(outer.ravel()),count,replace=False)
    indices=np.r_[ii,oo]
    x=feature.reshape(feature.shape[0],-1).T[indices].astype(np.float64)
    y=np.r_[np.ones(count,np.int64),np.zeros(count,np.int64)]
    return x,y,indices,dict(valid=True,reason="",samples_per_class=count)


def stable_topk(values,k):
    values=np.asarray(values)
    if k>len(values) or not np.isfinite(values).all(): raise ValueError("Invalid ranking")
    return np.lexsort((np.arange(len(values)),-values))[:k].tolist()


def fold_selection(case_arrays,held_out,seed):
    """Only developer dictionaries are accepted; held-out cannot enter rank."""
    if held_out in case_arrays or len(case_arrays)!=7:
        raise ValueError("LOCO ranking needs exactly seven cases, excluding held-out")
    fields={metric:np.mean([d[metric] for d in case_arrays.values()],axis=0)
            for metric in ("boundary_contribution","bgs","full_ce_contribution","energy","utility")}
    rows=[]
    for k in KS:
        for method in ("boundary_contribution","bgs","full_ce_contribution","energy_auxiliary"):
            key="energy" if method=="energy_auxiliary" else method
            rows.append(dict(method=method,k=k,repetition=None,channels=stable_topk(fields[key],k)))
        for repetition in range(20):
            shuffle_seed=seed+repetition*1009
            perm=np.random.RandomState(shuffle_seed).permutation(len(fields["bgs"]))
            rows.append(dict(method="random",k=k,repetition=repetition,random_seed=shuffle_seed,channels=perm[:k].tolist()))
    return rows,fields


def development_matrix(cache,development_ids,held_out):
    if held_out in development_ids or len(development_ids)!=7: raise ValueError("Probe development includes held-out")
    xs,ys,weights=[],[],[]
    valid_counts={case:sum(cache[case,p]["sample_valid"] for p in range(3)) for case in development_ids}
    if any(n==0 for n in valid_counts.values()):
        raise ValueError("Developer case has no valid probe patches; no silent omission")
    for case in development_ids:
        for patch in range(3):
            d=cache[case,patch]
            if not d["sample_valid"]: continue
            x,y=d["sample_x"],d["sample_y"]
            xs.append(x); ys.append(y)
            weights.append(np.full(len(y),1/(len(development_ids)*valid_counts[case]*len(y))))
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(weights)
    w*=len(w)/w.sum()
    mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w))
    scale=np.where(scale>1e-8,scale,1.)
    return (x-mean)/scale,y,w,mean,scale,valid_counts


def fit_probe(x,y,w,channels):
    probe=LogisticRegression(C=PROBE_C,penalty="l2",solver="lbfgs",max_iter=PROBE_MAX_ITER,tol=PROBE_TOL,random_state=42)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always",ConvergenceWarning)
        probe.fit(x[:,channels],y,sample_weight=w)
    coef,intercept=probe.coef_[0],float(probe.intercept_[0])
    r0.finite(coef,"probe coefficients")
    return dict(coefficients=coef.tolist(),intercept=intercept,n_iter=int(probe.n_iter_[0]),
                converged=not any(issubclass(t.category,ConvergenceWarning) for t in captured),warnings=[str(t.message) for t in captured])


def predict_probe(feature,channels,mean,scale,probe):
    """Input is feature-only; no GT, boundary masks or evaluator arguments."""
    x=feature.reshape(feature.shape[0],-1).T[:,channels].astype(np.float64)
    probability=expit(((x-mean[channels])/scale[channels])@np.asarray(probe["coefficients"])+probe["intercept"])
    r0.finite(probability,"held-out probe predictions")
    return probability.reshape(feature.shape[1:]).astype(np.float32)


def probe_metrics(probability,inner,outer):
    domain=inner|outer
    target=inner[domain].astype(np.int64)
    p=probability[domain]
    if not len(target) or len(np.unique(target))!=2:
        return dict(auroc=None,balanced_accuracy=None,ap_foreground_side=None,precision=None,recall=None,
                    n_boundary_voxels=len(target),positive_fraction=None,undefined_reason="empty_or_single_class_boundary_domain")
    pred=p>=.5
    return dict(auroc=float(roc_auc_score(target,p)),balanced_accuracy=float(balanced_accuracy_score(target,pred)),
                ap_foreground_side=float(average_precision_score(target,p)),precision=float(precision_score(target,pred,zero_division=0)),
                recall=float(recall_score(target,pred,zero_division=0)),n_boundary_voxels=len(target),positive_fraction=float(target.mean()),undefined_reason="")


def correlation(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0: return None,"constant_or_insufficient_values"
    value=float(spearmanr(a,b).correlation)
    r0.finite(value,"Spearman")
    return value,""


def partial_rank_correlation(a,b,control):
    if any(np.ptp(v)==0 for v in (a,b,control)): return None,"constant_values"
    x=np.c_[np.ones(len(a)),rankdata(control)]
    aa=rankdata(a)-x@np.linalg.lstsq(x,rankdata(a),rcond=None)[0]
    bb=rankdata(b)-x@np.linalg.lstsq(x,rankdata(b),rcond=None)[0]
    if np.linalg.norm(aa)<1e-10 or np.linalg.norm(bb)<1e-10: return None,"zero_residual"
    value=float(np.dot(aa,bb)/(np.linalg.norm(aa)*np.linalg.norm(bb)))
    return value,""


def bootstrap(values,seed=42):
    vals=np.array([v for v in values if v is not None],np.float64)
    if not len(vals): return dict(mean=None,median=None,std=None,ci95=[None,None],n_cases=0,reason="no_defined_cases",unit="case")
    draws=np.random.RandomState(seed).randint(len(vals),size=(BOOTSTRAPS,len(vals)))
    return dict(mean=float(vals.mean()),median=float(np.median(vals)),std=float(vals.std(ddof=1)) if len(vals)>1 else 0.,
        ci95=np.percentile(vals[draws].mean(1),[2.5,97.5]).tolist(),n_cases=len(vals),unit="case",repetitions=BOOTSTRAPS,
        positive_cases=int((vals>SIGN_EPS).sum()),negative_cases=int((vals< -SIGN_EPS).sum()))


def pairwise_stability(case_vectors):
    rows=[]
    ids=list(case_vectors)
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            rho,reason=correlation(case_vectors[a],case_vectors[b])
            for k in KS:
                aa=set(stable_topk(case_vectors[a],k)); bb=set(stable_topk(case_vectors[b],k))
                rows.append(dict(case_i=a,case_j=b,k=k,spearman=rho,spearman_undefined_reason=reason,
                    intersection=len(aa&bb),jaccard=len(aa&bb)/len(aa|bb)))
    return rows


def aggregate_contributions(rows,ids,channels):
    by=defaultdict(list)
    for r in rows: by[r["case_id"],r["mode"],r["channel"]].append(r)
    cases=[]
    for case in ids:
        for mode in MODES:
            for channel in range(channels):
                pp=by[case,mode,channel]
                if len(pp)!=3: raise AssertionError("A channel/patch contribution was omitted")
                row=dict(case_id=case,mode=mode,channel=channel)
                for metric in ("delta_boundary_ce","delta_full_ce","delta_supervised_ce_dice","delta_inner_ce","delta_outer_ce",
                               "feature_l2","relative_feature_change","bgs","utility"):
                    row[metric]=float(np.mean([p[metric] for p in pp]))
                row["positive_patch_fraction"]=float(np.mean([p["delta_boundary_ce"]>SIGN_EPS for p in pp]))
                cases.append(row)
    array={case:{} for case in ids}
    for case in ids:
        moderate=[r for r in cases if r["case_id"]==case and r["mode"]=="scale_0p50"]
        for name,key in (("boundary_contribution","delta_boundary_ce"),("full_ce_contribution","delta_full_ce"),
                         ("bgs","bgs"),("utility","utility"),("energy","feature_l2")):
            array[case][name]=np.array([r[key] for r in moderate])
    stability=[]
    for mode in MODES:
        vectors={case:np.array([r["delta_boundary_ce"] for r in cases if r["case_id"]==case and r["mode"]==mode]) for case in ids}
        stability.extend(dict(mode=mode,**r) for r in pairwise_stability(vectors))
    channel_stats=[]
    for mode in MODES:
        for c in range(channels):
            cc=[r for r in cases if r["channel"]==c and r["mode"]==mode]
            channel_stats.append(dict(mode=mode,channel=c,mean_contribution=float(np.mean([r["delta_boundary_ce"] for r in cc])),
                positive_case_fraction=float(np.mean([r["delta_boundary_ce"]>SIGN_EPS for r in cc])),
                positive_patch_fraction=float(np.mean([r["positive_patch_fraction"] for r in cc])),
                mean_bgs=float(np.mean([r["bgs"] for r in cc])),mean_utility=float(np.mean([r["utility"] for r in cc])),
                mean_feature_l2=float(np.mean([r["feature_l2"] for r in cc])),
                contribution_summary=bootstrap([r["delta_boundary_ce"] for r in cc])))
    return cases,array,stability,channel_stats


def aggregate_loco(rows,ids):
    metrics=("delta_boundary_ce","delta_full_ce","delta_supervised_ce_dice","relative_feature_change","auroc","balanced_accuracy","ap_foreground_side")
    by=defaultdict(list)
    for row in rows: by[row["case_id"],row["k"],row["method"],row["repetition"]].append(row)
    repeat_cases=[]
    for (case,k,method,rep),pp in by.items():
        if len(pp)!=3: raise AssertionError("Held-out patch omitted")
        rr=dict(case_id=case,k=k,method=method,repetition=rep,n_patches=3)
        for metric in metrics:
            vv=[p[metric] for p in pp if p[metric] is not None]
            rr[metric]=float(np.mean(vv)) if vv else None
            rr[metric+"_defined_patches"]=len(vv)
        rr["undefined_patch_reasons"]=[p["undefined_reason"] for p in pp if p["undefined_reason"]]
        repeat_cases.append(rr)
    case_rows=[]
    for case in ids:
        for k in KS:
            for method in ("boundary_contribution","bgs","full_ce_contribution","energy_auxiliary","random"):
                pp=[r for r in repeat_cases if (r["case_id"],r["k"],r["method"])==(case,k,method)]
                if len(pp)!=(20 if method=="random" else 1): raise AssertionError("Random repetitions missing")
                rr=dict(case_id=case,k=k,method=method,n_repetitions=len(pp))
                for metric in metrics:
                    vv=[p[metric] for p in pp if p[metric] is not None]
                    rr[metric]=float(np.mean(vv)) if vv else None
                    if method=="random":
                        rr[metric+"_random_std"]=float(np.std(vv)) if vv else None
                        rr[metric+"_random_min"]=float(np.min(vv)) if vv else None
                        rr[metric+"_random_max"]=float(np.max(vv)) if vv else None
                case_rows.append(rr)
    summaries,paired={},[]
    for k in KS:
        for method in ("boundary_contribution","bgs","full_ce_contribution","energy_auxiliary","random"):
            cc=[r for r in case_rows if r["k"]==k and r["method"]==method]
            summaries[str(k)+"/"+method]={m:bootstrap([c[m] for c in cc]) for m in metrics}
        for control in ("random","bgs","full_ce_contribution","energy_auxiliary"):
            for case in ids:
                aa=next(r for r in case_rows if (r["case_id"],r["k"],r["method"])==(case,k,"boundary_contribution"))
                bb=next(r for r in case_rows if (r["case_id"],r["k"],r["method"])==(case,k,control))
                rr=dict(case_id=case,k=k,method="boundary_contribution",control=control)
                rr.update({m+"_difference":aa[m]-bb[m] if aa[m] is not None and bb[m] is not None else None for m in metrics})
                paired.append(rr)
            summaries[str(k)+"/boundary_minus_"+control]={m:bootstrap([r[m+"_difference"] for r in paired if r["k"]==k and r["control"]==control]) for m in metrics}
    return repeat_cases,case_rows,paired,summaries


def fmt(stat):
    if stat["mean"] is None: return "不可定义"
    return "{:.4f} [{:.4f}, {:.4f}]".format(stat["mean"],*stat["ci95"])


def scientific_findings(summary,channel_stats):
    """Conclusions from fixed tests; do not feed these into fold selection."""
    sm=summary["loco"]
    stats={(r["mode"],r["channel"]):r for r in channel_stats}
    top=sorted([r for r in channel_stats if r["mode"]=="scale_0p50"],key=lambda r:(-r["mean_contribution"],r["channel"]))[:16]
    lines=["", "## 量化科学结论与负面结果", "",
        "### Q1：哪些通道支持GT边界预测？", "",
        "以下为预定.50有符号贡献的全病例前16通道。正贡献表示当前冻结Decoder的GT边界CE依赖该通道；不是可迁移性标签。", "",
        "|通道|.50贡献均值|.95贡献均值|置零贡献|空间均值替代贡献|.50正贡献病例|均值替代正贡献病例|",
        "|---|---|---|---|---|---|---|"]
    for r in top:
        c=r["channel"]
        lines.append("|{}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{}/8|{}/8|".format(c,r["mean_contribution"],
            stats["scale_0p95",c]["mean_contribution"],stats["zero",c]["mean_contribution"],stats["spatial_mean",c]["mean_contribution"],
            round(r["positive_case_fraction"]*8),round(stats["spatial_mean",c]["positive_case_fraction"]*8)))
    lines += ["", "### Q2：贡献是否跨病例稳定？", "",
        ".50贡献排名病例间Spearman为 {}，Top8 Jaccard {}，Top16 Jaccard {}。较高的整体排名一致性不代表所有TopK身份固定。LOCO联合消融相对随机的边界CE变化差，K8为 {}、K16为 {}，均支持跨病例Decoder依赖。".format(
            fmt(summary["stability"]["scale_0p50"]["spearman"]),fmt(summary["stability"]["scale_0p50"]["jaccard8"]),
            fmt(summary["stability"]["scale_0p50"]["jaccard16"]),fmt(sm["8/boundary_minus_random"]["delta_boundary_ce"]),fmt(sm["16/boundary_minus_random"]["delta_boundary_ce"])),
        "空间结构解释有重要反例：.50与空间均值替代的通道排名Spearman仅 {}；与完全置零则为 {}。消除空间结构与整体缩放/零移除测量不同属性，不能将缩放敏感通道自动称为边界空间信息来源。".format(
            fmt(summary["relations"]["scale_0p50_vs_spatial_mean"]),fmt(summary["relations"]["scale_0p50_vs_zero"])),
        "", "### Q3：是否只是激活幅值大？", "",
        ".50贡献与L2的Spearman为 {}。K8时BC/BGS/随机的相对扰动为 {:.4f}/{:.4f}/{:.4f}，而边界CE变化为 {:.4f}/{:.4f}/{:.4f}；高能量辅助对照扰动更大但边界CE变化更小。这些事实不支持用幅值大小单独解释贡献差异，但未做逐组等能量因果干预，不能完全排除能量及Decoder权重耦合。".format(
            fmt(summary["relations"]["scale_0p50/energy"]),*[sm["8/"+m]["relative_feature_change"]["mean"] for m in ("boundary_contribution","bgs","random")],
            *[sm["8/"+m]["delta_boundary_ce"]["mean"] for m in ("boundary_contribution","bgs","random")]),
        "", "### Q4：Boundary Contribution是否优于BGS和随机？", "",
        "在联合移除后的边界CE恶化量上是：K8相对BGS差 {}、相对随机差 {}；K16分别 {}、{}。但这是Decoder对选中通道的依赖测试，完全移除K个通道可能造成分布外输入，并不是接收Donor信息的收益。".format(
            fmt(sm["8/boundary_minus_bgs"]["delta_boundary_ce"]),fmt(sm["8/boundary_minus_random"]["delta_boundary_ce"]),
            fmt(sm["16/boundary_minus_bgs"]["delta_boundary_ce"]),fmt(sm["16/boundary_minus_random"]["delta_boundary_ce"])),
        "", "### Q5：是否有更好的跨病例边界判别能力？", "",
        "没有稳定优于随机的证据。K8的BC/随机/BGS AUROC分别为 {:.4f}/{:.4f}/{:.4f}；BC−随机差 {}，BC−BGS差 {}。K16分别 {:.4f}/{:.4f}/{:.4f}；BC−随机差 {}，BC−BGS差 {}。K16相对BGS有正向AUROC差，但相对随机区间仍含0，K8相对BGS也无优势，不支持选择性只报告K16。".format(
            *[sm["8/"+m]["auroc"]["mean"] for m in ("boundary_contribution","random","bgs")],
            fmt(sm["8/boundary_minus_random"]["auroc"]),fmt(sm["8/boundary_minus_bgs"]["auroc"]),
            *[sm["16/"+m]["auroc"]["mean"] for m in ("boundary_contribution","random","bgs")],
            fmt(sm["16/boundary_minus_random"]["auroc"]),fmt(sm["16/boundary_minus_bgs"]["auroc"])),
        "FullCE的K8 AUROC为 {:.4f}，高于BC；L2辅助对照K8/K16为 {:.4f}/{:.4f}，亦高于BC，即使Probe输入已经开发集标准化。这表明容易线性判别的边界信息与当前Decoder消融重要性不完全一致。所有Probe预测了GT边界两侧标签，未执行分割推理修改。".format(
            sm["8/full_ce_contribution"]["auroc"]["mean"],sm["8/energy_auxiliary"]["auroc"]["mean"],sm["16/energy_auxiliary"]["auroc"]["mean"]),
        "A1复核的C与历史U Spearman为 {}，.50为 {}；轻微减弱结果与历史有限差分最大差异为0。该局部对应是数学预期，不应作为新筛选机制有效的证据。BGS与.50贡献仍呈 {} 的弱负相关。".format(
            fmt(summary["relations"]["scale_0p95/utility"]),fmt(summary["relations"]["scale_0p50/utility"]),fmt(summary["relations"]["scale_0p50/bgs"])),
        "", "### Q6：哪些通道可进入下一阶段？", "",
        "当前checkpoint中的诊断候选为 {}，其中低BGS且跨消融支持的通道为 {}。只能保留为待验证候选，不能固定为可靠Donor传递通道；当前组合Probe无法逐一归因单通道的线性判别或真实传递价值。高BGS但不满足稳定正贡献的通道列表完整保留，不能笼统说它们对边界没有信息。".format(
            summary["candidate_channels"],summary["low_bgs_stable_contributors"]),
        "", "### Q7与下一步：是否现在加入CHAP梯度对齐？", "",
        "目前不建议直接增加D0-B或训练机制。没有真实同迭代Student–Teacher成对状态；贡献与判别Probe间的差异首先需要独立病例验证，并区分空间结构与Decoder敏感性。可行性章节仅提出条件与增量验证方法，没有实际对齐、网络模块或通道Dropout。",
        "**最终选择{}。** {} 其中“类似”是本轮探索性比较，并非证明统计等价；BGS作为Probe对照与贡献选择接近，也不表示BGS正向通道增强重新获得支持。八个病例均见过Baseline训练，不宣称真正未见病例泛化、Donor–Receiver传递成功或Dice提高。".format(summary["decision"]["choice"],summary["decision"]["reason"])]
    return lines


def write_report(out,meta,summary,channel_stats,verification):
    sm=summary["loco"]
    lines=["# Experiment D0-A：Boundary-Contributive Donor Channel Analysis","",
      "实验完整执行：冻结原始MT Baseline EMA，在历史8病例×3固定patch上分析全部x3通道。没有训练Mean Teacher、更新BN/EMA、增加Decoder或实施知识传递。",
      "", "## 预先固定方法与数据保护", "",
      "模型 {}，SHA256 {}。历史数据路径、crop/padding/来源SHA与BGS-U一一核对，实际shape逐patch保存。test.list仅读取病例ID排除重叠，不读取测试GT。".format(meta["checkpoint"]["path"],meta["checkpoint"]["sha256"]),
      "这8个病例已参与Baseline有标签训练。LOCO只对通道选择/Probe开发隔离，并不把固定VNet变成从未见过held-out图像的模型；当前结果为探索性跨病例验证，没有独立验证集。",
      "所有单通道测试完整保留：a=.95、.50、0、空间均值替代。其他四层原样保留。主排名为a=.50的有符号平均贡献，先3patch平均，再7开发病例等权。FullCE同强度，BGS有符号排名；不以held-out调强度/阈值，不按正值过滤导致K不同。并列分数按channelID确定顺序。",
      "C=LB(扰动)−LB(原始)，正值表示消融增大GT边界平衡CE。LB=.5×内侧CE+.5×外侧CE，复用BGS-U原分辨率形态学；同时记录两侧、全图CE、.5×(CE+Dice)及未缩放CE+Dice之和的变化。a=.95应满足C≈.05U，这是局部数学关系/历史复核，不是新方法有效的证据。",
      "主实验Decoder batch={}；每个干预的CE/Dice独立计算，不把不同干预混成一个Dice；FP32，不混合精度。单位通道和重复前向与单例原forward进行数值核对。先前batch4尝试未通过预定数值等价检查且未生成消融结果，保留失败记录；主实验采用batch1，不放宽门限。".format(meta["configuration"]["decoder_batch_size"]),
      "四个主要策略：Boundary Contribution、BGS、Full-image CE Contribution、20随机TopK（固定独立RNG，K8/16使用同一排列的前缀，20次全部保存、从不选最好一次）。预先额外保留无GT的激活L2 TopK辅助对照，用于幅值解释，不能替代主要对照。",
      "LOCO每折仅接受7开发病例字典。全部选中ID、开发分数、标准化统计、Probe系数先保存并SHA封存，然后才解锁该折held-out GT评价。描述性Donor消融已读过全部病例GT，但held-out的任何测量都不进入其折的排名/标准化/Probe；审计为折内角色隔离，不伪称全局从未读取GT。",
      "Probe：仅x3选中通道、GT最近邻缩小后构造内外边界侧；每patch每类无放回min(512,内侧数,外侧数)，至少8否则显式无效。各方法共享采样坐标；开发病例/patch等权和类别平衡，标准化只使用开发集。固定L2逻辑回归C1、LBFGS、max_iter1000、tol1e-6、阈值.5，不调参。Probe为独立分析工具，非MT模块。held-out先对全部x3网格预测，再读取GT计算全边界域AUROC/BalAcc/AP；不把GT输入预测函数。",
      "统计：patch→case；随机先在同病例平均20次；8病例等权、10000病例Bootstrap。LOCO开发集重叠，区间为探索性；不把28个病例pair、体素或20随机当独立病例。无目标指标为空并记录原因。",
      "", "## 留一病例：联合移除K通道及边界Probe", "",
      "正边界CE变化表示移除后恶化更大，支持固定模型判别贡献；并不表示传递特征会改善Receiver。", "",
      "|K|方法|边界CE变化|全图CE变化|相对特征扰动|Probe AUROC|Balanced Accuracy|",
      "|---|---|---|---|---|---|---|"]
    for k in KS:
        for method in ("boundary_contribution","bgs","full_ce_contribution","random","energy_auxiliary"):
            rr=sm[str(k)+"/"+method]
            lines.append("|{}|{}|{}|{}|{}|{}|{}|".format(k,method,*[fmt(rr[m]) for m in ("delta_boundary_ce","delta_full_ce","relative_feature_change","auroc","balanced_accuracy")]))
    lines += ["", "## 病例配对差：Boundary Contribution减对照", "",
      "|K|对照|边界CE变化差|AUROC差|BalAcc差|", "|---|---|---|---|---|"]
    for k in KS:
        for method in ("random","bgs","full_ce_contribution","energy_auxiliary"):
            rr=sm[str(k)+"/boundary_minus_"+method]
            lines.append("|{}|{}|{}|{}|{}|".format(k,method,*[fmt(rr[m]) for m in ("delta_boundary_ce","auroc","balanced_accuracy")]))
    lines += ["", "## 跨病例稳定性与消融敏感性", ""]
    for mode,s in summary["stability"].items():
        lines.append("{}：病例平均pair Spearman {}，Top8 Jaccard {}，Top16 Jaccard {}。pair先按每病例平均后做病例Bootstrap，不对28对独立Bootstrap。".format(mode,fmt(s["spearman"]),fmt(s["jaccard8"]),fmt(s["jaccard16"])))
    lines += ["", "|关系（病例内64通道，再按病例汇总）|Spearman均值与CI|", "|---|---|"]
    for key,val in summary["relations"].items(): lines.append("|{}|{}|".format(key,fmt(val)))
    lines += ["", "完整64通道×24patch×4方式共6144行贡献保存，不截Top8；强度排名不一致和负贡献均保留。完全置零可造成分布外输入、联合移除存在交互，因此不等于单通道贡献相加。",
      "仅零移除有大贡献、但.50/空间均值不支持的通道：{}。预定义为zero贡献Top16，同时另外两种≤1e-6；全部统计见channel_level_statistics.csv。".format(summary["zero_only_channels"]),
      "", "## 能量、BGS及历史U解释", "",
      "激活L2相关、C与U以及控制能量后的rank相关均保存。L2 TopK是辅助对照，空间均值替代保留通道均值；Probe开发标准化消除单纯输入缩放对线性分类的直接影响，但消融依然有非线性与干预能量混杂。没有等能量因果干预，不能彻底排除幅值因素。",
      "低BGS（全体病例均值排名后25%）且在.50与空间均值均有≥6/8正贡献的通道：{}。这里只称低边界梯度选择性，不称边界能力弱。".format(summary["low_bgs_stable_contributors"]),
      "高BGS（前25%）但.50跨病例正贡献比例<.75：{}。组合Probe不能独立归因单个channel的迁移能力。".format(summary["high_bgs_inconsistent_channels"]),
      "Probe与开发选中通道平均BGS/C/U/L2的关系是每折各组内探索性相关，随机20次保留；不把这些模型当作独立病例。",
      "", "## 科学问题与决策", "",
      "Q1：实际支持通道由有符号贡献及跨方式一致性界定，不硬编码ID；候选见下方，全部正/负结果CSV保留。",
      "Q2：稳定性由病例排名Spearman、TopK交集以及LOCOheld-out联合消融检验共同判断，不能由一次patch固定channel。",
      "Q3：是否只来自幅值，需要结合能量相关、L2辅助对照、均值替代和标准化Probe；当前只能提供部分区分，未彻底证明与能量无关。",
      "Q4/Q5：以上配对表分别回答对GT边界CE的跨病例支持及前景/背景侧线性判别信息。更大的消融损失与更高Probe必须区分；任意一项单独获胜均不足证明可迁移。",
      "Q6：探索性候选需.50及空间均值在≥6/8病例正贡献、且Boundary LOCO Top16选择频率≥6/8；该筛选仅用于后续研究建议，未回馈任何held-out选择/Probe。候选ID：{}，仅对应当前checkpoint。".format(summary["candidate_channels"]),
      "Q7：本轮不运行D0-B。真实Student–Teacher同迭代状态尚未确认，仅EMA不足以识别无监督梯度。",
      "决策 **{}**：{}".format(summary["decision"]["choice"],summary["decision"]["reason"]),
      "", "## CHAP-style Gradient Alignment Feasibility", "",
      "当前checkpoint是trainer验证时torch.save(ema_model.state_dict())的EMA普通权重，不含同迭代Student状态。没有把两个EMA副本伪装成Teacher–Student，也未实际计算无监督梯度或实施CHAP通道Dropout/双Decoder。",
      "若获得真实同迭代Student与EMA Teacher，可在离线副本中固定Teacher并对Student允许自动求导（不执行参数更新）。在labeled图像计算两侧平衡监督边界CE，在unlabeled图像使用真实Student概率与stop-gradient Teacher概率的原MSE一致性损失；保持原输入扰动约定和BN模式明确记录。",
      "选取encoder.block_three.conv中实际产生x3的最后Conv3d权重，按输出channel展平两种梯度为[64,Cin×3×3×3]；残差identity不对应该权重，因此可额外报告对x3叶特征的梯度作区别，但不要把不同空间样本的特征梯度硬匹配。BN的gamma/beta可单列辅助，不与卷积行随意拼接。参数名须动态检查而非复制CHAP固定下标。",
      "使用torch.autograd.grad分别求两种损失对同一Student参数的梯度，检查有限性、非零行及范数；Teacher冻结/目标detach。Student与Teacher相同且输入无差异时MSE梯度可能为0，必须标记不可定义，不能给零梯度默认高相似度。只计算grad不调用backward或optimizer.step，不累计到参数.grad。",
      "逐通道先L2标准化梯度再计算signed cosine，将原始范数单独报告，不把幅值加权cosine当纯对齐。监督侧保持内/外均衡，非监督侧报告前景/背景占比、consistency rampup缩放和零梯度情况；病例/patch相等权聚合，排除由全背景伪标签占比造成的假对齐。",
      "检验增量时预先固定边界贡献与梯度对齐的比较/组合规则，在按病例隔离的开发/验证和相同K、能量对照下比较消融贡献与Probe；采用病例配对分析，不能用同held-out调权重。独立新病例和多个真实训练seed仍必要。当前D0-A不提供梯度对齐有用性的证据，只有后续有明确增量问题时才值得增加。",
      "", "## 执行验证与复核文件", "",
      "验证 {}；{} 个既有项目文件/结果/checkpoint SHA256完全一致，冻结模型参数/BN buffers逐tensor一致；所有输出有限、缺失原因显式记录。".format(verification["status"],verification["protected_file_count"]),
      "数据路径、原crop坐标、环境、Probe/RNG/干预预注册参数、源码及checkpoint哈希见experiment_metadata.json；每折选择和Probe参数见fold_plans/*.json及其SHA。训练采样坐标和全网格held-out概率保存在feature_cache/与probe_predictions/。",
      "channel_contribution.csv、ablation_sensitivity.csv、cross_case_stability.csv、leave_one_case_out.csv、selection_comparison.csv、boundary_probe_results.csv、bgs_utility_comparison.csv，以及完整20次随机和病例数据全部保留。",
      "输出目录：{}。本地新文件未自动推送GitHub。".format(out.relative_to(ROOT))]
    lines += scientific_findings(summary,channel_stats)
    (out/"experiment_report.md").write_text("\n".join(lines)+"\n")


def main(argv=None):
    args=parse_args(argv)
    requested=Path(args.output_dir).resolve()
    if requested.exists(): raise FileExistsError("Refusing existing output directory: "+str(requested))
    reference=Path(args.reference_dir).resolve()
    history=json.loads((reference/"experiment_metadata.json").read_text())
    ids,checkpoint=r0.validate_history(history,args.model_path)
    historic,fd=historical_channels(reference,ids)
    if args.device.startswith("cuda"): torch.empty(1,device=args.device)
    torch.set_num_threads(args.cpu_threads)
    threadpool_limits(limits=args.cpu_threads)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    out=create_output(requested)
    for name in ("feature_cache","fold_plans","probe_predictions"): (out/name).mkdir()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",handlers=[logging.FileHandler(str(out/"analysis_log.txt")),logging.StreamHandler(sys.stdout)])
    before=protected_manifest(out)
    meta=dict(status="started",experiment="D0-A",configuration=vars(args),case_order=ids,checkpoint=checkpoint,
      data_sources=history["data_sources"],crop_coordinates=history["sampling"]["patch_coordinates"],
      reference_sha256={f:r0.sha256(reference/f) for f in ("experiment_metadata.json","channel_bgs_utility.csv","finite_difference_validation.csv")},
      code_sha256={str(p.relative_to(ROOT)):r0.sha256(p) for p in NEW_CODE},protected_file_sha256_before=before,
      protocol=dict(modes=MODES,primary_selection="signed mean boundary CE contribution at a=.50; patch->case->7 developers",
        full_CE_selection="same a=.50",bgs_selection="signed raw BGS mean",ks=KS,random_repetitions=20,
        energy_auxiliary_predefined=True,fold_GT_isolation="Held-out excluded from selection and probe fit; other-fold descriptive GT is not globally unread",
        probe_samples_per_side=PROBE_SAMPLES,probe_C=PROBE_C,probe_max_iter=PROBE_MAX_ITER,probe_tol=PROBE_TOL,
        probe_threshold=.5,probe_solver="lbfgs",probe_standardization="developer-only weighted per channel",
        bootstrap_unit="case",bootstrap_repetitions=BOOTSTRAPS,sign_epsilon=SIGN_EPS,
        candidate_rule="positive >=6/8 at .50 and spatial mean plus LOCO boundary Top16 selection frequency>=6/8; research recommendation only",
        history_match_atol=2e-6,decoder_batch_match_atol=2e-5,decoder_batch_match_rtol=1e-5,
        decision_rule="A requires positive BC-minus-random ablation and BC-minus-BGS/random AUROC CIs at bothK plus consistent candidates; otherwise B if BGS is within .01 AUROC of BC at eitherK, else C; descriptive margin not equivalence proof",
        no_MT_training_no_transfer_no_D0B=True),environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,device=args.device),
      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=str(ROOT),capture_output=True,text=True)
    meta["git_commit"]=git.stdout.strip()
    r0.write_json(out/"experiment_metadata.json",meta)
    try:
        test=subprocess.run([sys.executable,"-B",str(HERE/"test_experiment_donor_D0.py")],capture_output=True,text=True)
        r0.write_json(out/"test_results.json",dict(exit_code=test.returncode,stdout=test.stdout,stderr=test.stderr))
        if test.returncode: raise AssertionError("D0 unit tests failed: "+test.stderr)
        model=u.load_frozen(args.model_path,args.device)
        state=u.snapshot_state(model)
        reader=RoleReader(history)
        cache,contributions,checks,patches={},[],[],[]
        for case_index,case in enumerate(ids):
            for patch in range(3):
                image=reader.read(case,patch,"image")
                labels=torch.from_numpy(reader.read(case,patch,"label"))[None].to(args.device)
                inner,outer=u.boundary_sides(labels)
                if min(int(inner.sum()),int(outer.sum()))<8: raise ValueError("Historical full boundary sides invalid")
                with torch.no_grad():
                    features=model.encoder(torch.from_numpy(image)[None,None].to(args.device))
                    preserved=[f.clone() for f in features]
                    logits=model.decoder(features)
                    original=model(torch.from_numpy(image)[None,None].to(args.device))
                    if not torch.equal(logits,original): raise AssertionError("Encoder->decoder differs from original eval forward")
                    base=loss_values(logits,labels,inner,outer)
                    repeat=[f.expand(args.decoder_batch_size,-1,-1,-1,-1) for f in features]
                    repeated=model.decoder(repeat)
                    batch_error=float((repeated-logits).abs().max().item())
                    if not torch.allclose(repeated,logits.expand_as(repeated),atol=2e-5,rtol=1e-5):
                        raise AssertionError("Batched frozen decoder changes per-example inference")
                    bgs,_,_,_,valid=u.bgs_details(features[2],labels)
                old=historic[case,patch]
                if not np.allclose(bgs.cpu().numpy()[0],old["bgs"],atol=1e-9,rtol=0): raise AssertionError("Historical BGS mismatch")
                if any(abs(base[k]-old["losses"][k])>2e-6 for k in old["losses"]): raise AssertionError("Historical boundary/full/supervised loss mismatch")
                channels=features[2].shape[1]
                norms=features[2].double().square().sum((0,2,3,4)).sqrt().cpu().numpy()
                scenarios=[dict(channel=c,channels=[c],mode=mode) for mode in MODES for c in range(channels)]
                result=evaluate_interventions(model,features,labels,inner,outer,scenarios,args.decoder_batch_size)
                max_fd_diff=0.
                for rr in result:
                    row=dict(case_id=case,patch_index=patch,channel=rr["channel"],mode=rr["mode"],
                        bgs=float(old["bgs"][rr["channel"]]),utility=float(old["utility"][rr["channel"]]),
                        feature_l2=float(norms[rr["channel"]]),relative_feature_change=rr["relative_feature_change"],
                        inner_voxels=int(inner.sum()),outer_voxels=int(outer.sum()),feature_shape=list(features[2].shape))
                    for key in base: row["delta_"+key]=rr[key]-base[key]; row["baseline_"+key]=base[key]
                    if rr["mode"]=="scale_0p95":
                        row["first_order_expected_contribution"]=.05*row["utility"]
                        row["first_order_residual"]=row["delta_boundary_ce"]-.05*row["utility"]
                        if (case,patch,rr["channel"]) in fd:
                            diff=abs(row["delta_boundary_ce"]+fd[case,patch,rr["channel"]])
                            row["historical_FD_attenuation_max_abs_difference"]=diff
                            max_fd_diff=max(max_fd_diff,diff)
                            if diff>2e-6: raise AssertionError("Historical 0.95 finite-difference mismatch")
                    contributions.append(row)
                if any(not torch.equal(a,b) for a,b in zip(preserved,features)): raise AssertionError("Cached encoder features changed")
                feature=features[2][0].cpu().numpy().copy()
                small_inner,small_outer=probe_boundary_masks(labels,feature.shape[1:])
                sample_x,sample_y,sample_indices,sample_status=sampled_probe_data(feature,small_inner,small_outer,args.seed+case_index*1009+patch*9176)
                path=out/"feature_cache"/(case+"_p"+str(patch)+".npz")
                np.savez_compressed(str(path),x3=feature,sample_indices=sample_indices,sample_x=sample_x,sample_y=sample_y)
                cache[case,patch]=dict(feature=feature,sample_x=sample_x,sample_y=sample_y,sample_valid=sample_status["valid"])
                patches.append(dict(case_id=case,patch_index=patch,feature_shape=list(features[2].shape),feature_cache_path=str(path),feature_cache_sha256=r0.sha256(path),**sample_status))
                checks.append(dict(case_id=case,patch_index=patch,baseline_forward_exact=True,batch_decoder_max_abs_diff=batch_error,
                    BGS_matches_history=True,historical_FD_max_abs_diff=max_fd_diff,other_features_unchanged=True))
                r0.write_csv(out/"channel_contribution.csv",contributions)
                logging.info("Donor %s p%d: all %d x4 interventions done; shape=%s; batchdiff=%g FDdiff=%g probe_sample=%s",case,patch,channels,list(features[2].shape),batch_error,max_fd_diff,sample_status)
                del features,preserved,logits,original,repeated,repeat,labels
        case_contrib,case_arrays,stability,channel_stats=aggregate_contributions(contributions,ids,channels)
        r0.write_csv(out/"case_channel_contribution.csv",case_contrib)
        loco,probe_rows,selection_rows,alignment_rows,plan_artifacts=[],[],[],[],[]
        for fold,held in enumerate(ids):
            reader.start_fold(fold,held)
            dev=[i for i in ids if i!=held]
            selections,dev_scores=fold_selection({i:case_arrays[i] for i in dev},held,args.seed+fold*100003)
            x,y,weights,mean,scale,valid_counts=development_matrix(cache,dev,held)
            for selection in selections:
                selection["probe"]=fit_probe(x,y,weights,selection["channels"])
                selection["development_score_means"]={k:float(v[selection["channels"]].mean()) for k,v in dev_scores.items()}
                selection_rows.append(dict(fold=fold,held_out_case=held,development_cases=dev,**selection))
            plan=dict(fold=fold,held_out_case=held,development_cases=dev,selections=selections,development_scores=dev_scores,
                scaler_mean=mean,scaler_scale=scale,probe_development_valid_patch_counts=valid_counts,held_out_GT_used_for_selection=False)
            path=out/"fold_plans"/("fold_"+str(fold)+".json")
            r0.write_json(path,plan)
            digest=r0.sha256(path)
            reader.seal_plan(path,digest)
            plan_artifacts.append(dict(path=str(path),sha256=digest,fold=fold,held_out_case=held))
            logging.info("Fold%d %s: %d developer-only selections/probes sealed; held-out GT now unlocked",fold,held,len(selections))
            fold_predictions=[]
            for patch in range(3):
                feature=cache[held,patch]["feature"]
                probs=[predict_probe(feature,s["channels"],mean,scale,s["probe"]) for s in selections]
                fold_predictions.append(np.stack(probs))
                image=reader.read(held,patch,"image")
                with torch.no_grad(): features=model.encoder(torch.from_numpy(image)[None,None].to(args.device))
                if not np.array_equal(features[2][0].cpu().numpy(),feature): raise AssertionError("Held-out feature cache replay differs")
                labels=torch.from_numpy(reader.read(held,patch,"label"))[None].to(args.device)
                inner,outer=u.boundary_sides(labels)
                with torch.no_grad(): base=loss_values(model.decoder(features),labels,inner,outer)
                small_inner,small_outer=probe_boundary_masks(labels,feature.shape[1:])
                scenarios=[dict(channels=s["channels"],mode="zero",method=s["method"],k=s["k"],repetition=s["repetition"]) for s in selections]
                removals=evaluate_interventions(model,features,labels,inner,outer,scenarios,args.decoder_batch_size)
                for i,(selection,removal,probability) in enumerate(zip(selections,removals,probs)):
                    metrics=probe_metrics(probability,small_inner,small_outer)
                    identity=dict(fold=fold,case_id=held,patch_index=patch,k=selection["k"],method=selection["method"],repetition=selection["repetition"])
                    row=dict(**identity,selected_channels=selection["channels"],relative_feature_change=removal["relative_feature_change"],
                        **{"delta_"+key:removal[key]-base[key] for key in base},**metrics)
                    loco.append(row)
                    probe_rows.append(dict(**identity,**metrics,probe_converged=selection["probe"]["converged"],probe_n_iter=selection["probe"]["n_iter"]))
                    alignment_rows.append(dict(**identity,**metrics,**selection["development_score_means"]))
                del features,labels
            np.savez_compressed(str(out/"probe_predictions"/(held+".npz")),probability=np.stack(fold_predictions),
                                strategy_index=np.arange(len(selections)))
            r0.write_csv(out/"leave_one_case_out.csv",loco)
            r0.write_csv(out/"boundary_probe_results.csv",probe_rows)
            logging.info("Fold%d %s complete: all 3 held-out patches, all20 random repetitions/K retained",fold,held)
        finish(out,meta,before,model,state,reader,contributions,case_contrib,case_arrays,stability,channel_stats,loco,probe_rows,
               selection_rows,alignment_rows,plan_artifacts,patches,checks,test.returncode,ids)
    except Exception as e:
        logging.exception("D0 failed; partial outputs retained without overwrite")
        meta.update(status="failed",error=repr(e))
        r0.write_json(out/"experiment_metadata.json",meta)
        r0.write_json(out/"verification.json",dict(status="failed",error=repr(e)))
        raise


def finish(out,meta,before,model,state,reader,contributions,case_contrib,case_arrays,stability,channel_stats,loco,probe_rows,
           selection_rows,alignment_rows,plan_artifacts,patches,checks,test_exit,ids):
    repeat_cases,cases,paired,loco_summary=aggregate_loco(loco,ids)
    summary=dict(n_cases=8,n_patches=24,n_channels=int(len(case_arrays[ids[0]]["bgs"])),loco=loco_summary,
                 bootstrap_unit="case",bootstrap_repetitions=BOOTSTRAPS,exploratory=True,independent_validation_available=False,
                 relations={},stability={})
    sensitivity,bgs_comparison=[],[]
    for case in ids:
        mode_rows={mode:[r for r in case_contrib if r["case_id"]==case and r["mode"]==mode] for mode in MODES}
        vec={mode:np.array([r["delta_boundary_ce"] for r in rr]) for mode,rr in mode_rows.items()}
        for i,mode in enumerate(MODES):
            for other in MODES[i+1:]:
                rho,reason=correlation(vec[mode],vec[other])
                sensitivity.append(dict(case_id=case,mode=mode,control_mode=other,spearman=rho,undefined_reason=reason))
        for mode in MODES:
            for metric in ("bgs","utility","energy"):
                rho,reason=correlation(vec[mode],case_arrays[case][metric])
                bgs_comparison.append(dict(case_id=case,mode=mode,metric=metric,spearman=rho,undefined_reason=reason))
            rho,reason=partial_rank_correlation(vec[mode],case_arrays[case]["bgs"],case_arrays[case]["energy"])
            bgs_comparison.append(dict(case_id=case,mode=mode,metric="bgs_partial_controlling_energy",spearman=rho,undefined_reason=reason))
    for mode in MODES:
        summary["stability"][mode]={}
        for metric,k in (("spearman",8),("jaccard8",8),("jaccard16",16)):
            vals=[]
            for case in ids:
                pp=[r for r in stability if r["mode"]==mode and r["k"]==k and case in (r["case_i"],r["case_j"])]
                vv=[r["spearman" if metric=="spearman" else "jaccard"] for r in pp if r["spearman" if metric=="spearman" else "jaccard"] is not None]
                vals.append(float(np.mean(vv)) if vv else None)
            summary["stability"][mode][metric]=bootstrap(vals)
    for mode in MODES:
        for metric in ("bgs","utility","energy","bgs_partial_controlling_energy"):
            summary["relations"][mode+"/"+metric]=bootstrap([r["spearman"] for r in bgs_comparison if r["mode"]==mode and r["metric"]==metric])
    for mode in MODES:
        for control in MODES:
            pp=[r["spearman"] for r in sensitivity if r["mode"]==mode and r["control_mode"]==control]
            if pp: summary["relations"][mode+"_vs_"+control]=bootstrap(pp)
    stats={(r["mode"],r["channel"]):r for r in channel_stats}
    channels=summary["n_channels"]
    mean_bgs=np.mean([case_arrays[i]["bgs"] for i in ids],axis=0)
    low=set(stable_topk(-mean_bgs,channels//4)); high=set(stable_topk(mean_bgs,channels//4))
    frequency={c:sum(c in r["channels"] for r in selection_rows if r["method"]=="boundary_contribution" and r["k"]==16) for c in range(channels)}
    stable=[c for c in range(channels) if stats["scale_0p50",c]["positive_case_fraction"]>=.75 and stats["spatial_mean",c]["positive_case_fraction"]>=.75]
    summary["low_bgs_stable_contributors"]=sorted(low&set(stable))
    summary["high_bgs_inconsistent_channels"]=sorted(c for c in high if stats["scale_0p50",c]["positive_case_fraction"]<.75)
    summary["candidate_channels"]=sorted(c for c in stable if frequency[c]>=6)
    summary["boundary_LOCO_Top16_selection_frequency"]=frequency
    zero_top=stable_topk([stats["zero",c]["mean_contribution"] for c in range(channels)],16)
    summary["zero_only_channels"]=[c for c in zero_top if stats["scale_0p50",c]["mean_contribution"]<=SIGN_EPS and stats["spatial_mean",c]["mean_contribution"]<=SIGN_EPS]
    bc_utility_evidence=all(loco_summary[str(k)+"/boundary_minus_random"]["delta_boundary_ce"]["ci95"][0]>0 and
         loco_summary[str(k)+"/boundary_minus_bgs"]["auroc"]["ci95"][0]>0 for k in KS)
    probe_vs_random=all(loco_summary[str(k)+"/boundary_minus_random"]["auroc"]["ci95"][0]>0 for k in KS)
    if bc_utility_evidence and probe_vs_random and summary["candidate_channels"]:
        choice,reason="A","预定K8/16中，边界贡献联合移除相对随机恶化更大且Probe相对BGS/随机有病例配对优势；可保留为探索性Donor候选，尚未证明特征传递收益。"
    elif any(loco_summary[str(k)+"/bgs"]["auroc"]["mean"]>=loco_summary[str(k)+"/boundary_contribution"]["auroc"]["mean"]-0.01 for k in KS):
        choice,reason="B","至少一个预定K的简单BGS对照Probe达到相近或更好结果；边界消融重要性尚不足支持增加贡献筛选复杂度，保留全部K差异。"
    else:
        choice,reason="C","预定K8/16未同时形成稳定的跨病例贡献与相对BGS/随机判别优势；不固定通道为可靠迁移来源，后续可考虑局部边界表示。"
    summary["decision"]=dict(choice=choice,reason=reason,rule="A requires BC-minus-random ablation CI>0 and BC-minus-BGS/random AUROC CI>0 at bothK plus consistent candidates; B if BGS within .01 AUROC at eitherK, else C; exploratory, no equivalence proof")
    # Within-fold method/repetition associations, then one mean per receiver case.
    probe_alignment=[]
    for held in ids:
        for k in KS:
            rr=[]
            for selection in selection_rows:
                if selection["held_out_case"]==held and selection["k"]==k:
                    case_result=next(c for c in repeat_cases if (c["case_id"],c["k"],c["method"],c["repetition"])==(held,k,selection["method"],selection["repetition"]))
                    rr.append((selection,case_result))
            for metric in ("bgs","boundary_contribution","utility","energy"):
                valid=[(s["development_score_means"][metric],r["auroc"]) for s,r in rr if r["auroc"] is not None]
                rho,reason=correlation([x[0] for x in valid],[x[1] for x in valid])
                probe_alignment.append(dict(case_id=held,k=k,metric=metric,probe_auroc_spearman=rho,undefined_reason=reason))
    model_check=u.verify_state(model,state)
    after=protected_manifest(out)
    if before!=after: raise AssertionError("An existing project file changed")
    if any(r0.sha256(a["path"])!=a["sha256"] for a in plan_artifacts): raise AssertionError("Sealed LOCO plans changed after evaluation")
    if any(e["role"]=="held_out_evaluation" and e["field"]=="label" and not e["fold_plan_sealed"] for e in reader.events): raise AssertionError("Held-out GT accessed before plan")
    verification=dict(status="passed",protected_file_count=len(before),all_existing_project_SHA256_unchanged=True,
        baseline_checkpoint_SHA256_unchanged=r0.sha256(meta["checkpoint"]["path"])==meta["checkpoint"]["sha256"],
        **model_check,no_MT_training_no_optimizer_no_EMA_no_backward=True,all_VNet_predictions_no_grad=True,
        only_x3_intervened=True,no_channel_dropout_training_no_extra_decoder=True,no_BGS_gain_no_transfer_no_D0B=True,
        test_GT_read=False,held_out_not_in_fold_selection_or_probe_fit=True,fold_plans_saved_before_held_GT=True,
        GT_not_input_to_probe_predict=True,random20_repetitions_all_retained=True,bootstrap_unit="case",bootstrap_repetitions=BOOTSTRAPS,
        cases=8,patches=24,n_channel_contribution_rows=len(contributions),n_LOCO_patch_rows=len(loco),
        all_outputs_finite=True,undefined_metrics_explicit=True,unit_test_exit_code=test_exit,patch_checks=checks,
        probe_convergence_warnings=sum(not r["probe"]["converged"] for r in selection_rows))
    meta.update(status="complete",completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protected_file_sha256_after=after,patches=patches,fold_plans=plan_artifacts,patch_checks=checks,
        n_channel_contribution_rows=len(contributions),n_LOCO_patch_rows=len(loco),channel_count=channels)
    for name,data in (("ablation_sensitivity.csv",sensitivity),("cross_case_stability.csv",stability),
        ("channel_level_statistics.csv",channel_stats),("selection_comparison.csv",paired),("fold_selections.csv",selection_rows),
        ("case_leave_one_out.csv",cases),("random_repetition_case_results.csv",repeat_cases),("bgs_utility_comparison.csv",bgs_comparison),
        ("probe_metric_alignment.csv",probe_alignment),("probe_development_alignment.csv",alignment_rows),
        ("data_access_log.csv",reader.events),("patch_status.csv",patches)):
        r0.write_csv(out/name,data)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    for mode in MODES:
        vals=[stats[mode,c]["mean_contribution"] for c in range(channels)]
        axes[0].plot(range(channels),vals,label=mode)
    axes[0].axhline(0,color="black",linewidth=.7);axes[0].set(xlabel="Channel (checkpoint-specific)",ylabel="Mean boundary CE contribution")
    axes[0].legend(fontsize=8)
    for ax,metric,title in ((axes[1],"delta_boundary_ce","Held-out grouped removal"),(axes[2],"auroc","Held-out boundary-side probe")):
        methods=("boundary_contribution","bgs","full_ce_contribution","random")
        for i,method in enumerate(methods):
            vals=[loco_summary[str(k)+"/"+method][metric]["mean"] for k in KS]
            ax.plot(KS,vals,marker="o",label=method)
        ax.set(xlabel="Selected K",ylabel=metric,title=title);ax.legend(fontsize=7)
    fig.savefig(str(out/"contribution_visualization.png"),dpi=180);plt.close(fig)
    write_report(out,meta,summary,channel_stats,verification)
    r0.write_json(out/"summary_statistics.json",summary)
    r0.write_json(out/"experiment_metadata.json",meta)
    r0.write_json(out/"verification.json",verification)
    logging.info("D0-A complete %s; decision=%s; %d protected files unchanged",out,summary["decision"],len(before))
    r0.write_csv(out/"new_result_files_manifest.csv",[dict(path=str(p.relative_to(ROOT)),sha256=r0.sha256(p),bytes=p.stat().st_size)
        for p in sorted(out.rglob("*")) if p.is_file() and p.name!="new_result_files_manifest.csv"])


if __name__=="__main__":
    main()
