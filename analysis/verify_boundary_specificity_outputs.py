#!/usr/bin/env python3
"""Independently replay the labeled patches and verify Experiment 0.5 exports.

Uses scipy morphology, NumPy finite differences and scipy Spearman to check
the audit's measurements. Writes verification.json only in the new directory.
"""
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.stats import spearmanr, rankdata
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.vnet import VNet
from utils.transforms import RandomCrop

REGIONS=("near_in","near_out","far_in","far_out","foreground_core","background_core")
KINDS=("near","far","global","bgs")


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def cosine(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))


def topk(scores,k):
    return set(np.argsort(-scores,kind="stable")[:k])


@torch.no_grad()
def verify(out):
    meta=json.loads((out/"specificity_metadata.json").read_text())
    args=meta["arguments"]
    required=["specificity_summary.csv","near_far_global_cosine.csv","case_near_vectors.npz","case_far_vectors.npz",
              "case_global_vectors.npz","case_bgs_vectors.npz","bgs_pairwise_stability.csv","btv_bgs_relation.csv",
              "augmentation_bgs_stability.csv","specificity_log.txt","specificity_report.md","near_far_global_cosine.png",
              "bgs_cross_case_stability.png","btv_bgs_topk_overlap.png","near_strength_bgs_scatter.png"]
    assert all((out/f).stat().st_size>0 for f in required)
    assert meta["status"]=="complete" and meta["protected_files_unchanged"]
    assert meta["model_state_sha256_before"]==meta["model_state_sha256_after"]
    for path,digest in meta["protected_sha256_before"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    initial=Path('/tmp/boundary_specificity_initial_sha256.json')
    if initial.is_file():
        for path,digest in json.loads(initial.read_text()).items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    data={kind:np.load(out/f"case_{kind}_vectors.npz",allow_pickle=False) for kind in KINDS}
    sampled=np.load(out/"sampled_voxel_indices.npz",allow_pickle=False)
    for dataset in [*data.values(),sampled]:
        for key in dataset.files:
            if dataset[key].dtype.kind in "fc":assert np.isfinite(dataset[key]).all(),key
    near,bgs=data["near"],data["bgs"]
    cases=near["case_ids"].tolist()
    channels=near["case_vectors"].shape[1]
    n_patches=len(near["patch_ids"])
    case_status=rows(out/"case_status.csv")
    labeled_ids=Path(args["data_path"],"train.list").read_text().splitlines()[:meta["labeled_cases_requested"]]
    assert cases==sorted(labeled_ids)
    assert all(r["status"]=="complete" and int(r["valid_patches"])==args["patches_per_case"] for r in case_status)
    for kind,dataset in data.items():
        assert dataset["case_ids"].tolist()==cases
        assert dataset["patch_vectors"].shape==(n_patches,channels)
        for index,case_id in enumerate(cases):
            selected=dataset["patch_case_ids"]==case_id
            np.testing.assert_allclose(dataset["case_vectors"][index],dataset["patch_vectors"][selected].mean(axis=0))
        if kind!="bgs":
            expected=(dataset["patch_mu_in"]-dataset["patch_mu_out"])/np.sqrt((dataset["patch_var_in"]+dataset["patch_var_out"])/2+1e-6)
            np.testing.assert_allclose(dataset["patch_vectors"],expected)
        else:
            expected=(dataset["patch_g_boundary"]-dataset["patch_g_nonboundary"])/(dataset["patch_g_boundary"]+dataset["patch_g_nonboundary"]+1e-6)
            np.testing.assert_allclose(dataset["patch_vectors"],expected)
    torch.set_num_threads(args["cpu_threads"])
    model=VNet(n_channels=args["in_channels"],n_classes=args["num_classes"])
    kwargs={"weights_only":True} if "weights_only" in inspect.signature(torch.load).parameters else {}
    model.load_state_dict(torch.load(args["model_path"],map_location="cpu",**kwargs))
    model.eval();model.requires_grad_(False)
    np.random.seed(args["seed"])
    crop=RandomCrop(args["patch_size"])
    index_map={(c,int(p)):i for i,(c,p) in enumerate(zip(near["patch_case_ids"],near["patch_ids"]))}
    replayed=0
    for path,status in zip(meta["labeled_paths"],case_status):
        case_id=status["case_id"]
        with h5py.File(path,"r") as f:image,label=f["image"][:],f["label"][:]
        for attempt in range(1,int(status["attempts"])+1):
            patch=crop({"image":image,"label":label})
            if (case_id,attempt) not in index_map:continue
            pi=index_map[(case_id,attempt)]
            feature=model.encoder(torch.from_numpy(patch["image"].astype("float32"))[None,None])[2][0].numpy()
            size=feature.shape[1:]
            assert list(size)==sampled["spatial_shapes"][pi].tolist()
            coords=[np.floor(np.arange(n)*patch["label"].shape[axis]/n).astype(int) for axis,n in enumerate(size)]
            mask=patch["label"][np.ix_(*coords)].astype(bool)
            es,ds=[mask],[mask]
            for _ in range(3):
                es.append(minimum_filter(es[-1],size=3,mode="nearest"))
                ds.append(maximum_filter(ds[-1],size=3,mode="nearest"))
            regions=dict(near_in=mask & ~es[1],near_out=ds[1] & ~mask,far_in=es[2] & ~es[3],far_out=ds[3] & ~ds[2],
                         foreground_core=es[3],background_core=~ds[3])
            start,end=sampled["offsets"][pi:pi+2]
            sample_n=int(end-start)
            assert sample_n==min(int(r.sum()) for r in regions.values())
            assert sample_n>=args["min_band_voxels"]
            actual_indices={name:sampled[f"{name}_indices"][start:end] for name in REGIONS}
            for name,indices in actual_indices.items():
                assert len(np.unique(indices))==sample_n
                assert regions[name].reshape(-1)[indices].all()
            flat=feature.reshape(channels,-1).astype("float64")
            for kind,a,b in (("near","near_in","near_out"),("far","far_in","far_out"),("global","foreground_core","background_core")):
                inside,outside=flat[:,actual_indices[a]],flat[:,actual_indices[b]]
                expected=(inside.mean(axis=1)-outside.mean(axis=1))/np.sqrt((inside.var(axis=1)+outside.var(axis=1))/2+1e-6)
                np.testing.assert_allclose(expected,data[kind]["patch_vectors"][pi],rtol=1e-6,atol=1e-7)
            inside,outside=flat[:,regions["near_in"].reshape(-1)],flat[:,regions["near_out"].reshape(-1)]
            full=(inside.mean(axis=1)-outside.mean(axis=1))/np.sqrt((inside.var(axis=1)+outside.var(axis=1))/2+1e-6)
            np.testing.assert_allclose(full,near["patch_full_band_vectors"][pi],rtol=1e-6,atol=1e-7)
            differences=[]
            for axis in (1,2,3):
                pads=[(0,0)]*4;pads[axis]=(0,1)
                differences.append(np.pad(abs(np.diff(feature,axis=axis)),pads,mode="edge"))
            gradient=(sum(differences)/3).reshape(channels,-1).astype("float64")
            boundary=ds[1] & ~es[1]
            nonboundary=~maximum_filter(boundary,size=3,mode="nearest")
            gb,gn=gradient[:,boundary.reshape(-1)].mean(axis=1),gradient[:,nonboundary.reshape(-1)].mean(axis=1)
            expected=(gb-gn)/(gb+gn+1e-6)
            np.testing.assert_allclose(expected,bgs["patch_vectors"][pi],rtol=1e-6,atol=1e-7)
            replayed+=1
    assert replayed==n_patches
    reference=np.load(Path(args["reference_output_dir"])/"case_transition_vectors.npz",allow_pickle=False)
    for pi,(case_id,patch_id) in enumerate(zip(near["patch_case_ids"],near["patch_ids"])):
        index=np.flatnonzero((reference["x3_patch_case_ids"]==case_id)&(reference["x3_patch_ids"]==patch_id))
        assert len(index)==1
        np.testing.assert_allclose(near["patch_full_band_vectors"][pi],reference["x3_patch_d"][index[0]],rtol=1e-6,atol=1e-7)
    transition=rows(out/"near_far_global_cosine.csv")
    for row in transition:
        a,b,_=row["comparison"].split("_")
        ci=cases.index(row["case_id"])
        if row["unit"]=="patch":
            pi=index_map[(row["case_id"],int(row["patch_id"]))]
            expected=cosine(data[a]["patch_vectors"][pi],data[b]["patch_vectors"][pi])
        elif row["unit"]=="case_mean_of_patch_cosines":
            selected=np.flatnonzero(near["patch_case_ids"]==row["case_id"])
            expected=np.mean([cosine(data[a]["patch_vectors"][pi],data[b]["patch_vectors"][pi]) for pi in selected])
        else:expected=cosine(data[a]["case_vectors"][ci],data[b]["case_vectors"][ci])
        np.testing.assert_allclose(float(row["cosine"]),expected)
    for row in rows(out/"bgs_pairwise_stability.csv"):
        i,j,k=cases.index(row["case_i"]),cases.index(row["case_j"]),int(row["k"])
        va,vb=bgs["case_vectors"][[i,j]]
        np.testing.assert_allclose(float(row["spearman_bgs"]),spearmanr(va,vb).statistic)
        a,b=topk(va,k),topk(vb,k)
        np.testing.assert_allclose(float(row["jaccard_bgs"]),len(a&b)/len(a|b))
    for row in rows(out/"btv_bgs_relation.csv"):
        i,k=cases.index(row["case_id"]),int(row["k"])
        a,b=abs(near["case_vectors"][i]),bgs["case_vectors"][i]
        np.testing.assert_allclose(float(row["spearman_abs_near_bgs"]),spearmanr(a,b).statistic)
        sa,sb=topk(a,k),topk(b,k)
        np.testing.assert_allclose(float(row["topk_jaccard"]),len(sa&sb)/len(sa|sb))
    aug_map={(c,int(p),v):i for i,(c,p,v) in enumerate(zip(bgs["aug_case_ids"],bgs["aug_patch_ids"],bgs["aug_names"]))}
    for row in rows(out/"augmentation_bgs_stability.csv"):
        va,vb=[bgs["aug_vectors"][aug_map[(row["case_id"],int(row["patch_id"]),row[key])]] for key in ("augmentation_i","augmentation_j")]
        np.testing.assert_allclose(float(row["spearman_bgs"]),spearmanr(va,vb).statistic)
        np.testing.assert_allclose(float(row["cosine_bgs"]),cosine(va,vb))
        sa,sb=topk(va,int(row["k"])),topk(vb,int(row["k"]))
        np.testing.assert_allclose(float(row["jaccard_bgs"]),len(sa&sb)/len(sa|sb))
    for row in rows(out/"channel_specificity.csv"):
        c=int(row["channel"])
        mn,mb=abs(near["case_vectors"]).mean(axis=0),bgs["case_vectors"].mean(axis=0)
        np.testing.assert_allclose(float(row["mean_abs_d_near"]),mn[c])
        np.testing.assert_allclose(float(row["mean_bgs"]),mb[c])
        assert float(row["d_near_rank"])==rankdata(-mn)[c]
        assert float(row["bgs_rank"])==rankdata(-mb)[c]
    summary=rows(out/"specificity_summary.csv")
    assert all(int(r["n_undefined"])==0 for r in summary)
    result=dict(status="passed",numerical_unit_tests=11,labeled_cases=len(cases),replayed_identity_patches=replayed,
                matched_voxel_min=int(near["sample_counts"].min()),matched_voxel_max=int(near["sample_counts"].max()),
                nonfinite_vector_elements=0,baseline_and_experiment0_unchanged=True,
                experiment0_full_near_patch_vectors_reproduced=True,
                independent_checks=["scipy morphology and nearest resize coordinates","equal unique voxel indices inside six disjoint regions",
                                    "NumPy sampled standardized transitions from replayed x3 features","NumPy gradients and BGS from replayed features",
                                    "case means and saved moments","all patch/case cosines","signed BGS Spearman and Top-K",
                                    "BTV/BGS relationships","all augmentation pair scores","channel mean scores and ranks"])
    (out/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir",type=Path,default=ROOT/"analysis/results/LA_10pct_seed42_specificity")
    verify(p.parse_args().output_dir)
