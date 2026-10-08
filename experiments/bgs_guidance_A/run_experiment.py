#!/usr/bin/env python3
"""Run the single authorized LA/seed42/alpha0.1 experiment through evaluation."""
import argparse
import contextlib
import csv
from datetime import datetime
import hashlib
import inspect
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def validation_history(path):
    pattern = r'iteration (\d+) : mean_dice : ([\d.]+) mean_hd95 : ([\d.]+)'
    return [dict(iteration=int(m[1]), dice=float(m[2]), hd95=float(m[3]))
            for m in re.finditer(pattern, path.read_text())]


def verify_protected():
    manifest = json.loads((ROOT / "experiments/bgs_guidance_A/protected_files_before.json").read_text())
    changed = [p for p,h in manifest.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != h]
    if changed:
        raise RuntimeError(f"Protected baseline/audit files changed: {changed}")
    return len(manifest)


def evaluate(checkpoint, device, output_prefix):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from model.vnet import VNet
    from prediction import test_calculate_metric
    args = SimpleNamespace(data_path=str(ROOT.parent / "Dataset/LA"), dataset="LA",
                           patch_size=[112,112,80], num_classes=2, in_channels=1, nms=True)
    model = VNet(n_channels=1, n_classes=2).to(device).eval()
    kwargs = {"weights_only": True} if "weights_only" in inspect.signature(torch.load).parameters else {}
    model.load_state_dict(torch.load(checkpoint, map_location=device, **kwargs))
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), torch.no_grad():
        validation_dice, validation_hd95 = test_calculate_metric(args, model, val=False)
    output_prefix.with_suffix(".log").write_text(capture.getvalue())
    cases = (Path(args.data_path)/"test.list").read_text().splitlines()
    measurements = []
    for line in capture.getvalue().splitlines():
        tokens = line.split()
        if len(tokens) != 4:
            continue
        try:
            values = [float(t) for t in tokens]
        except ValueError:
            continue
        measurements.append(values)
    if len(measurements) != len(cases):
        raise RuntimeError("Could not recover every per-case metric from unchanged prediction.py")
    with output_prefix.with_suffix(".csv").open("w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id","dice","jaccard","hd95","asd"])
        writer.writerows([case,*values] for case,values in zip(cases,measurements))
    means = np.mean(np.array(measurements),axis=0)
    if abs(means[0]-validation_dice)>1e-8 or abs(means[2]-validation_hd95)>1e-8:
        raise RuntimeError("Parsed metrics disagree with original prediction return values")
    return dict(n_cases=len(cases),dice=float(means[0]),jaccard=float(means[1]),hd95=float(means[2]),asd=float(means[3]))


def report(run_dir, metadata, baseline_best, guidance_best, baseline_test, guidance_test):
    delta = guidance_test["dice"] - baseline_test["dice"]
    if delta > 0:
        conclusion = f"本次单 seed 试验 Dice 提升 {delta*100:.3f} 个百分点，对 labeled signature 的 unlabeled transfer utility 提供初步支持。"
    else:
        conclusion = f"本次单 seed 试验 Dice 变化 {delta*100:.3f} 个百分点，未观察到该 alpha=0.1 设置的 transfer utility 提升。"
    lines = ["# Experiment A — Labeled Boundary-Selective Channel Guidance", "", conclusion, "",
             "仅运行 LA 10%、seed 42、alpha=0.1，从相同 seed 初始化训练 30,000 iterations。"
             "batch=2 labeled+2 unlabeled，patch=112×112×80，每 200 iterations 原流程验证。", "",
             "| 模型 | best validation Dice | iteration | HD95 |", "|---|---:|---:|---:|",
             f"| MT baseline | {baseline_best['dice']:.6f} | {baseline_best['iteration']} | {baseline_best['hd95']:.6f} |",
             f"| BGS guidance A | {guidance_best['dice']:.6f} | {guidance_best['iteration']} | {guidance_best['hd95']:.6f} |", "",
             "| 最佳 EMA checkpoint 的最终评估 | Dice | Jaccard | HD95 | ASD |", "|---|---:|---:|---:|---:|"]
    for name,values in (("MT baseline",baseline_test),("BGS guidance A",guidance_test)):
        lines.append(f"| {name} | {values['dice']:.6f} | {values['jaccard']:.6f} | {values['hd95']:.6f} | {values['asd']:.6f} |")
    lines += ["", "LA 原代码 validation 和最终 test 使用相同 20 例 test.list。上述最终评估是相同病例上的重评估，"
              "不构成独立 held-out test；本轮按要求保持 split、validation 和 inference 原样。单 seed 的差值不能建立统计显著性。", "",
              "BGS 仅由当前 Student x3 的 labeled prefix 和 labeled GT 计算，ROI 与 finite difference 定义和 Experiment 0.5 一致。"
              "R detach，signed 正值归一化生成全部 channel 的连续 w。只对 unlabeled x3 应用 (1+0.1w)，无 Top-K hard selection。"
              "Teacher、CE+Dice、概率 MSE、optimizer、scheduler、EMA、augmentation 和 inference 均保留 baseline。", "",
              "decoder 的原有 BatchNorm 在整个 batch 上计算统计量。因此，虽然 labeled x3 完全不变，"
              "unlabeled x3 的缩放仍可能间接影响 labeled logits；保留 BatchNorm 原行为，本实验不能单独隔离这种影响。", "",
              "恢复包装脚本只在原 validation 完成后保存 Student、Teacher、optimizer、scheduler、iteration 和 RNG 状态。"
              "baseline 的训练源码未因恢复机制而改变；新 run 从 seed 42 重新初始化，旧的中断记录已独立保留。", "",
              "默认 use_bgs_guidance=False 直接执行 self.model(volume_batch)。数值测试覆盖禁用/alpha=0 完整 step 与原始 Trainer"
              "的 loss、Student/Teacher 参数和 buffers 完全一致；覆盖 labeled x3、detach、Teacher 无梯度和非法数值。", "",
              "完整训练日志：result_LA_10l/fold_0/log.txt；stdout.log 保存控制台输出；validation_history.csv 保存所有验证点；"
              "baseline_test/guidance_test 的 CSV 和 log 保存每例四项指标。metadata.json 和 protected_files_after.json 记录配置与校验。"
              "原始 train.py/trainer.py 已归档于 experiments/bgs_guidance_A/baseline_source。未实施 A+。", ""]
    (run_dir/"experiment_report.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--output_dir",type=Path,default=ROOT/"Results/experiment_A_bgs_alpha0p1_seed42")
    args = p.parse_args()
    run_dir = args.output_dir.resolve()
    if run_dir.exists():
        raise FileExistsError("Experiment output directory already exists; refusing to overwrite")
    run_dir.mkdir(parents=True)
    baseline_log = ROOT/"Results/seed_42/result_LA_10l/fold_0/log.txt"
    baseline_best = max(validation_history(baseline_log),key=lambda r:r["dice"])
    snapshot = run_dir/"result_LA_10l/fold_0"
    entrypoint = ROOT/"experiments/bgs_guidance_A/checkpointed_train.py"
    (run_dir/"checkpointed_train.py").write_bytes(entrypoint.read_bytes())
    command = [sys.executable,str(entrypoint),"--data_path",str(ROOT.parent/"Dataset/LA"),
               "--dataset","LA","--labeled_num","10","--patch_size","112","112","80",
               "--batch_size","4","--labeled_bs","2","--seed","42","--base_lr","0.01",
               "--max_iterations","30000","--test_interval","200","--n_fold","1",
               "--consistency_rampup","200","--consistency","0.1","--ema_decay","0.9",
               "--device",args.device,"--output_dir",str(run_dir),"--use_bgs_guidance","--bgs_alpha","0.1"]
    metadata = dict(command=command,device=args.device,stage="starting",start_time=datetime.now().isoformat(),
                    cuda_visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
                    recovery_entrypoint_sha256=hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
                    protected_files=verify_protected(),baseline_best_validation=baseline_best,
                    training_source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                            for name in ["train.py","trainer.py","utils/boundary_guidance.py"]})
    write_json(run_dir/"metadata.json",metadata)
    write_json(run_dir/"run_status.json",dict(stage="starting",max_iterations=30000))
    print(f"Experiment A starting; output={run_dir}",flush=True)
    try:
        start = time.monotonic()
        with (run_dir/"stdout.log").open("w") as stdout:
            process = subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=subprocess.STDOUT)
            while True:
                try:
                    code = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    code = None
                progress = dict(stage="training",pid=process.pid,max_iterations=30000,
                                elapsed_seconds=time.monotonic()-start)
                log = snapshot/"log.txt"
                if log.is_file():
                    content = log.read_text()
                    iterations = re.findall(r'iteration (\d+) : loss :',content)
                    progress["last_completed_iteration"] = int(iterations[-1]) if iterations else None
                    history = validation_history(log)
                    if history:
                        progress["latest_validation"] = history[-1]
                        progress["best_validation"] = max(history,key=lambda r:r["dice"])
                write_json(run_dir/"run_status.json",progress)
                if code is not None:
                    if code:
                        raise RuntimeError(f"Training exited with status {code}; see stdout.log")
                    break
        history = validation_history(snapshot/"log.txt")
        if not history or history[-1]["iteration"] != 30000 or progress.get("last_completed_iteration") != 29999:
            raise RuntimeError("Training did not finish all 30,000 iterations/required validation")
        with (run_dir/"validation_history.csv").open("w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=["iteration","dice","hd95"])
            writer.writeheader();writer.writerows(history)
        guidance_best = max(history,key=lambda r:r["dice"])
        write_json(run_dir/"run_status.json",dict(stage="baseline_evaluation",best_validation=guidance_best))
        baseline_checkpoint = baseline_log.parent/f"Model_iter_{baseline_best['iteration']}.pth"
        baseline_test = evaluate(baseline_checkpoint,args.device,run_dir/"baseline_test")
        write_json(run_dir/"run_status.json",dict(stage="guidance_evaluation",best_validation=guidance_best))
        guidance_checkpoint = snapshot/f"Model_iter_{guidance_best['iteration']}.pth"
        guidance_test = evaluate(guidance_checkpoint,args.device,run_dir/"guidance_test")
        if abs(guidance_test["dice"]-guidance_best["dice"])>1e-5:
            raise RuntimeError("Final inference does not reproduce best validation Dice")
        metadata.update(stage="complete",end_time=datetime.now().isoformat(),best_validation=guidance_best,
                        baseline_test=baseline_test,guidance_test=guidance_test,
                        guidance_checkpoint=str(guidance_checkpoint),protected_files_unchanged=bool(verify_protected()))
        write_json(run_dir/"metadata.json",metadata)
        protected = json.loads((ROOT/"experiments/bgs_guidance_A/protected_files_before.json").read_text())
        write_json(run_dir/"protected_files_after.json",{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected})
        report(run_dir,metadata,baseline_best,guidance_best,baseline_test,guidance_test)
        write_json(run_dir/"run_status.json",dict(stage="complete",best_validation=guidance_best,
                   baseline_test=baseline_test,guidance_test=guidance_test))
        print(json.dumps(dict(stage="complete",best_validation=guidance_best,guidance_test=guidance_test)),flush=True)
    except BaseException as e:
        metadata.update(stage="failed",error=str(e),traceback=traceback.format_exc())
        write_json(run_dir/"metadata.json",metadata)
        write_json(run_dir/"run_status.json",dict(stage="failed",error=str(e)))
        raise


if __name__=="__main__":
    main()
