# Experiment A — Labeled Boundary-Selective Channel Guidance

本次单 seed 试验 Dice 变化 -1.070 个百分点，未观察到该 alpha=0.1 设置的 transfer utility 提升。

仅运行 LA 10%、seed 42、alpha=0.1，从相同 seed 初始化训练 30,000 iterations。batch=2 labeled+2 unlabeled，patch=112×112×80，每 200 iterations 原流程验证。

| 模型 | best validation Dice | iteration | HD95 |
|---|---:|---:|---:|
| MT baseline | 0.873561 | 27000 | 8.888493 |
| BGS guidance A | 0.862856 | 29000 | 11.416208 |

| 最佳 EMA checkpoint 的最终评估 | Dice | Jaccard | HD95 | ASD |
|---|---:|---:|---:|---:|
| MT baseline | 0.873561 | 0.777685 | 8.888792 | 2.208702 |
| BGS guidance A | 0.862858 | 0.764721 | 11.416208 | 3.093234 |

LA 原代码 validation 和最终 test 使用相同 20 例 test.list。上述最终评估是相同病例上的重评估，不构成独立 held-out test；本轮按要求保持 split、validation 和 inference 原样。单 seed 的差值不能建立统计显著性。

BGS 仅由当前 Student x3 的 labeled prefix 和 labeled GT 计算，ROI 与 finite difference 定义和 Experiment 0.5 一致。R detach，signed 正值归一化生成全部 channel 的连续 w。只对 unlabeled x3 应用 (1+0.1w)，无 Top-K hard selection。Teacher、CE+Dice、概率 MSE、optimizer、scheduler、EMA、augmentation 和 inference 均保留 baseline。

decoder 的原有 BatchNorm 在整个 batch 上计算统计量。因此，虽然 labeled x3 完全不变，unlabeled x3 的缩放仍可能间接影响 labeled logits；保留 BatchNorm 原行为，本实验不能单独隔离这种影响。

恢复包装脚本只在原 validation 完成后保存 Student、Teacher、optimizer、scheduler、iteration 和 RNG 状态。baseline 的训练源码未因恢复机制而改变；新 run 从 seed 42 重新初始化，旧的中断记录已独立保留。

默认 use_bgs_guidance=False 直接执行 self.model(volume_batch)。数值测试覆盖禁用/alpha=0 完整 step 与原始 Trainer的 loss、Student/Teacher 参数和 buffers 完全一致；覆盖 labeled x3、detach、Teacher 无梯度和非法数值。

完整训练日志：result_LA_10l/fold_0/log.txt；stdout.log 保存控制台输出；validation_history.csv 保存所有验证点；baseline_test/guidance_test 的 CSV 和 log 保存每例四项指标。metadata.json 和 protected_files_after.json 记录配置与校验。原始 train.py/trainer.py 已归档于 experiments/bgs_guidance_A/baseline_source。未实施 A+。
