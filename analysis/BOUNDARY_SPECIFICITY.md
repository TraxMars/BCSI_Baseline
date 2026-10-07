# Experiment 0.5: Boundary Specificity Audit

脚本为 `boundary_specificity_audit.py`，独立读取 Experiment 0 的相同 checkpoint、
LA 10% labeled 前缀（8 个病例）、seed=42 和 `112 112 80` patch，只分析 x3。
不会修改 baseline、旧分析脚本或旧结果；输出使用新目录。

```bash
cd /home/duanxuelong/BCSI_Baseline_BGCE
conda run --no-capture-output -n SSL python analysis/boundary_specificity_audit.py \
  --matching_policy all_six_min --device cpu \
  --output_dir analysis/results/LA_10pct_seed42_specificity
```

用户已确认采用 `all_six_min`。原附件要求 near/far 四条 band 取最小 voxel 数，
global core 也匹配这个数量。实际预检查每病例 200 次 crop，所有 8 个病例的
foreground core 都小于四条 band 的最小值，严格规则得到 0 个有效 patch。
脚本保留原规则为默认 `strict_four_bands`，用显式参数选择确认后的规则：
六个区域统一取最小值，无放回等量随机抽样。原先 24 个 patch 可获得
12–284 个 voxel/区域，均达到默认最少 8 个。

near/far/core morphology 完全遵循附件，数字代表连续 3×3×3 pooling 次数。
每个 region 的 sample index 保存于 `sampled_voxel_indices.npz`，使用 x3
spatial grid 的 flatten index。所有六个 ROI 必须互不重叠；不足时重采样，
最多 200 次/病例。full-band near d 也保存，可检查与 Experiment 0 相同 patch
的一致性。crop 使用原始 `RandomCrop` 和独立的 legacy NumPy seed=42；
voxel 抽样使用单独 Generator，不改变 crop 序列；bootstrap/null 使用第三个 RNG。

d 使用 population variance、eps=1e-6。主 cosine 统计先按 patch 计算、
再求病例内均值，最终按病例 mean/std/median/95% CI 汇总。另存
cosine(case mean vectors) 作为补充，不与主统计混合。

BGS 使用三方向 forward difference 的绝对值平均，终端平面复制最后一个有效
差分以保持原 feature shape。gradient 均值分别取完整 GT boundary region 和
排除紧邻边界的 non-boundary control。BGS 使用原始 signed 值从高到低排名。
`abs(BGS)` 会把强负 selectivity 误认成强边界响应，不用于排名或 Spearman。
Top-K 的 K=round(C×ratio)，ties 按原始 channel index 稳定排序。

case-level d/BGS 都先对 raw patch vectors 取均值；BTV 关系使用
abs(case_d_near) 与 signed case_BGS。通道表的 mean abs(d_near) 为
mean(abs(case_d_near))；rank 为按跨病例平均 score 的降序排名，1 最高。
12/24/35/49 只作为报告标注对象，完全不参与采样、计算或模型选择。

跨病例 BGS pair 使用病例 bootstrap，不把多个相关 pair 当独立样本。
Top-K 随机基线为同 C/K 的 5000 对 subset。另提供 BGS channel-shuffle
Spearman 基线，200 次/pair。augmentation 使用原定义的 identity、两个 flip
和 90°旋转，image/label 同步，无 intensity noise，独立汇总。CI 针对均值；
augmentation 按病例 cluster bootstrap。所有未定义统计显式计数，NaN/Inf
原始向量逐 channel 记录后失败。

必需 CSV、四类 case/patch NPZ、日志、报告和四张图全部输出；额外保存
sampled indices、moments、case/patch 状态、完整 channel 表、随机 null 和
metadata。新旧目录必须分开且新目录为空。执行前后校验 baseline、checkpoint、
旧脚本和所有 Experiment 0 文件 SHA256，以及 model 参数/buffers。
脚本还校验本次设置与 Experiment 0 的 metadata 一致。

报告明确讨论附件要求的 A/B/C；不使用硬阈值自动宣称 boundary specificity。
正式结果必须同时考虑 near-global cosine、BGS 正响应、ranking、随机基线、
增强稳定性和 core 样本量。不会实现后续 segmentation module。

数值验证：

```bash
conda run --no-capture-output -n SSL python analysis/test_boundary_specificity_audit.py
```

独立核验会重放全部 labeled identity patch，以 scipy morphology、NumPy finite
differences 和 scipy Spearman 重算导出数据，并写入新目录的 verification.json：

```bash
conda run --no-capture-output -n SSL python analysis/verify_boundary_specificity_outputs.py
```

本次人工解释后的报告保存在 `results/LA_10pct_seed42_specificity/specificity_report.md`。
另有只基于已保存 vectors 的 post hoc 采样敏感性检查，独立存为
`near_sampling_sensitivity.csv` 和 `supplementary_summary.csv`，未增加模型更新或改变
主实验指标。复跑脚本会生成数据和 A/B/C 解释框架；正式研究判断需要结合输出核验。
