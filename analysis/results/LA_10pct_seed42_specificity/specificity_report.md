# Experiment 0.5 — Boundary Specificity Audit

结果整体更接近 **A 的证据模式：near transition 与远处的 semantic control 不同，且存在可复现的 boundary-selective channels**。
同时，BTV 强度与 BGS 排名相关性接近零，Experiment 0 的稳定高 BTV channel 并不普遍具有强边界梯度选择性。
因此，这组结果提供边界相关信息的支持，**尚不足以确认 BTV 整体是边界专属表征**。以下是结合实际数值和对照作出的研究推断，没有使用硬阈值自动分类。

沿用 Experiment 0 的 LA 10%、seed=42、checkpoint `Model_iter_27000.pth`、patch=[112, 112, 80]；只分析 x3，实际 feature shape=[[1, 64, 28, 28, 20]]。
有效病例=8，跳过病例=0，partial 病例=0，有效 identity patch=24。

采样规则经用户确认采用 `all_six_min`：四条 near/far band 和两个 global core 的 voxel 数统一取最小值，六个区域各自无放回等量抽样。原协议的四条 band 最小值在当前 LA/x3 上超过 foreground core 容量；预检查每病例 200 次 crop，8 个病例均无法按原数量匹配。该调整保持六组统计的样本数一致，不改变 ring/core 定义；实际 counts 和 sampled indices 均保存。此次每个 region 抽样 12–284 个 voxel，24 个 patch 均为各病例的前 3 次 crop。

以下 CI 都是均值的 95% bootstrap CI。先在每个 patch 计算 signed cosine，再对同一病例的 patch cosine 求均值，最后按病例汇总和 bootstrap；`cosine_of_case_mean_vectors` 是独立保存的补充统计，不能与主统计混用。

| 主比较 | mean [95% CI] |
|---|---:|
| near vs far | 0.523 [0.477, 0.565] |
| near vs global | 0.496 [0.436, 0.544] |
| far vs global | 0.878 [0.842, 0.913] |

far–global 的相似度高于 near–global。利用已保存向量作补充的病例配对比较，差值均值为
0.382，病例 bootstrap 95% CI 为 `[0.339,0.430]`。near–global 的 0.496 是中等相似度，
仍包含共享方向，不能解释为“没有 semantic contrast”；它与远处对照的方向一致性明显低于 far–global。

BGS 使用 signed 值排名，正值表示 boundary 的平均 feature gradient 高于 non-boundary control。按 abs(BGS) 选 Top-K 会误选强负值，本实现不这样做。BGS 的 spatial gradient 是三个方向 forward difference绝对值的平均，终端平面复制最后一个有效差分；使用全部 boundary/nonboundary voxel 求均值。

| BGS 指标 | real mean [95% CI] | random mean [95% CI] |
|---|---:|---:|
| 跨病例 signed BGS Spearman | 0.578 [0.493, 0.677] | -0.001 [-0.006, 0.003] |
| 跨病例 Top-12.5% Jaccard | 0.335 [0.233, 0.495] | 0.071 [0.069, 0.073] |
| 跨病例 Top-25% Jaccard | 0.411 [0.322, 0.511] | 0.146 [0.144, 0.148] |

mean BGS：0.039 [0.026, 0.056]；正 BGS channel 比例：0.691 [0.648, 0.742]。

| 同一 patch 的 BGS augmentation 指标 | mean [95% CI] |
|---|---:|
| bgs_spearman | 0.832 [0.804, 0.857] |
| bgs_cosine | 0.864 [0.841, 0.888] |
| Top-12.5% Jaccard | 0.545 [0.493, 0.592] |
| Top-25% Jaccard | 0.589 [0.554, 0.619] |

病例级 Spearman(abs(d_near), BGS)：-0.032 [-0.102, 0.028]。
BTV / BGS Top-12.5% Jaccard：0.070 [0.026, 0.115]。
BTV / BGS Top-25% Jaccard：0.134 [0.106, 0.160]。

上述 overlap 与同 C/K 的随机 subset 均值 0.071 / 0.146 接近，未显示高 BTV 排名集合与高 BGS
排名集合的一致性。Spearman 的 CI 包含 0，也不支持整体 channel 强度排名与 gradient selectivity 排名的关联。

以下 12/24/35/49 只作为 Experiment 0 的解释对象，未用于采样、feature 选择或任何模型操作。rank 按跨病例平均后的 channel score 排名，1 为最高；mean abs(d_near) 取 mean(abs(case_d))，case_d 先对原始 patch d 求均值。

| channel | mean abs(d_near) | mean BGS | d_near rank | BGS rank | 正 BGS 病例比例 |
|---|---:|---:|---:|---:|---:|
| 12 | 0.9538 | -0.0187 | 3 | 52 | 0.500 |
| 24 | 0.8334 | 0.1370 | 5 | 7 | 1.000 |
| 35 | 1.0132 | 0.0329 | 1 | 36 | 0.750 |
| 49 | 0.9156 | -0.0358 | 4 | 56 | 0.125 |

channel 24 同时具有较强 near transition 和较高 BGS，8/8 个病例为正，是这四个解释对象中最明确的
boundary-selective 候选。12 的平均 BGS 为负且仅 4/8 个病例为正；49 平均 BGS 为负，仅 1/8 个病例为正；
35 的 BGS 为弱正值，均值排名 36/64。它们在 Experiment 0 中的高 BTV 排名本身不能作为边界选择性的证据。
这些结果只用于解释，不对 channel 进行模型选择或操作。

三种可能结合本次结果讨论如下：

A. near–global similarity 不高且 BGS 稳定，支持 boundary-specific representation。
本次 near–global 为 0.496，低于 far–global 的 0.878；BGS 跨病例 Spearman 为 0.578，Top-K Jaccard
均值差相对随机基线的 95% CI 均为正，增强 Spearman 为 0.832。这些观察整体更符合 A 的证据模式，
支持 x3 中存在一定边界相关方向和可复现的 boundary-selective channels。

B. near–global similarity 高且 BGS 稳定，BTV 更像 semantic class contrast，同时存在 boundary-selective channels。
本次确有可复现的 BGS，但 near–global 的相似度未达到 far–global 对照所显示的高度一致性，因此不能将
整个 near BTV 归结为普通 foreground/background 对比。另一方面，near–global 仍有中等相似度，且 BTV/BGS
排名基本无关，说明语义对比与边界梯度选择性可能共同存在，不能把所有高 abs(d_near) channel 解释为边界专属。

C. near–global similarity 高且 BGS 不稳定，global channel-level boundary representation 缺乏支持，可在后续研究考虑 local boundary modeling。
本次 BGS 排名和 Top-K 稳定性高于随机基线，增强结果也显示可复现性，未呈现 C 所要求的整体 BGS 不稳定模式。
该实验没有比较 local representation，不能据此判断局部建模是否更好或是否必要。

上述“高/稳定”没有预注册硬阈值。这里的判断同时使用对照方向、完整 ranking、Top-K 相对随机基线、
增强稳定性和正 BGS 响应，结论限定于当前 labeled 病例和统计定义。

等量采样噪声的补充检查仅使用已保存向量，不增加 forward 或改变采样：balanced near 与 full-band near 的
patch cosine 先取病例均值后汇总为 0.917 `[0.870,0.957]`；full-band near 与同一 matched global 的 cosine
为 0.531 `[0.495,0.569]`。采用完整 near band 后，near–global 相似度仍远低于 far–global 的 0.878。
这减轻了“near–global 较低完全来自 near 小样本噪声”的疑虑，但没有排除 global core 的样本量及空间位置效应。
补充统计与主统计分开保存在 `near_sampling_sensitivity.csv` 和 `supplementary_summary.csv`，bootstrap seed=43。

该审计限于一个 checkpoint 和 8 个 labeled 病例。平均 BGS 仅为 0.039，整体 selectivity 的效应有限，
更适合解释为部分 channel 的选择性。等量 sampling 的样本数受 foreground core 限制，可能提高 d 的采样噪声；
near/far/core 采用 pooling 的离散网格距离，并非物理距离，这些 core 也不保证 feature 的感受野完全避开边界。
forward difference 的锚点及 nearest resize 相位会影响几何增强比较。BGS 的全局 non-boundary control
包含前景和背景不同组织，本身也不是边界专属机制的因果证明。没有跨 seed、跨数据集或未见病例验证。

保存原始 patch/case vectors、moments、BGS 分子所需的梯度均值和 voxel indices。NaN/Inf 按 case/patch/view/channel 明确报错；常数 ranking/零向量产生的无定义相关性单独记录。baseline 代码、checkpoint 和 Experiment 0 目录以执行前后 SHA256 校验；model 参数与 buffers 也校验。全程 eval/no_grad，不引入训练或 segmentation module。

11 项数值测试通过。独立 verifier 重新采样并 forward 全部 24 个 identity patch，用 scipy morphology、
NumPy gradients 和 scipy Spearman 重算了实际 ROI、等量抽样、transition、BGS、病例均值、pair scores 和排名，均一致。
full-band near 的所有 24 个 patch d 与 Experiment 0 导出的 x3 patch d 一致，确认实际使用相同病例和 patch。
所有原始向量均无 NaN/Inf、所有汇总无 undefined 项，33 个 protected 文件保持一致。
详见 [verification.json](verification.json)。

文件：specificity_summary.csv、near_far_global_cosine.csv、case_*_vectors.npz、bgs_pairwise_stability.csv、btv_bgs_relation.csv、augmentation_bgs_stability.csv、channel_specificity.csv、case_status.csv、patch_status.csv、sampled_voxel_indices.npz、random_baselines.csv、specificity_metadata.json、specificity_log.txt 及四张 PNG。

复跑使用新 output_dir，命令示例：

```bash
conda run --no-capture-output -n SSL python analysis/boundary_specificity_audit.py \
  --matching_policy all_six_min --device cpu \
  --output_dir analysis/results/LA_10pct_seed42_specificity_rerun
```

独立核验可执行：

```bash
conda run --no-capture-output -n SSL python analysis/verify_boundary_specificity_outputs.py
```
