# Experiment A-D — Failure Mechanism Diagnosis

本诊断没有训练、调参或实现 A+。正式配对汇总保留全部 20 个病例，异常病例没有被排除。Baseline 与 Experiment A 的 checkpoint 分别为 iteration 27000 和 29000。

## 已经直接验证的事实

- 五个受保护源文件在诊断前后 SHA256 均未变化：`True`。
- 异常病例重新执行了原始滑窗推理、0.5 阈值和最大连通域后处理；重算指标与已有逐病例 CSV 一致。
- BatchNorm 检查全程使用 `torch.no_grad()`，没有反向传播或参数更新；完成后模型参数与 buffers 逐张量恢复。

### 全部 20 例 paired metric delta

`delta = guidance - baseline`；HD95/ASD 的负 delta 表示改善。`utility delta` 已统一为正值表示改善。

| metric | improved | worsened | tie | mean raw delta | paired median raw delta | worst case | worst utility delta | LOO mean utility range |
|---|---|---|---|---|---|---|---|---|
| dice | 9 | 11 | 0 | -0.010703 | -0.001247 | WSJB9P4JCXUVHBOYFVWL | -0.189756 | [-0.012308, -0.001279] |
| jaccard | 9 | 11 | 0 | -0.012963 | -0.002105 | WSJB9P4JCXUVHBOYFVWL | -0.226990 | [-0.015396, -0.001699] |
| hd95 | 8 | 11 | 1 | 2.527417 | 0.578823 | WSJB9P4JCXUVHBOYFVWL | -36.506427 | [-2.742982, -0.739048] |
| asd | 9 | 11 | 0 | 0.884533 | 0.024192 | WSJB9P4JCXUVHBOYFVWL | -15.306145 | [-0.955895, -0.125500] |

leave-one-out 只用于敏感性分析；正式均值、median 和病例计数均使用完整 20 例。

### 预先指定异常病例 `WSJB9P4JCXUVHBOYFVWL`

- GT 前景体积：171083 voxels。
- Baseline 预测体积：138050 voxels （GT 比例 0.8069）。
- Guidance 预测体积：230572 voxels （GT 比例 1.3477）。
- Baseline FP/FN：14649 / 47682 voxels。
- Guidance FP/FN：108346 / 48857 voxels。

| metric | baseline | guidance | delta |
|---|---|---|---|
| dice | 0.798368 | 0.608612 | -0.189756 |
| jaccard | 0.664404 | 0.437413 | -0.226990 |
| hd95 | 17.233688 | 53.740115 | 36.506427 |
| asd | 2.705241 | 18.011386 | 15.306145 |

3D component 明细、原始阈值 mask、最大连通域后 mask、FP/FN 与关键切片图保存在 `failure_visualizations/`。数据没有 voxel spacing，因此体积只报告 voxel 数，不伪造物理体积。

### BGS 训练日志

- 共解析 30000 个 iteration；guidance skip rate=0.000000，nonfinite count=0。
- valid labeled samples：mean=2.000000，min=2，max=2。
- mean BGS=0.032195；positive fraction=0.651492。
- w.mean=0.232634；w.max=0.999996；relative change=0.036062。

| phase | w.mean | w.mean change | w.max | relative change | skip rate |
|---|---|---|---|---|---|
| 00000-04999 | 0.196975 | -0.307024 | 0.999996 | 0.033101 | 0.000000 |
| 05000-09999 | 0.214707 | 0.008256 | 0.999996 | 0.034581 | 0.000000 |
| 10000-14999 | 0.232885 | -0.033234 | 0.999995 | 0.036131 | 0.000000 |
| 15000-19999 | 0.243355 | 0.008981 | 0.999995 | 0.036903 | 0.000000 |
| 20000-24999 | 0.249568 | 0.032680 | 0.999995 | 0.037434 | 0.000000 |
| 25000-29999 | 0.258311 | 0.008301 | 0.999995 | 0.038220 | 0.000000 |

每个阶段固定为 5,000 iterations；完整分布、首尾变化和线性斜率见 `bgs_training_statistics.csv`。

### Decoder BatchNorm coupling

| decoder mode | labeled mean abs diff | labeled max abs diff | labeled relative L2 | unlabeled mean abs diff |
|---|---|---|---|---|
| train | 0.0392019264 | 0.5521154404 | 0.0033925680 | 0.0642011836 |
| eval | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0969392806 |

固定 batch：`06SR5RBREL16DQ6M8LWS, 0RZDK210BSMWAA6467LU, 3C2QTUNI0852XV7ZH4Q1, 3DA0T2V6JJ2NLUAV6FWM`。BGS valid samples=2，w.mean=0.259290，x3 relative change=0.036877。
train mode 下两分支最大的 BN running-mean 差异 L2=0.0096287681，running-var 差异 L2=0.0108500752。逐层数据见 `batchnorm_coupling.csv`。

## 尚未验证的原因假设

- 固定 batch 上已直接观察到 train mode 的 labeled logits 差异，而 eval mode 差异为 0，验证了混合 batch statistics 的耦合存在；但本诊断没有做反事实重训练，因此不能断言它造成了最终 Dice 下降。
- 异常病例的体积偏差、FP/FN 和连通域变化描述了失败形态；它们不能单独区分训练期优化偏移、病例解剖差异或阈值/后处理敏感性。
- BGS 权重随训练阶段的漂移是相关性证据。没有不同 alpha、冻结 BN 或独立 seed 对照，不能把该漂移解释为因果机制。
- 本实验 validation 与最终 test 使用同一 `test.list`，因此不能据此声称独立测试集泛化机制。

## 产物

- `paired_case_analysis.csv`：20 例完整 paired delta 与逐例 leave-one-out。
- `bgs_training_statistics.csv`：每 5,000 iterations 及 overall 的 BGS/weight 统计。
- `batchnorm_coupling.csv`：train/eval decoder logits 差异与逐层 BN buffers 变化。
- `failure_visualizations/`：异常病例 masks、FP/FN、3D components 和关键切片。
