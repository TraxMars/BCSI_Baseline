# LA 10% / seed 42：Boundary Representation Audit

本次结果支持**稳定的多 channel Boundary Transition Vector，同时存在部分稳定的固定 channel**。
这两个表现可以同时存在。在这组 labeled 病例中，没有观察到“两者都不稳定”的情况。
该结论是本 checkpoint、当前 GT boundary bands 和全局 d 定义下的研究推断。

使用现有 EMA teacher checkpoint `Model_iter_27000.pth`，只读取原始 LA 训练列表前
8 个 labeled 病例，未读取 unlabeled GT。沿用训练日志的 `112×112×80` patch。
实际 feature shape：x2=`[1,32,56,56,40]`，x3=`[1,64,28,28,20]`，
x4=`[1,128,14,14,10]`。每层 8 个有效病例、24 个 identity patch；
每个病例每层恰好 3 个有效 patch，均在 3 次 crop 内完成，没有跳过或 partial 病例。
四种视图得到每层 96 个 patch transition vectors。

下表为跨病例均值。随机 Jaccard 来自相同 C/K 的 5000 对随机 subset；
shuffled cosine 来自每个 case pair 的 200 次 channel permutation。

| 层 | Top-12.5% Jaccard：real / random | Top-25% Jaccard：real / random | abs(d) Spearman | signed BTV cosine：real / shuffled |
|---|---:|---:|---:|---:|
| x2 | 0.366 / 0.074 | 0.447 / 0.150 | 0.627 | 0.905 / 0.357 |
| x3 | 0.579 / 0.070 | 0.550 / 0.145 | 0.725 | 0.922 / 0.062 |
| x4 | 0.510 / 0.068 | 0.522 / 0.144 | 0.639 | 0.860 / 0.042 |

每层有 28 个不同病例 pair。跨病例 signed cosine 均值的病例 bootstrap 95% CI：
x2 `[0.883,0.931]`，x3 `[0.907,0.937]`，x4 `[0.828,0.898]`。
real-minus-shuffled 的均值差 CI 分别为 `[0.511,0.595]`、
`[0.836,0.883]`、`[0.784,0.853]`；各个 Top-K Jaccard 的
real-minus-random 均值差 CI 也全部为正。全部 mean/std/median/CI 保存在
[summary.csv](summary.csv)，CI 均针对均值，按病例重采样，不把 28 个 pair 当独立病例。
x2 的 shuffled cosine 为正且较高，应使用这个实际 null 比较，不能预设随机 cosine 为 0。

固定 channel 有稳定核心，但完整 Top-K 集合没有保持一致。以下是所有 8 个病例
共同进入 Top-K 的 channel index，使用从 0 开始的原始 channel 编号：

| 层 | Top-12.5%：共有 channel / K | Top-25%：共有 channel / K |
|---|---|---|
| x2 | `[15]` / 4 | `[7,15]` / 8 |
| x3 | `[12,24,35,49]` / 8 | `[12,24,35,42,46,49]` / 16 |
| x4 | `[2,6,33,37,82,85,86]` / 16 | `[2,6,21,33,37,58,72,76,82,85,86,118]` / 32 |

上述集合直接由 NPZ 的 abs(case_d) 排序得到，只用于解释稳定性。
本实验没有把这些 channel 接入模型或实现 channel selection。
较高的 signed cosine 与有限的 Top-K 重合共同说明：全局 transition 的整体方向
具有一致性，具体高排名集合仍随病例变化。x3 的 ranking 和 BTV 表现最稳定；
这些数值不构成排他性证据，不能据此断言边界只由某一个固定 channel 表示。

同一 patch 的 augmentation stability 单独统计四个同步几何视图的所有 6 个 pair，
每层共 144 个视图 pair。以下仍为均值，不与 cross-case 数据混合：

| 层 | Top-12.5% Jaccard | Top-25% Jaccard | abs(d) Spearman | signed cosine | cosine 均值 95% CI |
|---|---:|---:|---:|---:|---|
| x2 | 0.494 | 0.573 | 0.830 | 0.966 | `[0.954,0.978]` |
| x3 | 0.639 | 0.659 | 0.852 | 0.962 | `[0.956,0.967]` |
| x4 | 0.494 | 0.543 | 0.688 | 0.893 | `[0.884,0.901]` |

x4 对这些几何变换的稳定性低于 x2/x3，仍保持较高 signed cosine。
本次没有向 augmentation 加入 intensity noise。

中心化 PCA 的累计 explained variance 与保留共同方向的 uncentered SVD 分开报告：

| 层 | centered PC1 | centered PC1–2 | centered PC1–4 | centered PC1–8（实际 7） | uncentered 首个分量 energy |
|---|---:|---:|---:|---:|---:|
| x2 | 47.4% | 73.3% | 95.2% | 100.0% | 91.7% |
| x3 | 37.1% | 58.0% | 85.1% | 100.0% | 93.2% |
| x4 | 37.6% | 55.4% | 82.3% | 100.0% | 87.8% |

8 个病例使中心化 PCA 的最大有效 component 数为 7，末列累计 100% 是 rank
限制的结果。中心化 PCA 描述围绕均值方向的变化；其 PC1 不能直接作为共同
BTV 稳定性的证据。uncentered energy 和 signed cosine 共同支持较一致的方向。

本次只分析 8 个 labeled 病例和一个 checkpoint，未验证跨 seed、跨数据集或未见病例的
稳定性。GT bands 上的全局均值差也不能单独区分边界专属响应与前景/背景组织差异，
更不能判断空间局部表征是否稳定；因此本次诊断无需据此实现新的 segmentation module。

7 项数值测试通过，导出数据也经独立重算核验：case_d 等于原始 patch_d 均值；
先求均值再 normalization；全部 pair cosine、Top-K Jaccard 和 scipy Spearman 一致；
所有实际 band 均达到 8 voxel。所有 raw/case/augmentation d 均无 NaN/Inf，
没有 undefined pair。详见 [verification.json](verification.json)。
训练代码、VNet、数据/transform/utils/prediction 和 checkpoint 的 SHA256 与最初检查时一致，
内存中模型参数和 buffers 也保持一致。全程 eval/no_grad，无训练更新。

本次实际命令在 CPU 上完成：

```bash
conda run --no-capture-output -n SSL python analysis/boundary_representation_audit.py \
  --data_path /home/duanxuelong/Dataset/LA --dataset LA --labeled_num 10 \
  --model_path Results/seed_42/result_LA_10l/fold_0/Model_iter_27000.pth \
  --patch_size 112 112 80 --device cpu --patches_per_case 3 \
  --topk_ratios 0.125 0.25 --output_dir analysis/results/LA_10pct_seed42 --seed 42
```

复跑请指定新的空 output_dir。脚本及参数约定见 [analysis/README.md](../../README.md)。
全部要求的 CSV、NPZ、日志和四张 PNG 已保存于本目录。
