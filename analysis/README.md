# Boundary Representation Audit

该目录独立读取已训练 Mean Teacher + VNet checkpoint，仅调用原始
`model.encoder(image)` 的 x2/x3/x4。不调用训练或 segmentation prediction。
baseline 的代码和 checkpoint 均不写入。

LA 10% / seed 42 的实际训练日志使用 `112 112 80`，以下命令沿用该尺寸。
脚本默认尺寸为 `96 96 96`，所有特征和 band 尺寸均从实际 tensor 读取。

```bash
cd /home/duanxuelong/BCSI_Baseline_BGCE
conda run --no-capture-output -n SSL python analysis/boundary_representation_audit.py \
  --data_path /home/duanxuelong/Dataset/LA \
  --dataset LA --labeled_num 10 \
  --model_path Results/seed_42/result_LA_10l/fold_0/Model_iter_27000.pth \
  --patch_size 112 112 80 --device cuda:0 \
  --patches_per_case 3 --topk_ratios 0.125 0.25 \
  --min_band_voxels 8 --max_retries 50 --seed 42 \
  --output_dir analysis/results/LA_10pct_seed42
```

运行其他已有 checkpoint 时更改数据集、标注比例、路径和 output_dir。
`--labeled_num` 与 baseline 一致，表示标注比例，通过原始
`patients_to_slices()` 得到病例数；LA 10% 使用原训练列表前 8 个病例。
dataset 仅构造原始列表，不调用其 `__getitem__`，只打开 labeled 前缀的 HDF5。
没有 CUDA 时显式指定 `--device cpu`。输出目录必须为空或尚不存在，避免覆盖实验。

## 统计约定

- 使用 baseline 的 `RandomCrop` 和 `ToTensor`，不额外归一化 intensity。
  每个病例至多尝试 `max_retries` 个候选 crop；每层收集 3 个有效 identity
  patch 后停止收集该层，所有层满足目标或达到 retry 上限后结束病例。
  若部分尺度始终缺少 band，允许该尺度使用 1–2 个 patch，标记为 partial。
  零有效 patch 的病例在该层排除。各层病例数可能不同，不应直接忽略此差异。
- mask 在实际 feature resolution 上 nearest resize，使用 3D pool 构造
  inner/outer 一体素 band。严格沿用需求中的 padding=1 morphology。
  图像边缘处 pooling 忽略网格外位置，统计包括 patch 内有效截面。
  `min_band_voxels` 同时约束两条 band，默认 8；输入 foreground 至少 32。
- 以总体方差（`unbiased=False`）计算 signed standardized d，eps=1e-6。
  原始 patch d 先取 case mean，再 L2 normalization。Top-K 按 abs(case_d)，
  K 使用 Python round，限制为 [1,C]；相同值按 channel index 稳定排序，
  case Top-K 截断点有 tie 时记日志。cosine 保留正负号。
- `summary.csv` 的 std 是分布总体标准差，CI 是**均值**的 95% percentile
  bootstrap CI，不是分布的 95% 覆盖区间。跨病例按病例重采样 2000 次，
  使用重采样频数加权不同原始病例的 pair，排除 self pair；不把相关的
  case pairs 当独立样本。augmentation 也按病例 cluster 重采样，独立汇总。
- 每个 C/K 用 5000 对独立均匀随机 channel subsets 估计 Jaccard null；
  null 均值 CI 对这些独立随机抽样 bootstrap。每个真实 case pair 对其中
  一个 signed normalized d 的 channel 顺序打乱 200 次，记录全部 null
  cosine。shuffled cosine 均值 CI 使用每个 pair 的 Monte Carlo mean，
  再按病例 bootstrap。另报 real-minus-null 的病例 bootstrap CI；
  Monte Carlo null 的数值误差不计入该差值 CI。
- 对 case normalized d 做**中心化 PCA**，保存所有可用 component 的 explained
  variance，日志输出 PC1/1–2/1–4/1–8，最多 N-1 个有效 component。
  同时输出 uncentered SVD energy，保留共同方向；两者不能互换。
  中心化 PCA 的 PC1 高表示病例之间的变化集中，不能单独证明共同 BTV 稳定。
- augmentation 为 identity、axis 0 flip、axis 1 flip、axis 0/1 平面 90°旋转；
  image/label 使用同一操作，无 intensity noise。非立方 patch 旋转后两个
  维度互换，实际 feature shape 会另外记录。比较四个视图的所有 6 个 pair，
  只将 identity d 用于 cross-case 分析，不把视图当额外病例。
- 非有限 d 逐 channel 记录 layer/case/patch/view 后失败，禁止静默忽略。
  零向量 cosine 或常数 ranking Spearman 无定义，保存 NaN、记录日志，
  summary 报 `n_undefined`；这与 d 中出现 NaN/Inf 是不同情况。

## 产物与检查

必需的 7 个文件和 4 张 PNG 均写入 output_dir。额外保存
`pairwise_channel_spearman.csv`、`random_baselines.csv`、`patch_status.csv`、
`case_status.csv` 和 `audit_metadata.json`，以复查随机基线和跳过原因。
metadata 记录完整参数、labeled 路径、实际 feature shape、每层病例/patch
计数以及执行前后的 SHA256。原始 baseline 七个代码文件、checkpoint、
内存中 model 参数和 buffers 都必须保持一致。每次 encoder 调用检查
所有 modules 为 eval 且梯度关闭；没有 gradient/training/update 调用。

NPZ 可用 `np.load(path, allow_pickle=False)` 打开，每层包含：

- `{layer}_case_ids`、`case_d`、`case_abs_d`、`case_d_norm`、`case_patch_counts`。
- `{layer}_patch_case_ids`、`patch_ids`、`patch_d`、`patch_abs_d`。
- `{layer}_aug_case_ids`、`aug_patch_ids`、`aug_names`、`aug_d`。

有效病例过少时 pair CI 会为空/NaN；不自动给出稳定性硬阈值。研究结论
应同时看 real vs null、完整 ranking、signed cosine、augmentation 和样本数。
该诊断不会实现 feature exchange、prototype、attention、channel selection module
或新 loss。

数值验证：

```bash
conda run --no-capture-output -n SSL python analysis/test_boundary_representation_audit.py
```
