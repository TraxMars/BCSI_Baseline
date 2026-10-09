# Experiment A-Control

Only new training: LA 10%, seed 42, alpha 0.1, 30000 iterations,
112×112×80 patches, 2 labeled + 2 unlabeled. Existing seed42 Baseline (EMA 27000)
and Experiment A (EMA 29000) and their measured 20-case Raw/LCC results are reused.
Their checkpoints, logs and analysis outputs are protected by SHA256.

The mutually exclusive CLI is `--guidance_mode none|bgs|shuffled_bgs`, default
`none`. Legacy `--use_bgs_guidance` remains an alias for `bgs`; argparse rejects
using both options. `none` executes the literal `self.model(volume_batch)`;
`bgs` retains Experiment A's existing calculations.

`shuffled_bgs` computes the current batch's real, detached labeled BGS weights,
then indexes those weights by a new uniform channel permutation. A dedicated
CPU `torch.Generator`, seeded from the run's seed 42, handles every permutation.
It does not draw from global Torch CPU/CUDA, NumPy or Python RNGs. The weights are
not renormalized after permutation, and labeled x3 is unchanged. Teacher,
supervised CE+Dice, unlabeled probability MSE, optimizer, scheduler, EMA,
TwoStreamBatchSampler and inference sources are unchanged.

The sorted weight multiset is checked bit for bit every valid iteration.
Floating-point reductions can change their last bit when values are reordered:
direct FP64 mean/norm differences must be ≤1e-12. The mathematical mean and norm
are invariant. Full checkpoints preserve the independent permutation RNG along
with Student, Teacher, optimizer momentum, scheduler, iteration and main RNGs.
Initial, latest, best and final full checkpoints are retained. The initial
Student/Teacher tensor digests must match the original seed42 VNet initialization.

The primary evaluation and checkpoint selection are fixed to original **LCC Dice**.
Raw metrics are secondary and use the same best EMA checkpoint. Inference calls
original `test_single_case` with stride 18/18/4 and threshold >0.5; LCC uses the
original 26-connected `getLargestCC`. All four metrics call original
`calculate_metric_percase`. All 20 test.list cases are retained in all means.
Original LA validation and final evaluation share these 20 cases.

Before training, severe failures are defined as empty prediction, Dice <0.70,
or HD95 >40 voxel. These are descriptive thresholds. Failures remain in means;
`severe_failures.csv` reports them separately. Prior anomaly
`WSJB9P4JCXUVHBOYFVWL` is always included in `tracked_case.csv`, irrespective of
whether it crosses a threshold. Original distance units, one-way ASD and
empty-prediction distance 100 are retained.

```bash
conda run -n SSL python experiments/bgs_guidance_A_control/test_control.py
conda run -n SSL python experiments/bgs_guidance_A/test_bgs_guidance.py
conda run -n SSL python experiments/bgs_guidance_A/test_recovery.py
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n SSL python \
  experiments/bgs_guidance_A_control/run_control.py --device cuda:0
```

Output: `Results/experiment_A_control_shuffle_seed42/`. The runner refuses to
overwrite an existing directory. It writes the predefined protocol, reused
references and source snapshots before starting, tracks progress, and performs
Raw/LCC evaluation automatically after the complete training run. Main outputs:
`summary.csv`, `per_case_metrics.csv` (120 rows at completion), each mode's
per-case CSV, `severe_failures.csv`, `tracked_case.csv`, `control_checks.json`,
`validation_history.csv`, complete logs/checkpoints and `experiment_report.md`.

Tests cover original none/BGS numerical equivalence, alpha-zero shuffle,
Teacher forward/gradients and unchanged loss/update AST, exact weight multiset,
mean/norm, untouched labeled x3, identical initialization, RNG isolation, and
exact recovery of the next DataLoader epoch including sampler/augmentation.
The mixed-batch decoder BatchNorm behavior is preserved. Results compare
independently trained models, with no cross-model channel matching. A promising
single-seed result requires other-seed verification; this runner launches only
seed42 and does not implement A+.
