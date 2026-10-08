# Experiment A: labeled BGS guidance

Only LA 10%, seed 42, alpha 0.1, 30k iterations. Online labeled Student x3 BGS
produces a detached continuous positive channel signature. Only unlabeled x3 is
rescaled. The Teacher and all baseline losses/updates/evaluation are unchanged.

`--use_bgs_guidance` defaults to False, retaining the literal original Student
forward. `--bgs_alpha` defaults to 0.1. There is no layer option, new ramp-up or
hard channel selection. Fewer than 8 boundary or nonboundary voxels invalidates
that labeled sample; no valid samples returns the original x3 tensor.

Original train.py and trainer.py are archived in `baseline_source/`. All existing
checkpoints, logs, analysis outputs, model, data and inference sources are
protected by `protected_files_before.json`. Default baseline behavior is tested
against the archived original implementation through a complete training step.

```bash
conda run --no-capture-output -n SSL python experiments/bgs_guidance_A/test_bgs_guidance.py
conda run --no-capture-output -n SSL python experiments/bgs_guidance_A/run_experiment.py \
  --device cuda:0
```

Results go into `Results/experiment_A_bgs_alpha0p1_seed42/`. The runner refuses
an existing output directory. run_status.json tracks progress; after training it
evaluates the baseline and guidance best EMA checkpoints using unchanged
prediction.py, saving all four metrics, raw logs, validation history and report.

LA baseline validation and final test both use the same test.list. The final
metrics are therefore a reevaluation of that set rather than an independent test.
There is no resume from the trained baseline checkpoint: the full experiment uses
the original seed 42 initialization and hyperparameters.

Disabled flag and alpha=0 checks compare exact losses, Student/Teacher state,
optimizer LR and scheduler against the original. Tests also compare BGS against
Experiment 0.5, verify detach, unchanged labeled features, valid/invalid cases,
feature/logit shape, Teacher gradients and nonfinite detection. Training modules
do not import analysis code.

The original decoder uses BatchNorm over the mixed labeled/unlabeled batch.
Rescaling only unlabeled x3 can change the decoder's batch statistics and
indirectly affect labeled logits. Labeled x3 remains identical and R is detached
as required; the experiment retains this original BatchNorm behavior. Any
measured gain describes the complete prescribed training path and does not
isolate a contribution exclusively through probability consistency.

Recovery on 2026-10-08: the first attempt stopped after iteration 15287 and only
had best-EMA checkpoints (best at 2000). That attempt is preserved under
`Results/experiment_A_bgs_alpha0p1_seed42_interrupted_20261008_iter15287/`.
The complete experiment was restarted from the same seed 42 initialization,
with the same training configuration, in a detached tmux session on physical
GPU 3 (`CUDA_VISIBLE_DEVICES=3`, logical `cuda:0`). The usual Results path now
contains the new run. No model-only continuation is used for the comparison.

The independent `checkpointed_train.py` entrypoint executes unchanged train.py.
After original validation, it atomically saves `fold_0/recovery_latest.pt` with
Student, Teacher, optimizer momentum, scheduler, iteration, best Dice, main RNG
states and source hashes. Checkpoints must be at a complete DataLoader epoch
boundary; LA 10% has four batches per epoch and validation every 200 steps.
Its `--recovery_state` option restores a full snapshot and obtains the starting
iteration from that snapshot. All other train.py arguments are passed through.
This adds serialization only, with no extra forward or change to model updates.

```bash
conda run --no-capture-output -n SSL python experiments/bgs_guidance_A/test_recovery.py
```

The recovery tests reproduce the next epoch's case indices, worker augmentation
samples, Student/Teacher state, optimizer state, scheduler and RNG exactly on
CPU, including the original TwoStreamBatchSampler and one-worker DataLoader.
