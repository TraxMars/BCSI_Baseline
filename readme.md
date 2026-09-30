# BCSI-native Mean-Teacher Baseline

This independent derivative preserves the supplied release's VNet encoder and
decoder, data splits, crop sizes, two-stream sampler, optimizer, scheduler,
validation, inference, and six experiment commands. The original
`BCSI-main/BCSI-main` directory is unmodified.

The paper defines its no-SSP/no-BCI/no-CR ablation as a Mean-Teacher (MT)
baseline. The release contains the EMA helper but no separately runnable MT
implementation. `trainer.py` reconnects that existing helper to the released
VNet and existing Dice/CE loss objects, without changing the backbone or data
partitions.

## Removed BCSI contributions

| Paper component | Equations | Released implementation removed |
| --- | --- | --- |
| Semantic-Spatial Perturbation (SSP) | (1)-(3), (12)-(14) | strong brightness/noise augmentation; copy-paste masks and mixed predictions; weak-to-strong pseudo-label, consistency, and entropy-weighted losses |
| Channel-selective Router (CR) | (4)-(6) | `mask_learner`, Top-K channel selection, scores/masks |
| Bidirectional Channel-wise Interaction (BCI) | (7)-(9) | labeled/unlabeled FIFO containers, similarity search, cross-attention, re-insertion, and queue updates |

The remaining SSL method is the paper-designated MT baseline: student VNet
supervised by labels plus EMA-teacher probability consistency on the unlabelled
portion of the original two-stream batch. Validation and saved checkpoints use
the EMA teacher.

## Run the six experiments

```bash
conda run -n SSL bash scripts/reproduce_la_10pct.sh
conda run -n SSL bash scripts/reproduce_la_20pct.sh
conda run -n SSL bash scripts/reproduce_pancreas_10pct.sh
conda run -n SSL bash scripts/reproduce_pancreas_20pct.sh
conda run -n SSL bash scripts/reproduce_brats_10pct.sh
conda run -n SSL bash scripts/reproduce_brats_20pct.sh
```

Each keeps the release's 30,000 iterations, data path, labelled percentage,
batch size, patches, optimizer, and validation interval. Checkpoints and logs
are written to this directory's `Results/` folder.
