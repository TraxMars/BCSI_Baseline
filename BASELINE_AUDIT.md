# Paper-to-code audit

## Paper reading result

The supplied paper defines the baseline in Table 3 as **Mean Teacher (MT)**:
when SSP is absent, it says the weak-to-strong framework is replaced by MT.
The three BCSI innovations are SSP, CR, and BCI. The VNet backbone, ordinary
random crop/rotation/flip, two-stream labelled/unlabelled batch construction,
SGD, polynomial warm-restart scheduler, validation sliding window, split lists,
and metrics are framework infrastructure, not BCSI contributions.

| Innovation | Paper role and equations | Original location | Action in baseline |
| --- | --- | --- | --- |
| SSP | semantic colour/noise and spatial copy-paste perturbations; weak prediction supervises two strong predictions; Eq. 1-3, 10-14 | `dataloader/dataset.py`: strong view; `utils/transforms.py`: brightness/noise; `utils/mix_up.py`; `trainer.py`: mixing, sharpening pseudo-label, entropy weights, `weit_loss`, strong/mixed losses | Removed. Dataset returns a single ordinary view; `mix_up.py` removed. |
| CR | learn router scores and retain Top-K channels; Eq. 4-6 | `model/Semi_MoE.py`: `mask_learner`, `topk`; `forward` mask logic | Removed. |
| BCI | FIFO labelled/unlabelled containers, nearest channel retrieval, cross-attention and re-insertion; Eq. 7-9 | `model/Semi_MoE.py`: queues, `interpolation_save`, `enhance_selected_channels`, `ChannelCrossAttention`; `trainer.py`: queue calls | Removed. |

## Remaining execution path

`train.py` -> `dataloader/dataset.py` -> `TwoStreamBatchSampler` ->
`trainer.Trainer` -> `model.vnet.VNet` -> CE + Dice supervised loss and
EMA-teacher probability consistency -> optimizer/scheduler ->
`prediction.test_calculate_metric` -> checkpoint.

The source has no runnable author MT implementation: the only MT artefact is
the unused `update_ema_variables` function in the supplied `trainer.py`.
To implement the baseline the paper itself specifies, this directory retains
that function and applies it to a deepcopy of the supplied VNet. It does not
add a second decoder, an alternate backbone, data augmentation, confidence
filter, copy-paste, feature exchange, or BCSI-specific loss.

## Verification performed

* All six real dataset loaders construct the expected counts and an ordinary
  tensor sample: LA 80, Pancreas 62, BraTS-2019 250.
* The VNet forward pass returns `[2, 2, 32, 32, 32]`.
* A CPU student/teacher forward, loss, backward, optimizer, scheduler, and EMA
  update completes.
* Python compilation succeeds; repository search finds no live BCSI module,
  mask/router, queue, copy-paste, strong-view, pseudo-label, entropy-weighted
  loss, or Top-K call.

## Experiment status

The six 30k-iteration commands are present under `scripts/`, with the release
settings unchanged. This host reports `torch.cuda.is_available() == False` and
zero CUDA devices, so none was started: training uses the release's `cuda`
default and cannot run correctly here. Results must be produced on a CUDA host.
