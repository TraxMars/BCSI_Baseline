"""Run unchanged train.py with atomic, complete recovery snapshots.

Snapshots are taken after the original validation at an epoch boundary. A new
DataLoader worker is created each epoch by the baseline, so restoring the main
RNG states also reproduces the next worker seed and TwoStreamBatchSampler order.
No forward, loss, optimizer update, scheduler update, or EMA code is replaced.
"""
import argparse
import hashlib
import inspect
from pathlib import Path
import random
import runpy
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SOURCE_FILES = ("train.py", "trainer.py", "utils/boundary_guidance.py",
                "model/vnet.py", "prediction.py", "dataloader/dataset.py",
                "dataloader/TwoStreamBatchSampler.py", "utils/transforms.py", "utils/utils.py")


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def save_recovery_state(trainer, completed_iterations, path, epoch_length):
    if completed_iterations % epoch_length:
        raise ValueError("Recovery snapshot must be at a complete epoch boundary")
    device = next(trainer.model.parameters()).device
    state = dict(
        format_version=1, completed_iterations=completed_iterations,
        epoch_length=epoch_length, args=dict(vars(trainer.args)),
        source_sha256=source_hashes(),
        student=trainer.model.state_dict(), teacher=trainer.ema_model.state_dict(),
        optimizer=trainer.optimizer.state_dict(), scheduler=trainer.scheduler.state_dict(),
        best_performance=trainer.best_performance,
        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_recovery_state(path):
    # These full snapshots include NumPy RNG state and are created by this script.
    kwargs = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
    state = torch.load(path, map_location="cpu", **kwargs)
    if state.get("format_version") != 1:
        raise ValueError("A complete recovery snapshot is required, not EMA weights")
    if state["source_sha256"] != source_hashes():
        raise ValueError("Training source differs from the recovery snapshot")
    if state["completed_iterations"] % state["epoch_length"]:
        raise ValueError("Recovery snapshot does not represent a complete epoch")
    return state


def restore_recovery_state(trainer, state):
    mutable_args = {"device", "output_dir", "start_iteration", "resume_model_path"}
    for name, old_value in state["args"].items():
        if name not in mutable_args and getattr(trainer.args, name) != old_value:
            raise ValueError("Recovery configuration differs: " + name)
    trainer.model.load_state_dict(state["student"])
    trainer.ema_model.load_state_dict(state["teacher"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    trainer.best_performance = state["best_performance"]
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    device = next(trainer.model.parameters()).device
    if device.type == "cuda":
        if state["cuda_rng"] is None:
            raise ValueError("CUDA RNG state is missing")
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    elif state["cuda_rng"] is not None:
        raise ValueError("Moving a CUDA training run to CPU changes its numerical behavior")


def main():
    import logging
    from trainer import Trainer
    from utils.utils import patients_to_slices

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--recovery_state", type=Path)
    recovery_args, training_args = parser.parse_known_args()
    state = load_recovery_state(recovery_args.recovery_state) if recovery_args.recovery_state else None
    if "--resume_model_path" in training_args:
        raise ValueError("Use a full recovery state for this experiment")
    if state:
        if "--start_iteration" in training_args:
            raise ValueError("Iteration is restored from the full recovery state")
        training_args.extend(["--start_iteration", str(state["completed_iterations"])])

    original_init, original_test = Trainer.__init__, Trainer.test

    def recovering_init(self, args):
        original_init(self, args)
        if not args.use_bgs_guidance or args.n_fold != 1:
            raise ValueError("This recovery launcher is for the single Experiment A run")
        if state:
            restore_recovery_state(self, state)
            logging.info("Restored complete training state at iteration %d",
                         state["completed_iterations"])

    def checkpointed_test(self, snapshot_path, iter_num):
        original_test(self, snapshot_path, iter_num)
        epoch_length = patients_to_slices(self.args.dataset, self.args.labeled_num) // self.args.labeled_bs
        if iter_num % epoch_length == 0:
            save_recovery_state(self, iter_num, Path(snapshot_path) / "recovery_latest.pt", epoch_length)
            logging.info("Saved full recovery state at iteration %d", iter_num)

    Trainer.__init__, Trainer.test = recovering_init, checkpointed_test
    sys.argv = [str(ROOT / "train.py"), *training_args]
    runpy.run_path(str(ROOT / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
