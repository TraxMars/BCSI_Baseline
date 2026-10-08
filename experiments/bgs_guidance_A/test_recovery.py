"""Check exact recovery across the original four-batch DataLoader epoch."""
import copy
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainer import Trainer
from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler
from experiments.bgs_guidance_A.checkpointed_train import (
    load_recovery_state, restore_recovery_state, save_recovery_state)


class RandomPatchDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        image = torch.randn(1, 32, 32, 32) + np.random.rand() + random.random()
        label = torch.zeros(32, 32, 32, dtype=torch.long)
        label[12:20, 12:20, 12:20] = 1
        if index >= 8:
            label.fill_(999)
        return dict(image=image, label=label, index=index)


def worker_init_fn(worker_id):
    random.seed(42 + worker_id)


def loader():
    sampler = TwoStreamBatchSampler(list(range(8)), list(range(8, 12)), 4, 2)
    return DataLoader(RandomPatchDataset(), batch_sampler=sampler, num_workers=1,
                      pin_memory=True, worker_init_fn=worker_init_fn)


def args(start=0):
    return SimpleNamespace(in_channels=1, num_classes=2, device="cpu", base_lr=0.01,
                           max_iterations=30000, labeled_bs=2, ema_decay=0.9,
                           consistency=0.1, consistency_rampup=200, seed=42,
                           use_bgs_guidance=True, bgs_alpha=0.1, start_iteration=start,
                           dataset="LA", labeled_num=10, patch_size=[32, 32, 32])


class RecoveryChecks(unittest.TestCase):
    def equal_tree(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.equal_tree(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.equal_tree(x, y)
        else:
            self.assertEqual(a, b)

    def test_epoch_boundary_resumes_sampler_augmentation_and_full_training_state(self):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        trainer = Trainer(args())
        train_loader = loader()
        iterator = iter(train_loader)
        for iteration in range(len(train_loader)):
            trainer.train(next(iterator), iteration, "/tmp")
        trainer.best_performance = 0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recovery.pt"
            rng_before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            save_recovery_state(trainer, 4, path, len(train_loader))
            self.equal_tree(rng_before, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
            with self.assertRaises(StopIteration):
                next(iterator)
            self.equal_tree(rng_before, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
            state = load_recovery_state(path)
            expected_batches = []
            for iteration, batch in enumerate(train_loader, 4):
                expected_batches.append((batch["index"].clone(), batch["image"].clone()))
                trainer.train(batch, iteration, directory)
            expected_model = copy.deepcopy(trainer.model.state_dict())
            expected_teacher = copy.deepcopy(trainer.ema_model.state_dict())
            expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
            expected_scheduler = copy.deepcopy(trainer.scheduler.state_dict())
            expected_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            recovered = Trainer(args(4))
            restore_recovery_state(recovered, state)
            for iteration, batch in enumerate(loader(), 4):
                self.equal_tree(expected_batches[iteration - 4], (batch["index"], batch["image"]))
                recovered.train(batch, iteration, directory)
            self.equal_tree(expected_model, recovered.model.state_dict())
            self.equal_tree(expected_teacher, recovered.ema_model.state_dict())
            self.equal_tree(expected_optimizer, recovered.optimizer.state_dict())
            self.equal_tree(expected_scheduler, recovered.scheduler.state_dict())
            self.equal_tree(expected_rng, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
            self.assertEqual(recovered.best_performance, 0.5)
            self.assertTrue(all(p.grad is None for p in recovered.ema_model.parameters()))

    def test_partial_epoch_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_recovery_state(Trainer(args()), 5, Path(directory) / "bad.pt", 4)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
