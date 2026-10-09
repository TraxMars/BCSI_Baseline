"""Permutation invariants, reference behavior and RNG isolation for A-Control."""
import ast
import copy
import importlib.util
import logging
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainer import Trainer
from utils.boundary_guidance import apply_labeled_bgs_guidance, shuffle_bgs_weights, resolve_guidance_mode


def arguments(mode, alpha=0.1):
    return SimpleNamespace(in_channels=1, num_classes=2, device='cpu', base_lr=0.01,
                           max_iterations=30000, labeled_bs=2, ema_decay=0.9,
                           consistency=0.1, consistency_rampup=200, seed=42,
                           use_bgs_guidance=mode != 'none', guidance_mode=mode, bgs_alpha=alpha)


def labels():
    result = torch.zeros(4, 32, 32, 32, dtype=torch.long)
    result[:, 12:20, 12:20, 12:20] = 1
    result[2:] = 999
    return result


class ControlChecks(unittest.TestCase):
    def equal_tree(self, x, y):
        if isinstance(x, torch.Tensor):
            self.assertTrue(torch.equal(x, y))
        elif isinstance(x, dict):
            self.assertEqual(x.keys(), y.keys())
            for k in x:
                self.equal_tree(x[k], y[k])
        elif isinstance(x, (tuple, list)):
            for a, b in zip(x, y):
                self.equal_tree(a, b)
        else:
            self.assertEqual(x, y)

    def test_permutation_is_exact_and_does_not_consume_global_rng(self):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        w = torch.linspace(0, 1, 64).detach()
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        generator = torch.Generator(device='cpu').manual_seed(42)
        shuffled, permutation, diagnostics = shuffle_bgs_weights(w, generator)
        self.assertTrue(torch.equal(w.sort().values, shuffled.sort().values))
        self.assertTrue(torch.equal(shuffled, w[permutation]))
        self.assertFalse(torch.equal(w, shuffled))
        torch.testing.assert_close(w.mean(), shuffled.mean(), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(w.norm(), shuffled.norm(), rtol=1e-6, atol=1e-7)
        self.assertTrue(diagnostics['shuffle_distribution_equal'])
        self.assertFalse(shuffled.requires_grad)
        self.assertEqual(before[0], random.getstate())
        np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])
        self.assertEqual(before[1][2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))
        other = torch.Generator(device='cpu').manual_seed(42)
        self.assertTrue(torch.equal(shuffled, shuffle_bgs_weights(w, other)[0]))

    def test_guidance_keeps_labeled_feature_and_real_weight_multiset(self):
        x3 = torch.randn(4, 13, 8, 8, 8, requires_grad=True)
        plain, real_w, _ = apply_labeled_bgs_guidance(x3, labels()[:2], 2)
        shuffled, w, diagnostics = apply_labeled_bgs_guidance(
            x3, labels()[:2], 2, shuffle_generator=torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(shuffled[:2], x3[:2]))
        self.assertTrue(torch.equal(real_w.sort().values, w.sort().values))
        self.assertTrue(diagnostics['shuffle_distribution_equal'])
        self.assertEqual(shuffled[2:].shape, x3[2:].shape)
        self.assertTrue(torch.equal(shuffled[2:], x3[2:] + 0.1*w[None,:,None,None,None]*x3[2:]))
        self.assertFalse(w.requires_grad)
        self.assertTrue(torch.isfinite(shuffled).all())
        shuffled[2:].square().sum().backward()
        self.assertTrue(torch.equal(x3.grad[:2], torch.zeros_like(x3.grad[:2])))

    def test_all_modes_have_the_same_initialization_and_global_rng(self):
        states, rng_states = [], []
        for mode in ('none', 'bgs', 'shuffled_bgs'):
            torch.manual_seed(42)
            trainer = Trainer(arguments(mode))
            states.append(copy.deepcopy(trainer.model.state_dict()))
            rng_states.append(torch.get_rng_state())
            self.equal_tree(trainer.model.state_dict(), trainer.ema_model.state_dict())
        self.equal_tree(states[0], states[1]); self.equal_tree(states[0], states[2])
        self.assertTrue(all(torch.equal(rng_states[0], r) for r in rng_states))

    def test_bgs_matches_archived_experiment_A_and_none_matches_zero_shuffle(self):
        path = Path(__file__).parent/'reference_source/trainer.py'
        spec = importlib.util.spec_from_file_location('experiment_A_reference', path)
        reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
        torch.manual_seed(7)
        batch = dict(image=torch.randn(4, 1, 32, 32, 32), label=labels())
        results = []
        for cls, mode, alpha in [(reference.Trainer,'bgs',0.1), (Trainer,'bgs',0.1),
                                 (reference.Trainer,'none',0.1), (Trainer,'none',0.1),
                                 (Trainer,'shuffled_bgs',0.0), (Trainer,'shuffled_bgs',0.1)]:
            torch.manual_seed(42); trainer = cls(arguments(mode, alpha))
            recorded = []
            def log(fmt, *values):
                if fmt.startswith('iteration %d : loss'):
                    recorded.append(tuple(float(v) for v in values[1:4]))
            torch.manual_seed(82)
            with mock.patch.object(trainer.ema_model, 'forward', wraps=trainer.ema_model.forward) as teacher:
                with mock.patch('logging.info', side_effect=log):
                    trainer.train(batch, 1000, '/tmp')
                self.assertEqual(teacher.call_count, 1)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in trainer.ema_model.parameters()))
            results.append((recorded, copy.deepcopy(trainer.model.state_dict()),
                            copy.deepcopy(trainer.ema_model.state_dict()),
                            copy.deepcopy(trainer.optimizer.state_dict()),
                            copy.deepcopy(trainer.scheduler.state_dict()), torch.get_rng_state()))
        self.equal_tree(results[0], results[1])
        self.equal_tree(results[2], results[3]); self.equal_tree(results[2], results[4])
        self.assertTrue(torch.equal(results[1][-1],results[5][-1]))

    def test_none_uses_the_literal_original_forward(self):
        trainer = Trainer(arguments('none'))
        with mock.patch('trainer.apply_labeled_bgs_guidance', side_effect=AssertionError('Guidance used')):
            with mock.patch.object(trainer.model,'forward',wraps=trainer.model.forward) as forward:
                trainer.train(dict(image=torch.randn(4,1,32,32,32),label=labels()),1,'/tmp')
                self.assertEqual(forward.call_count, 1)
        with self.assertRaises(ValueError):
            resolve_guidance_mode(SimpleNamespace(guidance_mode='none',use_bgs_guidance=True))

    def test_teacher_loss_and_updates_have_identical_source_ast(self):
        old = ast.parse((Path(__file__).parent/'reference_source/trainer.py').read_text())
        new = ast.parse((ROOT/'trainer.py').read_text())
        def method(tree, name):
            return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef,ast.ClassDef)) and n.name==name)
        for name in ['PolyWarmRestartScheduler','update_ema_variables','sigmoid_rampup',
                     'get_current_consistency_weight','segmentation_loss','test']:
            self.assertEqual(ast.dump(method(old,name)),ast.dump(method(new,name)),name)
        def call_name(node):
            if isinstance(node,ast.Name): return node.id
            if isinstance(node,ast.Attribute): return call_name(node.value)+'.'+node.attr
            return ''
        def statements(tree):
            names={'supervised_loss','consistency_loss','consistency_weight','loss'}
            return [ast.dump(n) for n in method(tree,'train').body if
                    isinstance(n,ast.With) or
                    (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)) or
                    (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and call_name(n.value.func) in
                     ('self.optimizer.zero_grad','loss.backward','self.optimizer.step','update_ema_variables','self.scheduler.step'))]
        self.assertEqual(statements(old),statements(new))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
