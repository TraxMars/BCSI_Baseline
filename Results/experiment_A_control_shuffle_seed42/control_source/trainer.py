import copy
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vnet import VNet
from prediction import test_calculate_metric
from utils.losses import DiceLoss
from utils.boundary_guidance import apply_labeled_bgs_guidance, resolve_guidance_mode


class PolyWarmRestartScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Unchanged scheduler from the released training code."""

    def __init__(self, optimizer, base_lr, max_iters, power=0.9,
                 warm_restart_iters=5000, last_epoch=-1):
        self.base_lr = base_lr
        self.max_iters = max_iters
        self.power = power
        self.warm_restart_iters = warm_restart_iters
        self.current_cycle_start = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch - self.current_cycle_start
        if t >= self.warm_restart_iters:
            self.current_cycle_start = self.last_epoch
            t = 0
        factor = (1 - t / self.warm_restart_iters) ** self.power
        return [self.base_lr * factor for _ in self.base_lrs]


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha, global_step):
    """The released EMA helper, retained for the paper's MT baseline."""
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


class Trainer(nn.Module):
    """Mean-Teacher baseline specified by the paper's no-SSP/BCI/CR ablation."""

    def __init__(self, args):
        super().__init__()
        self.best_performance = 0.0
        self.args = args
        self.model = VNet(n_channels=args.in_channels, n_classes=args.num_classes).to(args.device)
        self.ema_model = copy.deepcopy(self.model).to(args.device)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.00001)
        self.scheduler = PolyWarmRestartScheduler(
            self.optimizer, base_lr=args.base_lr, max_iters=args.max_iterations,
            power=0.9, warm_restart_iters=8000)
        self.dice_loss = DiceLoss(args.num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.guidance_mode = resolve_guidance_mode(args)
        self.bgs_shuffle_generator = None
        if self.guidance_mode == 'shuffled_bgs':
            self.bgs_shuffle_generator = torch.Generator(device='cpu')
            self.bgs_shuffle_generator.manual_seed(getattr(args, 'seed', 42))

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def get_current_consistency_weight(self, epoch):
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)

    def segmentation_loss(self, logits, labels):
        return 0.5 * (self.ce_loss(logits, labels) + self.dice_loss(logits, labels, softmax=True))

    def train(self, sampled_batch, iter_num, snapshot_path):
        volume_batch = sampled_batch['image'].to(self.args.device)
        label_batch = sampled_batch['label'].to(self.args.device)
        labeled_bs = self.args.labeled_bs
        if self.guidance_mode != 'none':
            features = self.model.encoder(volume_batch)
            try:
                features[2], w, diagnostics = apply_labeled_bgs_guidance(
                    features[2], label_batch[:labeled_bs], labeled_bs,
                    alpha=getattr(self.args, 'bgs_alpha', 0.1),
                    shuffle_generator=self.bgs_shuffle_generator)
            except FloatingPointError:
                logging.exception('BGS nonfinite measurement at iteration %d', iter_num)
                raise
            self.bgs_diagnostics = diagnostics
            self.bgs_weights = w
            logging.info(
                'BGS iteration %d : valid %d mean %f max %f positive_fraction %f '
                'w_mean %f w_max %f relative_change %f skip %s nonfinite %d',
                iter_num, diagnostics['valid_samples'], diagnostics['mean_bgs'],
                diagnostics['max_bgs'], diagnostics['positive_fraction'],
                diagnostics['w_mean'], diagnostics['w_max'],
                diagnostics['relative_change'], diagnostics['skipped'],
                diagnostics['nonfinite'])
            if self.guidance_mode == 'shuffled_bgs':
                logging.info('BGS shuffle iteration %d : distribution_equal %s mean_delta %.3e norm_delta %.3e skip %s',
                             iter_num, diagnostics.get('shuffle_distribution_equal', True),
                             diagnostics.get('shuffle_mean_delta', 0.0),
                             diagnostics.get('shuffle_norm_delta', 0.0), diagnostics['skipped'])
            if iter_num % 500 == 0:
                logging.info('BGS iteration %d : diagnostic Top-8 w channels %s', iter_num,
                             torch.topk(w, min(8, w.numel())).indices.cpu().tolist())
            student_logits = self.model.decoder(features)
            if not torch.isfinite(student_logits).all():
                logging.error('BGS nonfinite student logits at iteration %d', iter_num)
                raise FloatingPointError('Nonfinite student logits')
        else:
            student_logits = self.model(volume_batch)
        with torch.no_grad():
            teacher_logits = self.ema_model(volume_batch)
        supervised_loss = self.segmentation_loss(student_logits[:labeled_bs], label_batch[:labeled_bs])
        consistency_loss = F.mse_loss(
            torch.softmax(student_logits[labeled_bs:], dim=1),
            torch.softmax(teacher_logits[labeled_bs:], dim=1))
        consistency_weight = self.get_current_consistency_weight(iter_num // 150)
        loss = supervised_loss + consistency_weight * consistency_loss
        if self.guidance_mode != 'none':
            if not torch.isfinite(teacher_logits).all() or not torch.isfinite(loss):
                logging.error('BGS nonfinite teacher logits or loss at iteration %d', iter_num)
                raise FloatingPointError('Nonfinite teacher logits or loss')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        update_ema_variables(self.model, self.ema_model, self.args.ema_decay, iter_num)
        self.scheduler.step()
        logging.info('iteration %d : loss : %f supervised : %f consistency : %f lr : %f',
                     iter_num, loss, supervised_loss, consistency_loss,
                     self.optimizer.param_groups[0]['lr'])

    def test(self, snapshot_path, iter_num):
        self.ema_model.eval()
        dice, hd95 = test_calculate_metric(self.args, self.ema_model, val=True)
        if dice > self.best_performance:
            self.best_performance = dice
            torch.save(self.ema_model.state_dict(),
                       os.path.join(snapshot_path, 'Model_iter_' + str(iter_num) + '.pth'))
        logging.info('iteration %d : mean_dice : %f mean_hd95 : %f', iter_num, dice, hd95)
        self.model.train()
        self.ema_model.train()
