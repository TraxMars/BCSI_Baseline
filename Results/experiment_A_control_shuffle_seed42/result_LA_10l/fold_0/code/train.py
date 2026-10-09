import argparse
import numpy as np
import random
import torch
import os
import logging
import sys
from tqdm import tqdm
from dataloader.dataset import build_Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.utils import patients_to_slices
from utils.transforms import RandomRotFlip, RandomCrop, ToTensor

from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler


from trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/opt/data/private/data_3D',
                    help='Dataset root. For LA/Pancreas/BraTS2019 this is the dataset directory itself.')

# parser.add_argument('--dataset', type=str, default='/2018LA_Seg_Training Set',
#                     help='Name of Experiment')
# parser.add_argument('--dataset', type=str, default='/Pancreas',
#                     help='Name of Experiment')
parser.add_argument('--dataset', type=str, default='BraTS2019',
                    choices=['LA', 'Pancreas', 'BraTS2019', 'Lung',
                             '/2018LA_Seg_Training Set', '/Pancreas', '/BraTS2019', '/Lung'],
                    help='Dataset name. The short names support the local dataset layout.')
# parser.add_argument('--dataset', type=str, default='/Lung',
#                     help='Name of Experiment')


parser.add_argument('--labeled_num', type=int, default=10,
                    help='Percentage of label quantity')

parser.add_argument('--nms', type=int,  default=True,
                    help='output channel of network')

parser.add_argument('--in_channels', type=int,  default=1,
                    help='output channel of network')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

parser.add_argument('--patch_size', type=int, nargs=3, default=[96, 96, 96],
                    help='patch size of network input')

# parser.add_argument('--patch_size', type=list,  default=[112, 112, 80],
#                     help='patch size of network input')

parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')

parser.add_argument('--seed', type=int,  default=42,
                    help='random seed')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--max_iterations', type=int, default=30000,
                    help='maximum epoch number to train')

parser.add_argument('--n_fold', type=int, default=1, help='maximum epoch number to train')
parser.add_argument('--output_dir', type=str, default='./Results', help='Directory for logs and checkpoints')
parser.add_argument('--test_interval', type=int, default=200, help='Validate every N iterations (0 disables validation)')
parser.add_argument('--resume_model_path', type=str, default=None,
                    help='Model-only checkpoint to restore after an interrupted run')
parser.add_argument('--start_iteration', type=int, default=0,
                    help='Iteration represented by --resume_model_path')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--ema_decay', type=float, default=0.9, help='ema_decay')

guidance_group = parser.add_mutually_exclusive_group()
guidance_group.add_argument('--use_bgs_guidance', action='store_true', default=False,
                            help='Legacy alias for --guidance_mode bgs')
guidance_group.add_argument('--guidance_mode', choices=['none', 'bgs', 'shuffled_bgs'], default=None,
                            help='Mutually exclusive channel guidance mode (default: none)')
parser.add_argument('--bgs_alpha', type=float, default=0.1,
                    help='Fixed unlabeled x3 gain (no additional ramp-up)')

parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()
args.guidance_mode = args.guidance_mode or ('bgs' if args.use_bgs_guidance else 'none')
args.use_bgs_guidance = args.guidance_mode != 'none'


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def train(args, snapshot_path):
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    patch_size = args.patch_size

    # model
    trainer = Trainer(args)
    if args.resume_model_path:
        checkpoint = torch.load(args.resume_model_path, map_location=args.device)
        trainer.model.load_state_dict(checkpoint)
        trainer.ema_model.load_state_dict(checkpoint)
        # Existing checkpoints in this repository contain model weights only.
        # Align the LR scheduler with the saved iteration; optimizer momentum is
        # necessarily restarted because it was not stored in the checkpoint.
        trainer.scheduler.step(args.start_iteration)
        logging.info('restored model weights from %s at iteration %d',
                     args.resume_model_path, args.start_iteration)

    # dataset
    if args.dataset in ('LA', "/2018LA_Seg_Training Set"):
        data_dir = args.data_path if args.dataset == 'LA' else args.data_path + args.dataset
        train_dataset = build_Dataset(args, data_dir=data_dir, split="train_LA",
                                      transform=transforms.Compose([RandomRotFlip(), RandomCrop(patch_size),  ToTensor()]))
    elif args.dataset in ('Pancreas', "/Pancreas"):
        data_dir = args.data_path if args.dataset == 'Pancreas' else args.data_path + args.dataset
        train_dataset = build_Dataset(args, data_dir=data_dir, split="train_Pancreas",
                                      transform=transforms.Compose([RandomCrop(patch_size), ToTensor()]))
    elif args.dataset in ('BraTS2019', "/BraTS2019"):
        data_dir = args.data_path if args.dataset == 'BraTS2019' else args.data_path + args.dataset
        train_dataset = build_Dataset(args, data_dir=data_dir, split="train_BraTS2019",
                                      transform=transforms.Compose([RandomRotFlip(), RandomCrop(patch_size), ToTensor()]))
    elif args.dataset == "/Lung":
        train_dataset = build_Dataset(args, data_dir=args.data_path + args.dataset, split="train_Lung",
                                      transform=transforms.Compose([RandomRotFlip(), RandomCrop(patch_size), ToTensor()]))
    
    # sampler
    total_slices = len(train_dataset)
    labeled_slice = patients_to_slices(args.dataset, args.labeled_num)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)

    logging.info("{} iterations per epoch".format(len(train_loader)))
    # max_epoch = max_iterations // len(train_loader) + 1
    max_epoch = (max_iterations - args.start_iteration) // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    iter_num = args.start_iteration

    for _ in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            trainer.train(sampled_batch, iter_num, snapshot_path)
            iter_num = iter_num + 1
            if args.test_interval and iter_num > 0 and iter_num % args.test_interval == 0:
                trainer.test(snapshot_path, iter_num)
            if iter_num >= max_iterations:
                return


if __name__ == '__main__':
    import shutil
    for fold in range(args.n_fold):
        run_seed = args.seed + fold
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        dataset_name = args.dataset.strip('/').replace(' ', '_')
        snapshot_path = os.path.join(args.output_dir, "result_{}_{}l".format(dataset_name, args.labeled_num), "fold_" + str(fold))

        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if not os.path.exists(snapshot_path + '/code'):
            os.makedirs(snapshot_path + '/code')

        shutil.copyfile("./train.py", snapshot_path + "/code/train.py")
        shutil.copyfile("./trainer.py", snapshot_path + "/code/trainer.py")
        shutil.copyfile("./model/vnet.py", snapshot_path + "/code/vnet.py")
        if args.use_bgs_guidance:
            shutil.copyfile("./utils/boundary_guidance.py",
                            snapshot_path + "/code/boundary_guidance.py")

        logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

        train(args, snapshot_path)
