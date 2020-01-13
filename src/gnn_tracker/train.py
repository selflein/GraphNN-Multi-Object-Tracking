import sys
import math
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from IPython.core import ultratb
from torch.utils.data import DataLoader
from test_tube import Experiment
from torchvision.models import vgg16_bn
from torch_geometric.data import DataLoader

from src.gnn_tracker.modules.graph_nn import Net
from src.gnn_tracker.data_utils.dataset import PreprocessedDataset


sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                     color_scheme='Linux', call_pdb=1)


# Set seeds for reproducibility
torch.random.manual_seed(145325)
np.random.seed(435346)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)

    return parser


class GraphNNMOTracker:

    def __init__(self, config, writer):
        self.writer = writer
        self.config = config
        self.device = torch.device('cuda' if config.cuda else 'cpu')

        self.net = Net().to(self.device)

        log_dir = Path(
            self.writer.get_data_path(self.writer.name, self.writer.version))
        self.model_save_dir = log_dir / 'checkpoints'
        self.model_save_dir.mkdir(exist_ok=True)

        if config.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.epoch = 0

    def train_dataloader(self):
        ds = PreprocessedDataset(Path(self.config.dataset_path))
        train = DataLoader(ds, batch_size=self.config.batch_size)
        return train

    def train(self):
        train_loader = self.train_dataloader()

        # setup optimizer
        opt = torch.optim.Adam(self.net.parameters(),
                               lr=self.config.base_lr,
                               weight_decay=1e-4)

        sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)

        for epoch in range(self.config.epochs):
            self.epoch += 1
            self.net.train()
            train_metrics = defaultdict(list)
            pbar = tqdm(train_loader)
            for i, data in enumerate(pbar):
                data = data.to(self.device)
                gt = data.y
                out = self.net(data)
                loss = torch.tensor(0)

                train_metrics['loss'].append(loss.item())
                pbar.set_description(f"Loss: {loss.item()}")
                opt.zero_grad()
                loss.backward()
                opt.step()

            metrics = {k: np.mean(v) for k, v in train_metrics.items()}
            self.writer.log(metrics, epoch)
            sched.step(self.epoch)
            if epoch % 10 == 1:
                self.save(self.model_save_dir / 'checkpoints_{}.pth'.format(epoch))

    def save(self, path: Path):
        torch.save(self.net.state_dict(), path)

    def load(self, path: Path):
        self.net.load_state_dict(torch.load(path))

    @staticmethod
    def plot_matches_nonmatches(img_1, img_2, match_1, match_2, nonmatch_2):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        img_1_np = (to_cpu(img_1[0]).transpose(1, 2, 0) * 255).astype(np.uint8)
        match_1_np = to_cpu(match_1[0][0])

        img_2_np = (to_cpu(img_2[0]).transpose(1, 2, 0) * 255).astype(np.uint8)
        match_2_np = to_cpu(match_2[0][0])
        nonmatch_2_np = to_cpu(nonmatch_2[0])[:, 0, :]

        ax[0].imshow(img_1_np)
        ax[0].plot(match_1_np[0], match_1_np[1], 'o', markersize=3)

        ax[1].imshow(img_2_np)
        ax[1].plot(match_2_np[0], match_2_np[1], 'o', markersize=3)
        ax[1].plot(nonmatch_2_np[:, 0], nonmatch_2_np[:, 1], 'o', markersize=3)
        return fig

    @staticmethod
    def visualize_correspondences(img_1, img_2, match_1, match_2):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        img_1_np = (to_cpu(img_1[0]).transpose(1, 2, 0) * 255).astype(np.uint8)
        match_1_np = to_cpu(match_1[0])

        img_2_np = (to_cpu(img_2[0]).transpose(1, 2, 0) * 255).astype(np.uint8)
        match_2_np = to_cpu(match_2[0])

        ax[0].imshow(img_1_np)
        ax[0].plot(match_1_np[:, 0], match_1_np[:, 1], 'o', markersize=3)

        ax[1].imshow(img_2_np)
        ax[1].plot(match_2_np[:, 0], match_2_np[:, 1], 'o', markersize=3)
        return fig


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


if __name__ == '__main__':
    args = get_parser().parse_args()

    output_dir = Path(args.log_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = Experiment(output_dir, name=args.name, autosave=True,
                        flush_secs=15)
    logger.argparse(args)

    model = GraphNNMOTracker(args, logger)
    model.train()
