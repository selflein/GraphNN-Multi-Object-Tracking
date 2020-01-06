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

from src.tracker.data_track import MOT16


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
    parser.add_argument('--batch_size', type=int, default=32)

    return parser


class GraphNNMOTracker:

    def __init__(self, config, writer):
        self.writer = writer
        self.config = config
        self.device = torch.device('cuda' if config.cuda else 'cpu')

        self.appearance_cnn = vgg16_bn(pretrained=True).to(self.device)

        log_dir = Path(
            self.writer.get_data_path(self.writer.name, self.writer.version))
        self.model_save_dir = log_dir / 'checkpoints'
        self.model_save_dir.mkdir(exist_ok=True)

        if config.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.epoch = 0

    def train_dataloader(self):
        train = MOT16('../tracker', 'train')
        train_loader = DataLoader(train,
                                  batch_size=self.config.batch_size,
                                  num_workers=self.config.num_workers)
        return train_loader

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
            for i, batch in enumerate(pbar):
                img_1, img_2, match_1, match_2 = batch
                img_1 = img_1.to(self.device)
                img_2 = img_2.to(self.device)
                match_1 = match_1.to(self.device)
                match_2 = match_2.to(self.device)
                b, c, h, w = img_1.size()

                out_1 = self.net(img_1)
                out_2 = self.net(img_2)

                if i % 100 == 1:
                    self.writer.add_images('Features', out_1[:, :3, :, :])

                num_neg_points = self.config.num_neg_points
                num_pos_points = self.config.num_pos_points

                if self.config.close_points:
                    # Non matches in second image -> in ring around the matches
                    match_2_repeat = match_2.repeat(1, num_neg_points, 1)
                    num_offsets = match_2_repeat.size(1)
                    # Ring has width 25 and 5 pixel distance to point
                    vector_length = torch.rand((b, num_offsets)) * 25 + 5
                    angle = torch.rand((b, num_offsets)) * (2 * math.pi)

                    x_offsets = vector_length * torch.cos(angle)
                    y_offsets = vector_length * torch.sin(angle)
                    nonmatch_x = match_2_repeat[:, :, 0] + x_offsets
                    nonmatch_y = match_2_repeat[:, :, 1] + y_offsets
                    nonmatch_2 = torch.stack([nonmatch_x, nonmatch_y], dim=-1)

                    # Ensure points are on the image
                    nonmatch_2 = torch.clamp(nonmatch_2, 0, h - 1).long()
                    nonmatch_2 = nonmatch_2.view(b, num_neg_points, num_pos_points, 2)

                else:
                    # Sample non-matching points randomly
                    nonmatch_2 = torch.randint(0, h, (b, num_neg_points, num_pos_points, 2)).long()

                if self.config.debug_output and (i % 10 == 1):
                    fig1 = self.plot_matches_nonmatches(img_1, img_2, match_1, match_2, nonmatch_2)
                    fig2 = self.visualize_correspondences(img_1, img_2, match_1, match_2)
                    self.writer.add_figure('One match with nonmatches', fig1)
                    self.writer.add_figure('Correspondences', fig2)

                loss = criterion(out_1, out_2, match_1, match_2, nonmatch_2)

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

    model = SiameseNet(args, logger)
    model.train()
