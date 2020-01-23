import sys
import math
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from IPython.core import ultratb
from torch.utils.data import DataLoader
from test_tube import Experiment
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as RegLoader, TensorDataset

from src.gnn_tracker.modules.re_id import ReID
from src.gnn_tracker.modules.graph_nn import Net
from src.gnn_tracker.modules.losses import FocalLoss
from src.gnn_tracker.data_utils.dataset import PreprocessedDataset


sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                     color_scheme='Linux', call_pdb=1)

import sys

# from src.data_utils.debug import trace_calls
# import os
# os.environ['GPU_DEBUG'] = "0"
# os.environ['TRACE_INTO'] = 'train'
# sys.settrace(trace_calls)

# Set seeds for reproducibility
torch.random.manual_seed(145325)
np.random.seed(435346)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--base_lr', type=float, default=0.00001)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train_cnn', action='store_true')
    parser.add_argument("--use_focal", action='store_true')

    return parser


class GraphNNMOTracker:

    def __init__(self, config, writer):
        self.writer = writer
        self.config = config
        self.device = torch.device('cuda' if config.cuda else 'cpu')

        self.net = Net().to(self.device)
        if self.config.train_cnn:
            self.re_id_net = ReID(out_feats=32, pretrained=True).to(self.device)

        log_dir = Path(
            self.writer.get_data_path(self.writer.name, self.writer.version))
        self.model_save_dir = log_dir / 'checkpoints'
        self.model_save_dir.mkdir(exist_ok=True)

        self.epoch = 0

    def train_dataloader(self):
        ds = PreprocessedDataset(Path(self.config.dataset_path),
                                 sequences=self.config.train_sequences)
        train = DataLoader(ds, batch_size=self.config.batch_size,
                           num_workers=self.config.workers)
        return train

    def val_dataloader(self):
        ds = PreprocessedDataset(Path(self.config.dataset_path),
                                 sequences=self.config.val_sequences)
        train = DataLoader(ds, batch_size=self.config.batch_size,
                           num_workers=self.config.workers)
        return train

    def train(self):
        train_loader = self.train_dataloader()
        val_loader = self.val_dataloader()

        # setup optimizer
        opt = torch.optim.Adam(self.net.parameters(),
                               lr=3e-4,
                               weight_decay=1e-4,
                               betas=(0.9, 0.999))

        if self.config.train_cnn:
            opt_re_id = torch.optim.Adam(self.re_id_net.parameters(),
                                         lr=3e-6,
                                         weight_decay=1e-4,
                                         betas=(0.9, 0.999))

        if self.config.use_focal:
            criterion = FocalLoss()
        else:
            criterion = torch.nn.BCELoss()

        for epoch in range(self.config.epochs):
            self.epoch += 1
            self.net.train()
            if self.config.train_cnn:
                self.re_id_net.train()
            train_metrics = defaultdict(list)
            pbar = tqdm(train_loader)
            for i, data in enumerate(pbar):
                data = data.to(self.device)
                gt = data.y.float()

                if self.config.train_cnn:
                    img_tensor = data.imgs
                    img_ds = TensorDataset(img_tensor)
                    img_dl = RegLoader(img_ds, batch_size=2)

                    x_feats = []
                    for imgs in img_dl:
                        if len(imgs) == 1:
                            self.re_id_net.eval()
                        x_feats.append(self.re_id_net(imgs[0].to(self.device)))
                        self.re_id_net.train()

                    x_feats = torch.cat(x_feats)
                    data.x = x_feats
                    del data.imgs

                initial_x = data.x.clone()
                out = self.net(data, initial_x).squeeze(1)
                loss = criterion(out, gt)

                with torch.no_grad():
                    acc = ((out > 0.5) == gt).float().mean().item()

                train_metrics['train/loss'].append(loss.item())
                train_metrics['train/acc'].append(acc)
                pbar.set_description(f"Loss: {loss.item():.4f}, Acc: {acc:.2f}")

                opt.zero_grad()
                loss.backward()
                opt.step()
                if self.config.train_cnn:
                    opt_re_id.step()

            with torch.no_grad():
                self.net.eval()
                if self.config.train_cnn:
                    self.re_id_net.eval()
                val_metrics = defaultdict(list)
                pbar = tqdm(val_loader)
                for i, data in enumerate(pbar):
                    data = data.to(self.device)
                    gt = data.y.float()

                    if self.config.train_cnn:
                        img_tensor = data.imgs
                        img_ds = TensorDataset(img_tensor)
                        img_dl = RegLoader(img_ds, batch_size=2)

                        x_feats = []
                        for imgs in img_dl:
                            x_feats.append(self.re_id_net(imgs[0].to(self.device)))

                        x_feats = torch.cat(x_feats)
                        data.x = x_feats
                        del data.imgs

                    initial_x = data.x.clone()
                    out = self.net(data, initial_x).squeeze(1)
                    loss = criterion(out, gt)

                    with torch.no_grad():
                        acc = ((out > 0.5) == gt).float().mean().item()

                    val_metrics['val/loss'].append(loss.item())
                    val_metrics['val/acc'].append(acc)
                    pbar.set_description(f"Validation epoch {self.epoch}: "
                                         f"Loss: {loss.item():.4f}, "
                                         f"Acc: {acc:.2f}")

            metrics = {k: np.mean(v)
                       for k, v in {**train_metrics, **val_metrics}.items()}
            self.writer.log(metrics, epoch)
            if epoch % 10 == 1:
                self.save(self.model_save_dir / 'checkpoints_{}.pth'.format(epoch))

    def save(self, path: Path):
        torch.save(self.net.state_dict(), path)

    def load(self, path: Path):
        self.net.load_state_dict(torch.load(path))


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


if __name__ == '__main__':
    sequences = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10',
                 'MOT16-11', 'MOT16-13']
    args = get_parser().parse_args()
    args.train_sequences = sequences[:6]
    args.val_sequences = sequences[6:]

    output_dir = Path(args.log_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = Experiment(output_dir, name=args.name, autosave=True,
                        flush_secs=15)
    logger.argparse(args)

    model = GraphNNMOTracker(args, logger)
    model.train()
