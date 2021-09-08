from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torch_geometric.data import Dataset, Data


class PreprocessedDataset(Dataset):

    def __init__(self, dataset_path: Path, load_imgs=False, sequences=None):
        super(PreprocessedDataset, self). __init__(str(dataset_path / 'geometric'))

        subseqs = []
        if sequences is None:
            sequences = dataset_path.iterdir()
        else:
            sequences = [dataset_path / s for s in sequences]
        for seq_folder in sequences:
            if seq_folder.is_dir():
                for subseq_folder in seq_folder.iterdir():
                    subseqs.append(subseq_folder)

        self.subsequences = subseqs
        self.load_imgs = load_imgs

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, item):
        subseq = self.subsequences[item]
        edge_features = torch.load(subseq / 'edge_features.pth')
        edges = torch.load(subseq / 'edges.pth').long()
        node_features = torch.load(subseq / 'node_features.pth')
        gt = torch.load(subseq / 'gt.pth')
        node_timestamps = torch.load(subseq / 'node_timestamps.pth')

        if self.load_imgs:
            imgs = sorted((subseq / 'imgs').iterdir(),
                          key=lambda f: int(f.stem))
            transform = ToTensor()
            img_list = []
            for img in imgs:
                img_list.append(transform(Image.open(img).convert('RGB')))

            img_tensor = torch.stack(img_list)
            data = Data(x=node_features,
                        edge_index=edges.t(),
                        edge_attr=edge_features,
                        y=gt,
                        node_timestamps=node_timestamps,
                        imgs=img_tensor)
        else:

            data = Data(x=node_features,
                        edge_index=edges.t(),
                        edge_attr=edge_features,
                        y=gt,
                        node_timestamps=node_timestamps)
        return data

    def len(self):
        return len(self.subsequences)



