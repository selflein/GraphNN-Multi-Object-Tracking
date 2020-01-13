from pathlib import Path

import torch
from torch_geometric.data import Dataset, Data


class PreprocessedDataset(Dataset):

    def __init__(self, dataset_path: Path):
        super(PreprocessedDataset, self). __init__(str(dataset_path / 'geometric'))

        subseqs = []
        for seq_folder in dataset_path.iterdir():
            if seq_folder.is_dir():
                for subseq_folder in seq_folder.iterdir():
                    subseqs.append(subseq_folder)

        self.subsequences = subseqs

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

        data = Data(x=node_features,
                    edge_index=edges.t(),
                    edge_attr=edge_features,
                    y=gt,
                    node_timestamps=node_timestamps)

        return data

    def __len__(self):
        return len(self.subsequences)



