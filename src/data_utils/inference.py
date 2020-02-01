import csv
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
from torch_geometric.data import Data

from src.gnn_tracker.modules.graph_nn import Net


def load_subseq(subseq):
    edge_feats = torch.load(subseq / 'edge_features.pth')
    e = torch.load(subseq / 'edges.pth').long()
    node_feats = torch.load(subseq / 'node_features.pth')
    gt_edges = torch.load(subseq / 'gt.pth')
    node_ts = torch.load(subseq / 'node_timestamps.pth')
    node_boxes = torch.load(subseq / 'node_boxes.pth')
    return node_ts, node_feats, node_boxes, edge_feats, gt_edges, e


def combine_subsequences(subsequences, net, device: str = 'cuda'):
    """Takes the overlapping (with one frame each!) subsequences of an sequence
    and combines them into one big consistent graph where overlapping edge
    classifications are averaged.
    """
    edge_scores = defaultdict(float)
    t_box_to_global_id = {}

    global_id = 0
    for frame, subseq in enumerate(tqdm(subsequences)):
        (node_timestamps, node_features,
         boxes, edge_features, gt, e) = load_subseq(subseq)

        # Case where no detections in frame
        if len(node_features) == 0 or len(edge_features) == 0:
            continue

        data = Data(x=node_features,
                    edge_index=e.t(),
                    edge_attr=edge_features,
                    y=gt,
                    node_timestamps=node_timestamps,
                    batch=None).to(device)

        with torch.no_grad():
            # Shape (E,) array with classification score for every edge
            pred = net(data, node_features.clone().to(device)).cpu().numpy()

        # Since subsequence uses local time reference starting at 0
        node_timestamps += frame

        for node_id in range(len(node_features)):
            t = node_timestamps[node_id].item()
            box = boxes[node_id].numpy().tolist()
            if (t, *box) not in t_box_to_global_id.keys():
                t_box_to_global_id[(t, *box)] = global_id
                global_id += 1

        for edge_id, (i_edge, j_edge) in enumerate(e):
            t_i = node_timestamps[i_edge].item()
            box_i = boxes[i_edge].numpy().tolist()
            global_i = t_box_to_global_id[(t_i, *box_i)]

            t_j = node_timestamps[j_edge].item()
            box_j = boxes[j_edge].numpy().tolist()
            global_j = t_box_to_global_id[(t_j, *box_j)]

            # Combine predictions for same edge by averaging predictions
            edge_scores[(global_i, global_j)] = \
                (edge_scores[(global_i, global_j)] + pred[edge_id]) / 2.

    pred_edges = [edge for edge, score in edge_scores.items() if score > 0.4]
    global_id_to_t_box = {v: k for k, v in t_box_to_global_id.items()}
    return global_id_to_t_box, pred_edges


def get_tracks(e):
    edge_to_next = defaultdict(list)
    for i, j in e:
        edge_to_next[i].append(j)

    def traverse(neighbors, rem_nodes, track):
        for neighbor in neighbors:
            if neighbor in rem_nodes:
                track.append(neighbor)
                rem_nodes.remove(neighbor)
                traverse(edge_to_next[neighbor], rem_nodes, track)
                break
        return track

    all_tracks = []
    remaining_nodes = list(edge_to_next.keys())
    while remaining_nodes:
        all_tracks.append(traverse([remaining_nodes[0]], remaining_nodes, []))
    return all_tracks


def write_tracks_to_csv(tracks: dict, out: Path):
    with out.open('w') as of:
        writer = csv.writer(of, delimiter=',')
        for track_id, track in tracks.items():
            for frame, bb in track.items():
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                writer.writerow([frame + 1, track_id + 1, x1 + 1, y1 + 1,
                                 x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1])


def get_track_dict(subseqs_dir: Path, net_weight_path: Path,
                   device: str = 'cuda'):
    subseqs = sorted(subseqs_dir.iterdir(), key=lambda f: int(f.stem.split('_')[1]))

    edge_classifier = Net().to(device).eval()
    edge_classifier.load_state_dict(torch.load(net_weight_path))

    id_to_t_box, final_edges = combine_subsequences(subseqs,
                                                    edge_classifier,
                                                    device)
    tracks = get_tracks(final_edges)

    track_dict = defaultdict(dict)
    for track_id, track in enumerate(tracks):
        for node_id in track:
            t_box = id_to_t_box[node_id]
            t = t_box[0]
            box = t_box[1:]
            track_dict[track_id][t] = box

    return track_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_dir', type=str)
    parser.add_argument('--net_weights', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    all_tracks = get_track_dict(Path(args.preprocessed_dir),
                                Path(args.net_weights))

    write_tracks_to_csv(all_tracks, Path(args.out))
