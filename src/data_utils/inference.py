import csv
import argparse
from pathlib import Path
from statistics import mean
from collections import defaultdict
from typing import Dict, Tuple, List

import torch
from tqdm import tqdm
from torch_geometric.data import Data

from src.gnn_tracker.modules.graph_nn import Net


def load_subseq(subseq):
    edge_feats = torch.load(subseq / "edge_features.pth").float()
    e = torch.load(subseq / "edges.pth").long()
    node_feats = torch.load(subseq / "node_features.pth")
    gt_edges = torch.load(subseq / "gt.pth")
    node_ts = torch.load(subseq / "node_timestamps.pth")
    node_boxes = torch.load(subseq / "node_boxes.pth")
    return node_ts, node_feats, node_boxes, edge_feats, gt_edges, e


def combine_subsequences(
    subsequences, net, device: str = "cuda", threshold: float = 0.4
):
    """Takes the overlapping (with one frame each!) subsequences of an sequence
    and combines them into one big consistent graph where overlapping edge
    classifications are averaged.
    """
    edge_scores = defaultdict(list)
    t_box_to_global_id = {}

    global_id = 0
    for frame, subseq in enumerate(tqdm(subsequences)):
        (node_timestamps, node_features, boxes, edge_features, gt, e) = load_subseq(
            subseq
        )

        # Case where no detections in frame
        if len(node_features) == 0 or len(edge_features) == 0:
            continue

        data = Data(
            x=node_features,
            edge_index=e.t(),
            edge_attr=edge_features,
            y=gt,
            node_timestamps=node_timestamps,
            batch=None,
        ).to(device)

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

            # Store edge predictions for each edge
            edge_scores[(global_i, global_j)].append(pred[edge_id].item())

    # Average predictions of each edge
    edge_scores = {
        edge: mean(scores)
        for edge, scores in edge_scores.items()
        if mean(scores) > threshold
    }

    global_id_to_t_box = {v: k for k, v in t_box_to_global_id.items()}
    return global_id_to_t_box, edge_scores


def get_tracks(edge_scores: Dict[Tuple[int, int], float]) -> List[List[int]]:
    """Implements the greedy rounding scheme described in Section B.1. in order
    to obtain a graph satisfying the flow constraints. 

    Args:
        edge_scores: Dictionary containing the edges (as tuples (i, j)) as keys
            and assigned probability as value.

    Returns:
        List of tracks consisting of node IDs.
    """
    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    for (i, j), score in edge_scores.items():
        out_edges[i].append((j, score))
        in_edges[j].append((i, score))

    greedy_edges = set()
    max_node_id = max(out_edges.keys() | in_edges.keys())
    edges_to_remove = set()
    for node_id in range(max_node_id + 1):
        # Only consider outgoing edges with maximum score satisfying condition 1
        if node_id in out_edges:
            out_edge = max(out_edges[node_id], key=lambda e: e[1])[0]
            greedy_edges.add((node_id, out_edge))

            for (rem, _) in out_edges[node_id]:
                if rem != out_edge and (node_id, rem) in greedy_edges:
                    edges_to_remove.add((node_id, rem))

        # Only consider incoming edges with maximum score satisfying condition 2
        if node_id in in_edges and len(in_edges[node_id]) > 1:
            in_edge = max(in_edges[node_id], key=lambda e: e[1])[0]
            greedy_edges.add((in_edge, node_id))

            for (rem, _) in in_edges[node_id]:
                if rem != in_edge and (rem, node_id) in greedy_edges:
                    edges_to_remove.add((rem, node_id))

    greedy_edges -= edges_to_remove
    edge_to_next = {}
    for (i, j) in greedy_edges:
        if i in greedy_edges:
            raise ValueError(
                "Greedy rounding did not realize a graph "
                "satisfying flow constraints."
            )
        edge_to_next[i] = j

    # Convert edge transitions to tracks of nodes
    visited_nodes = set()
    all_tracks = []
    for node_id in range(max_node_id + 1):
        if node_id in visited_nodes:
            continue

        visited_nodes.add(node_id)
        track = [node_id]
        cur_node_id = node_id
        while cur_node_id in edge_to_next:
            next_node_id = edge_to_next[cur_node_id]
            track.append(next_node_id)
            visited_nodes.add(next_node_id)
            cur_node_id = next_node_id
        all_tracks.append(track)
    return all_tracks


def write_tracks_to_csv(tracks: dict, out: Path):
    with out.open("w") as of:
        writer = csv.writer(of, delimiter=",")
        for track_id, track in tracks.items():
            for frame, bb in track.items():
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                writer.writerow(
                    [
                        frame + 1,
                        track_id + 1,
                        x1 + 1,
                        y1 + 1,
                        x2 - x1 + 1,
                        y2 - y1 + 1,
                        -1,
                        -1,
                        -1,
                        -1,
                    ]
                )


def get_track_dict(subseqs_dir: Path, net_weight_path: Path, device: str = "cuda"):
    subseqs = sorted(subseqs_dir.iterdir(), key=lambda f: int(f.stem.split("_")[1]))

    edge_classifier = Net().to(device).eval()
    edge_classifier.load_state_dict(
        torch.load(net_weight_path, map_location=torch.device(device))
    )

    id_to_t_box, edge_scores = combine_subsequences(subseqs, edge_classifier, device)
    tracks = get_tracks(edge_scores)

    track_dict = defaultdict(dict)
    for track_id, track in enumerate(tracks):
        for node_id in track:
            t_box = id_to_t_box[node_id]
            t = t_box[0]
            box = t_box[1:]
            track_dict[track_id][t] = box

    return track_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed_sequences",
        type=str,
        nargs="+",
        help="Path(s) to the preprocessed sequence (!) folder",
    )
    parser.add_argument("--net_weights", type=str, help="Path to the trained GraphNN")
    parser.add_argument(
        "--out",
        type=str,
        help="Path of the directory where to write output "
        "files of the tracks in the MOT16 format",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    for sequence_folder in args.preprocessed_sequences:
        seq_folder = Path(sequence_folder)
        all_tracks = get_track_dict(seq_folder, Path(args.net_weights), args.device)

        write_tracks_to_csv(all_tracks, Path(args.out) / f"{seq_folder.name}.txt")
