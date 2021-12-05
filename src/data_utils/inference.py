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
    edge_to_nexts = defaultdict(list)
    for (i, j), scores in edge_scores.items():
        score = mean(scores)
        if score > threshold:
            edge_to_nexts[i].append((j, score))

    global_id_to_t_box = {v: k for k, v in t_box_to_global_id.items()}
    return global_id_to_t_box, edge_to_nexts, global_id


def get_tracks(edge_scores: Dict[int, List[Tuple[int, float]]], num_nodes: int) -> List[List[int]]:
    """Implements the greedy rounding scheme described in Section B.1. in order
    to obtain a graph satisfying the flow constraints. 

    Args:
        edge_scores: Dictionary containing the source node ID as keys
            and list of target node ID and probability as value.

    Returns:
        List of tracks consisting of node IDs.
    """
    visited_nodes = set()
    all_tracks = []

    # We can iterate based on the node ID since the IDs imply a temporal
    # sorting of the graph
    for node_id in range(num_nodes):
        if node_id in visited_nodes:
            continue

        visited_nodes.add(node_id)
        track = [node_id]
        cur_node_id = node_id
        while cur_node_id in edge_scores:
            remaining_edges = [n for n in edge_scores[cur_node_id] if n[0] not in visited_nodes]
            if len(remaining_edges) == 0:
                break

            # Get the next edge of maximum score
            next_node_id, _ = max(remaining_edges, key=lambda n: n[1])

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

    id_to_t_box, edge_scores, num_nodes = combine_subsequences(subseqs, edge_classifier, device)
    tracks = get_tracks(edge_scores, num_nodes)

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
