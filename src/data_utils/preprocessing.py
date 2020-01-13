from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize

from src.tracker.data_track import MOT16
from src.gnn_tracker.modules.re_id import ReID


def compute_box_features(box_1, box_2):
    top_1, left_1 = (box_1[0], box_1[3])
    top_2, left_2 = (box_2[0], box_2[3])

    width_1 = box_1[2] - box_1[0]
    width_2 = box_2[2] - box_2[0]

    height_1 = box_1[3] - box_1[1]
    height_2 = box_2[3] - box_2[1]

    y_rel_dist = 2 * (top_1 - top_2) / (height_1 + height_2)
    x_rel_dist = 2 * (left_1 - left_2) / (height_1 + height_2)
    rel_size_y = np.log(height_1 / height_2)
    rel_size_x = np.log(width_1 / width_2)
    return [x_rel_dist, y_rel_dist, rel_size_y, rel_size_x]


def get_top_k_nodes(cur_node, existing_nodes, k=50):
    cur_node_feat = cur_node['vis_feat']
    scores = []
    for ex in existing_nodes:
        scores.append(np.dot(cur_node_feat, ex['vis_feat']) / (np.linalg.norm(cur_node_feat) * np.linalg.norm(ex['vis_feat'])))

    sorted_nodes = [node for (score, node) in
                    sorted(zip(scores, existing_nodes), reverse=True,
                           key=lambda x: x[0])]

    try:
        return sorted_nodes[:k]
    except IndexError:
        return sorted_nodes


if __name__ == '__main__':
    out_dir = Path('./data/preprocessed/')
    out_dir.mkdir(exist_ok=True)

    net = ReID().to('cuda')
    net.eval()

    dataset = MOT16('./data/MOT16', 'train')

    for sequence in dataset:
        tqdm.write('Processing "{}"'.format(str(sequence)))
        seq_out = out_dir / str(sequence)
        seq_out.mkdir(exist_ok=True)

        for i in tqdm(range(len(sequence) - 16)):
            subseq_out = seq_out / 'subseq_{}'.format(i)

            try:
                subseq_out.mkdir()
            except FileExistsError:
                continue

            edges = []  # (2, num_edges) with pairs of connected node ids
            edge_features = []  # (num_edges, num_feat_edges) edge_id with features
            gt_edges = []  # (num_edges) with 0/1 depending on edge is gt
            node_timestamps = []  # (num_nodes,)

            existing_nodes = []
            node_id = 0

            for t, j in enumerate(range(i, i + 15)):
                item = sequence[j]
                gt = item['gt']
                cropped = item['cropped_imgs']

                cur_nodes = []
                for gt_id, box in gt.items():
                    node_timestamps.append(t)

                    with torch.no_grad():
                        try:
                            img = resize(cropped[gt_id].numpy().transpose(1, 2, 0),
                                         (256, 128))
                            feat = net(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(
                                0).cuda().float()).cpu().squeeze().numpy()
                        except Exception as e:
                            tqdm.write('Error when processing image: {}'.format(str(e)))
                            continue

                    cur_nodes.append({'box': box,
                                      'gt_id': gt_id,
                                      'img': img,
                                      'node_id': node_id,
                                      'time': t,
                                      'vis_feat': feat})

                    node_id += 1

                for cur in cur_nodes:
                    best_nodes = get_top_k_nodes(cur, existing_nodes)
                    for ex in best_nodes:
                        if True:
                            ex_id, cur_id = ex['node_id'], cur['node_id']
                            edges.append([ex_id, cur_id])

                            gt_edges.append(0 if ex['gt_id'] != cur['gt_id'] else 1)

                            box_feats = compute_box_features(ex['box'], cur['box'])
                            rel_appearance = np.linalg.norm(
                                cur['vis_feat'] - ex['vis_feat'], ord=2)
                            box_feats.append(cur['time'] - ex['time'])
                            box_feats.append(rel_appearance)
                            edge_features.append(box_feats)

                existing_nodes.extend(cur_nodes)

            all_nodes = sorted(existing_nodes, key=lambda n: n['node_id'])

            edges = torch.tensor(edges)
            gt_edges = torch.tensor(gt_edges)
            node_timestamps = torch.tensor(node_timestamps)
            edge_features = torch.tensor(edge_features)
            node_features = torch.tensor([node['vis_feat'] for node in all_nodes])

            torch.save(edges, subseq_out / 'edges.pth')
            torch.save(gt_edges, subseq_out / 'gt.pth')
            torch.save(node_timestamps, subseq_out / 'node_timestamps.pth')
            torch.save(edge_features, subseq_out / 'edge_features.pth')
            torch.save(node_features, subseq_out / 'node_features.pth')

            imgs_out = subseq_out / 'imgs'
            imgs_out.mkdir()
            for n in all_nodes:
                plt.imsave(imgs_out / '{}.png'.format(n['node_id']), n['img'])