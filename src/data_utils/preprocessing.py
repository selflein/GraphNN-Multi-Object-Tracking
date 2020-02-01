import pickle
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from skimage.io import imsave
from skimage.transform import resize
from sklearn.decomposition import PCA
from torchreid.models.osnet import osnet_x0_5

from src.tracker.data_track import MOT16


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


def fit_pca(save_path: str, dataset_path: str, re_id_net):
    dataset = MOT16(dataset_path, 'train')

    instances = []
    for sequence in dataset:

        for i in tqdm(range(0, len(sequence), 50)):
            item = sequence[i]
            gt = item['gt']
            cropped = item['cropped_imgs']

            for gt_id, box in gt.items():
                with torch.no_grad():
                    try:
                        img = resize(cropped[gt_id].numpy().transpose(1, 2, 0),
                                     (256, 128))
                        feat = re_id_net(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(
                            0).cuda().float()).cpu().squeeze().numpy()
                        instances.append(feat)
                    except Exception as e:
                        tqdm.write('Error when processing image: {}'.format(str(e)))
                        continue
    print(f'Number of instances: {len(instances)}')
    pca_transform = PCA(n_components=32)
    pca_transform.fit(np.stack(instances))
    pickle.dump(pca_transform, open(save_path, 'wb'))


def preprocess(out_dir, re_id_net, mot_dataset, pca_transform, save_imgs=False):
    for sequence in mot_dataset:
        tqdm.write('Processing "{}"'.format(str(sequence)))
        seq_out = out_dir / str(sequence)
        seq_out.mkdir(exist_ok=True)

        for i in tqdm(range(len(sequence) - 15)):
            subseq_out = seq_out / 'subseq_{}'.format(i)

            try:
                subseq_out.mkdir()
            except FileExistsError:
                continue

            edges = []  # (2, num_edges) with pairs of connected node ids
            edge_features = []  # (num_edges, num_feat_edges) edge_id with features
            gt_edges = []  # (num_edges) with 0/1 depending on edge is gt

            existing_nodes = []
            node_id = 0

            for t, j in enumerate(range(i, i + 15)):
                item = sequence[j]
                gt = item['gt']
                cropped = item['cropped_imgs']

                cur_nodes = []
                for gt_id, box in gt.items():

                    with torch.no_grad():
                        try:
                            img = resize(cropped[gt_id].numpy().transpose(1, 2, 0),
                                         (256, 128))
                            feat = re_id_net(
                                torch.from_numpy(img).permute(2, 0, 1).unsqueeze(
                                    0).cuda().float()).cpu().numpy()
                            feat = pca_transform.transform(feat).squeeze()
                        except Exception as e:
                            tqdm.write(
                                'Error when processing image: {}'.format(str(e)))
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
            edge_features = torch.tensor(edge_features)
            node_features = torch.tensor([node['vis_feat'] for node in all_nodes])
            node_timestamps = torch.tensor([n['time'] for n in all_nodes])
            node_boxes = torch.tensor([n['box'] for n in all_nodes])

            torch.save(edges, subseq_out / 'edges.pth')
            torch.save(gt_edges, subseq_out / 'gt.pth')
            torch.save(node_timestamps, subseq_out / 'node_timestamps.pth')
            torch.save(edge_features, subseq_out / 'edge_features.pth')
            torch.save(node_features, subseq_out / 'node_features.pth')
            torch.save(node_boxes, subseq_out / 'node_boxes.pth')

            if save_imgs:
                imgs_out = subseq_out / 'imgs'
                imgs_out.mkdir()
                for n in all_nodes:
                    imsave(imgs_out / '{:5d}.png'.format(n['node_id']),
                           (n['img'] * 255.).astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./data/reprocessed')
    parser.add_argument('--pca_path', type=str, default='./data/pca.sklearn')
    parser.add_argument('--dataset_path', type=str, default='./data/MOT16')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--threshold', type=float, default=.1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=False)

    net = osnet_x0_5(pretrained=True).cuda()
    net.eval()

    ds = MOT16(args.dataset_path, args.mode, vis_threshold=args.threshold)
    pca = pickle.load(open(args.pca_path, 'rb'))
    preprocess(output_dir, net, ds, pca)
