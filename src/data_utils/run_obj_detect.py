import argparse

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.tracker.object_detector import FRCNN_FPN
from src.tracker.utils import obj_detect_transforms
from src.tracker.data_obj_detect import MOT16ObjDetect


def get_object_detector(model_path: str, device):
    obj_detect = FRCNN_FPN(num_classes=2, nms_thresh=0.3)
    obj_detect_state_dict = torch.load(model_path,
                                       map_location=lambda storage, loc: storage)
    obj_detect.load_state_dict(obj_detect_state_dict)
    obj_detect.eval()
    obj_detect.to(device)
    return obj_detect


def get_dataloader(split_path: str = 'data/MOT16/train'):
    dataset_test = MOT16ObjDetect(split_path,
                                  obj_detect_transforms(train=False))

    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False,
                                  num_workers=4, collate_fn=collate_fn)
    return data_loader_test, dataset_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run object detection on MOT16 sequences and generate '
                    'output files with detections for each sequence in the '
                    'same format as the `gt.txt` files of the training '
                    'sequences'
    )
    parser.add_argument('--model_path', type=str, help='Path to the FasterRCNN '
                                                       'model')
    parser.add_argument('--dataset_path', type=str, default='data/MOT16/train',
                        help='Path to the split of MOT16 to run detection on.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_path', type=str,
                        help='Output directory of the .txt files with '
                             'detections')
    args = parser.parse_args()

    net = get_object_detector(args.model_path, args.device)
    dl, ds = get_dataloader(args.dataset_path)

    result_dict = {}
    for i, (img, target) in enumerate(tqdm(dl)):
        img_id = target[0]['image_id'].item()

        with torch.no_grad():
            imgs = [im.to(args.device) for im in img]
            preds = net(imgs)
            result_dict[img_id] = preds[0]

    ds.write_results_files(result_dict, args.out_path)
