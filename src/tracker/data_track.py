import configparser
import csv
import os
import os.path as osp
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

_sets = {}

# Fill all available datasets, change here to modify / add new datasets.
for split in [
    "train",
    "test",
    "all",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
]:
    name = f"MOT16-{split}"
    _sets[name] = lambda root_dir, *args, split=split: MOT16(root_dir, split, *args)


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith("."):
            yield f


class MOT16Sequences:
    """A central class to manage the individual dataset loaders.

    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed.
    """

    def __init__(self, dataset, root_dir, *args):
        """Initialize the corresponding dataloader.

        Keyword arguments:
        dataset --  the name of the dataset
        args -- arguments used to call the dataset
        """
        assert dataset in _sets, "[!] Dataset not found: {}".format(dataset)

        if len(args) == 0:
            args = [{}]

        self._data = _sets[dataset](root_dir, *args)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class MOT16(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, root_dir, split, **args):
        """Initliazes all subset of the dataset.

        Keyword arguments:
        root_dir -- directory of the dataset
        split -- the split of the dataset to use
        args -- arguments used to call the dataset
        """
        train_sequences = list(listdir_nohidden(os.path.join(root_dir, "train")))
        test_sequences = list(listdir_nohidden(os.path.join(root_dir, "test")))

        if "train" == split:
            sequences = train_sequences
        elif "test" == split:
            sequences = test_sequences
        elif "all" == split:
            sequences = train_sequences + test_sequences
        elif f"MOT16-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT16-{split}"]
        else:
            raise NotImplementedError("MOT split not available.")

        self._data = []
        for s in sequences:
            self._data.append(MOT16Sequence(root_dir, seq_name=s, **args))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class MOT16Sequence(Dataset):
    """Multiple Object Tracking Dataset.

    This dataset is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    """

    def __init__(self, root_dir, seq_name, vis_threshold=0.0, load_seg=False):
        """
        Args:
            root_dir -- directory of the dataset
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._vis_threshold = vis_threshold
        self._load_seg = load_seg
        self._mot_dir = root_dir

        self._train_folders = os.listdir(os.path.join(self._mot_dir, "train"))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, "test"))

        self.transforms = ToTensor()

        assert (
            seq_name in self._train_folders or seq_name in self._test_folders
        ), "Image set does not exist: {}".format(seq_name)

        self.data, self.no_gt = self._sequence()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        data = self.data[idx]

        img = Image.open(data["im_path"]).convert("RGB")

        img = self.transforms(img)

        sample = {}
        sample["img"] = img
        sample["img_path"] = data["im_path"]
        sample["gt"] = data["gt"]
        sample["vis"] = data["vis"]

        cropped_imgs = {}
        for gt_id, box in sample["gt"].items():
            box_crop = box.astype(np.int).clip(0, None)
            crop = img[:, box_crop[1] : box_crop[3], box_crop[0] : box_crop[2]]
            cropped_imgs[gt_id] = crop
        sample["cropped_imgs"] = cropped_imgs

        # segmentation
        if data["seg_img"] is not None:
            seg_img = np.array(data["seg_img"])
            # filter only pedestrians
            class_img = seg_img // 1000
            seg_img[class_img != 2] = 0
            # get instance masks
            seg_img %= 1000
            sample["seg_img"] = seg_img

        return sample

    def _sequence(self):
        seq_name = self._seq_name
        if seq_name in self._train_folders:
            seq_path = osp.join(self._mot_dir, "train", seq_name)
        else:
            seq_path = osp.join(self._mot_dir, "test", seq_name)

        config_file = osp.join(seq_path, "seqinfo.ini")

        assert osp.exists(config_file), "Config file does not exist: {}".format(
            config_file
        )

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config["Sequence"]["seqLength"])
        img_dir = config["Sequence"]["imDir"]

        img_dir = osp.join(seq_path, img_dir)
        gt_file = osp.join(seq_path, "gt", "gt.txt")
        seg_dir = osp.join(seq_path, "seg_ins")

        data = []
        boxes = {}
        visibility = {}
        seg_imgs = {}

        for i in range(1, seqLength + 1):
            boxes[i] = {}
            visibility[i] = {}

        no_gt = False
        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=",")
                for row in reader:
                    # class person, certainity 1, visibility >= 0.25
                    if (
                        int(row[6]) == 1
                        and int(row[7]) == 1
                        and float(row[8]) >= self._vis_threshold
                    ):
                        # Make pixel indexes 0-based, should already be 0-based (or not)
                        x1 = int(float(row[2])) - 1
                        y1 = int(float(row[3])) - 1
                        # This -1 accounts for the width (width of 1 x1=x2)
                        x2 = x1 + int(float(row[4])) - 1
                        y2 = y1 + int(float(row[5])) - 1
                        bb = np.array([x1, y1, x2, y2], dtype=np.float32)
                        boxes[int(row[0])][int(row[1])] = bb
                        visibility[int(row[0])][int(row[1])] = float(row[8])
        else:
            no_gt = True

        if self._load_seg:
            if osp.exists(seg_dir):
                for seg_file in listdir_nohidden(seg_dir):
                    frame_id = int(seg_file.split(".")[0])
                    seg_img = Image.open(osp.join(seg_dir, seg_file))
                    seg_imgs[frame_id] = seg_img

        for i in range(1, seqLength + 1):
            img_path = osp.join(img_dir, f"{i:06d}.jpg")

            datum = {"gt": boxes[i], "im_path": img_path, "vis": visibility[i]}

            datum["seg_img"] = None
            if seg_imgs:
                datum["seg_img"] = seg_imgs[i]

            data.append(datum)

        return data, no_gt

    def __str__(self):
        return self._seq_name

    def write_results(self, all_tracks, output_dir):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Files to sumbit:
        ./MOT16-01.txt
        ./MOT16-02.txt
        ./MOT16-03.txt
        ./MOT16-04.txt
        ./MOT16-05.txt
        ./MOT16-06.txt
        ./MOT16-07.txt
        ./MOT16-08.txt
        ./MOT16-09.txt
        ./MOT16-10.txt
        ./MOT16-11.txt
        ./MOT16-12.txt
        ./MOT16-13.txt
        ./MOT16-14.txt
        """

        # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file = osp.join(output_dir, "MOT16-" + self._seq_name[6:8] + ".txt")

        print("Writing predictions to: {}".format(file))

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=",")
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow(
                        [
                            frame + 1,
                            i + 1,
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
