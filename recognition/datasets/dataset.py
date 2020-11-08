import json
import os

import cv2
import lmdb
import numpy as np
from megengine.data.dataset.meta_dataset import MapDataset

DATASET_BASE_DIR = "/home/megstudio/workspace/dataset"


class TrainSet(MapDataset):
    """base class for train set
    """

    def __init__(self, lmdb_path, filename_list_path):
        """initialization

        Args:
            lmdb_path (str): path to lmdb dir
            filename_list_path (str): path to filename list, i.e. the keys in lmdb
        """
        super().__init__()

        with open(filename_list_path, "r") as f:
            info = json.load(f)
        self.indice = info["index"]
        self.labels = info["label"]

        # lmdb related
        self.env = lmdb.open(lmdb_path, lock=False)
        self.txn = self.env.begin()

    def __getitem__(self, index):
        content = self.txn.get(self.indice[index].encode())
        image = cv2.imdecode(np.frombuffer(content, "uint8"), cv2.IMREAD_COLOR)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return self.env.stat()["entries"]


class EvaluateSet(MapDataset):
    """base class for evaluate set
    """

    def __init__(self, lmdb_path, filename_list_path, noise_list_path):
        """initialization

        Args:
            lmdb_path (str): path to lmdb dir
            filename_list_path (str): path to filename list, i.e. the keys in lmdb
            noise_list_path (str): path to noise list
        """
        super().__init__()
        with open(filename_list_path, "r") as f:
            info = json.load(f)
        self.filename_list = info["path"]

        if "id" in info:
            self.id2label = {_id: index for index, _id in enumerate(set(info["id"]))}
            self.labels = [self.id2label[_id] for _id in info["id"]]
            self.has_label = True
        else:
            self.id2label = None
            self.labels = None
            self.has_label = False

        with open(noise_list_path, "r") as f:
            noise_list = [line.strip() for line in f.readlines()]
        self.is_noise = []
        for filename in self.filename_list:
            is_noise = filename in noise_list
            self.is_noise.append(is_noise)
        assert sum(self.is_noise) > 0, "noise expected, there must be something wrong!"

        # lmdb related
        self.env = lmdb.open(lmdb_path, lock=False)
        self.txn = self.env.begin()

    def __getitem__(self, index):
        filename = self.filename_list[index]
        content = self.txn.get(filename.encode())
        image = cv2.imdecode(np.frombuffer(content, "uint8"), cv2.IMREAD_COLOR)
        if self.has_label:
            label = self.labels[index]
        else:
            label = -1
        is_noise = self.is_noise[index]
        return image, index, label, is_noise

    def __len__(self):
        return len(self.filename_list)


class WebFace(TrainSet):
    """webface (10k ids / 0.5m images), first proposed in
    `"Learning Face Representation from Scratch" <https://arxiv.org/pdf/1411.7923.pdf>`_
    preprocessed and provided by
    `"ArcFace: Additive Angular Margin Loss for Deep Face Recognition" <https://arxiv.org/pdf/1801.07698.pdf>`_
    """

    def __init__(self, lmdb_path):
        filename_list_path = os.path.join(lmdb_path, "webface_feature_list.json")
        super().__init__(lmdb_path, filename_list_path)


class MS1MV2(TrainSet):
    """ms1mv2 (85k ids / 5.8m images), proposed in
    `"ArcFace: Additive Angular Margin Loss for Deep Face Recognition" <https://arxiv.org/pdf/1801.07698.pdf>`_
    """

    def __init__(self, lmdb_path):
        filename_list_path = os.path.join(lmdb_path, "ms1mv2_feature_list.json")
        super().__init__(lmdb_path, filename_list_path)


class Dummy(MapDataset):
    """dummy dataset for debugging
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __getitem__(self, index):
        image = np.zeros((112, 112, 3), dtype="float32")
        label = 0
        return image, label

    def __len__(self):
        return 2 ** 14


class FaceScrub(EvaluateSet):
    """facescrub (80 ids / 3.5k images), part of megaface test protocol, proposed in
    `"The MegaFace Benchmark: 1 Million Faces for Recognition at Scale" <https://arxiv.org/pdf/1512.00596.pdf>`_
    preprocessed, cleaned and provided by
    `"ArcFace: Additive Angular Margin Loss for Deep Face Recognition" <https://arxiv.org/pdf/1801.07698.pdf>`_
    """

    def __init__(self, lmdb_path):
        filename_list_path = os.path.join(lmdb_path, "facescrub_features_list.json")
        noise_list_path = os.path.join(lmdb_path, "facescrub_noises.txt")
        super().__init__(lmdb_path, filename_list_path, noise_list_path)

    @property
    def num_class(self):
        return len(self.id2label)


class MegaFace(EvaluateSet):
    """megaface (? ids / 1.0m images) proposed in
    `"The MegaFace Benchmark: 1 Million Faces for Recognition at Scale" <https://arxiv.org/pdf/1512.00596.pdf>`_
    preprocessed, cleaned and provided by
    `"ArcFace: Additive Angular Margin Loss for Deep Face Recognition" <https://arxiv.org/pdf/1801.07698.pdf>`_
    """

    def __init__(self, lmdb_path):
        filename_list_path = os.path.join(lmdb_path, "megaface_features_list.json_1000000_1")
        noise_list_path = os.path.join(lmdb_path, "megaface_noises.txt")
        super().__init__(lmdb_path, filename_list_path, noise_list_path)


def get_train_dataset(name, dataset_dir=DATASET_BASE_DIR):
    """get train dataset class by name

    Args:
        name (str): costum name of dataset
        dataset_dir (str, optional): directory of dataset root. defaults to DATASET_BASE_DIR

    Returns:
        M.Module: corresponding dataset class
    """
    mapping = {
        "webface": WebFace,
        "ms1mv2": MS1MV2,
        "dummy": Dummy,
    }
    assert name in mapping, f"dataset {name} is not found, choose one from {mapping.keys()}"
    return mapping[name](lmdb_path=os.path.join(dataset_dir, name))


def get_eval_dataset(name, dataset_dir=DATASET_BASE_DIR):
    """get eval dataset class by name

    Args:
        name (str): costum name of dataset
        dataset_dir (str, optional): directory of dataset root. defaults to DATASET_BASE_DIR

    Returns:
        M.Module: corresponding dataset class
    """
    mapping = {
        "facescrub": FaceScrub,
        "megaface": MegaFace,
    }
    assert name in mapping, f"dataset {name} is not found, choose one from {mapping.keys()}"
    return mapping[name](lmdb_path=os.path.join(dataset_dir, name))
