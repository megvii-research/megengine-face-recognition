"""do the evaluation work with single gpu
"""
import argparse
import os

import megengine as mge
import megengine.data as data
import megengine.data.transform as T
import megengine.functional as F
import numpy as np
from tqdm.auto import tqdm

from recognition.datasets import get_eval_dataset
from recognition.models import FaceRecognitionModel
from recognition.tools.utils import load_config_from_path

logger = mge.get_logger(__name__)


def get_inference_func(configs):
    """load checkpoint and construct inference function

    Args:
        configs (dict): configuration, required fields include:
            base_dir: base directory of experiment outputs
            evaluate_epoch: model of evaluate_epoch to evaluate

    Raises:
        FileNotFoundError: model of given epoch is not found

    Returns:
        inference_func (function): inference function mapping image to embedding
    """
    model = FaceRecognitionModel(configs)
    evaluate_epoch = configs["evaluate_epoch"]
    checkpoint_path = os.path.join(configs["base_dir"], f"epoch-{evaluate_epoch}-checkpoint.pkl")
    if os.path.exists(checkpoint_path):
        checkpoint_data = mge.load(checkpoint_path)
        model.load_state_dict(checkpoint_data["state_dict"], strict=False)
    else:
        raise FileNotFoundError(f"{checkpoint_path} not found!!!")

    def inference_func(images):
        model.eval()
        # classic test-time mirror augment
        embedding_origin = model.forward_embedding_only(images)
        embedding_mirror = model.forward_embedding_only(images[:, :, :, ::-1])
        embedding = embedding_origin + embedding_mirror
        embedding = F.normalize(embedding, axis=1)
        return embedding

    return inference_func


def extract_feature_and_clean_noise(configs, inference_func):
    """extract feature and clean noise. the noise cleaning algorithm is proposed in
    `"ArcFace: Additive Angular Margin Loss for Deep Face Recognition" <https://arxiv.org/pdf/1801.07698.pdf>`_
    please refer to https://github.com/deepinsight/insightface/blob/master/Evaluation/Megaface/remove_noises.py for
    more detail. this implement does basicly the same thing as the above, but with much higher speed

    Args:
        configs (dict): configuration, required fields include:
            batch_size: inference batch size
            feature_dim: model output feature dimension
            base_dir: base directory of experiment outputs
            dataset_dir: directory of dataset root
        inference_func (function): constructed inference function

    Returns:
        facescrub_feature (np.array): noise-cleaned feature of facescrub (shape: n * (feature_dim + 1))
        facescrub_label (np.array): label of facescrub (shape: n)
        megaface_feature (np.array): noise-cleaned feature of megaface (shape: m * (feature_dim + 1))
    """

    def prepare_dataset(name):
        """prepare dataset

        Args:
            name (str): name of the dataset, should be one of {facescrub, megaface}

        Returns:
            dataset (data.Dataset): required dataset
            queue (data.DataLoader): corresponding dataloader
        """
        preprocess = T.Compose([T.Normalize(mean=127.5, std=128), T.ToMode("CHW")])
        dataset = get_eval_dataset(name, dataset_dir=configs["dataset_dir"])
        sampler = data.SequentialSampler(dataset, batch_size=configs["batch_size"])
        queue = data.DataLoader(dataset, sampler=sampler, transform=preprocess)
        return dataset, queue

    def extract_vanilla_feature(n, data_queue):
        """extract features without any postprocessing

        Args:
            n (int): size of dataset
            data_queue (data.DataLoader): dataloader to extract feature

        Returns:
            feature_store (np.array): extracted feature (shape: n * feature_dim)
            label (np.array): label of this instance, -1 if unknown (shape: n)
            is_noise (np.array): whether this instance is a noise (shape: n)
        """
        feature_store = np.zeros((n, configs["feature_dim"]), dtype="float32")
        label_store = np.zeros(n, dtype="int32")
        is_noise_store = np.zeros(n, dtype="bool")
        for images, indice, labels, is_noise in tqdm(data_queue):
            images = mge.tensor(images, dtype="float32")
            embedding = inference_func(images)
            embedding = embedding.numpy()

            feature_store[indice] = embedding
            label_store[indice] = labels
            is_noise_store[indice] = is_noise
        return feature_store, label_store, is_noise_store

    # prepare facescrub dataset
    logger.info("preparing facescrub dataset...")
    facescrub_dataset, facescrub_queue = prepare_dataset("facescrub")

    # extract facescrub feature
    logger.info("extracting facescrub...")
    facescrub_feature_store, facescrub_label, facescrub_is_noise = extract_vanilla_feature(
        n=len(facescrub_dataset), data_queue=facescrub_queue
    )

    # prepare megaface dataset
    logger.info("preparing megaface dataset...")
    megaface_dataset, megaface_queue = prepare_dataset("megaface")

    # extract feature for megaface
    logger.info("extracting megaface...")
    megaface_feature_store, _, megaface_is_noise = extract_vanilla_feature(
        n=len(megaface_dataset), data_queue=megaface_queue
    )

    # parse facescrub noise, replace noisy feature with class center of same person
    facescrub_feature_center = np.zeros((facescrub_dataset.num_class, configs["feature_dim"]), dtype="float32")
    for i in range(facescrub_dataset.num_class):
        mask = (facescrub_label == i) & (~facescrub_is_noise)
        center = facescrub_feature_store[mask].sum(axis=0)
        center = center / np.linalg.norm(center)
        facescrub_feature_center[i] = center
    for index in np.where(facescrub_is_noise)[0]:
        center = facescrub_feature_center[facescrub_label[index]]
        disturb = np.random.uniform(-1e-5, 1e-5, (configs["feature_dim"],))
        feat = center + disturb  # avoid identical features with minor disturb
        feat = feat / np.linalg.norm(feat)
        facescrub_feature_store[index] = feat

    # extend feature by 1 dimension
    # the extended feature is infinitly large (100) if and only if megaface noise, 0 otherwise
    # so, the distance between probe and a noisy distractor is infinitly large, while other distances remain unchanged
    facescrub_feature_extend = np.zeros((len(facescrub_dataset), 1), dtype="float32")
    facescrub_feature = np.concatenate([facescrub_feature_store, facescrub_feature_extend], axis=1)
    megaface_feature_extend = megaface_is_noise.astype("float32").reshape(-1, 1) * 100
    megaface_feature = np.concatenate([megaface_feature_store, megaface_feature_extend], axis=1)

    # write to file system
    facescrub_feature_path = os.path.join(configs["base_dir"], "facescrub.npy")
    np.save(facescrub_feature_path, facescrub_feature)
    facescrub_label_path = os.path.join(configs["base_dir"], "facescrub_label.npy")
    np.save(facescrub_label_path, facescrub_label)
    megaface_feature_path = os.path.join(configs["base_dir"], "megaface.npy")
    np.save(megaface_feature_path, megaface_feature)

    return facescrub_feature, facescrub_label, megaface_feature


def calculate_score(configs, facescrub, labels, megaface):
    """calculate megaface identification top1 score. this evaluation implement strictly follows the description of
    `"The MegaFace Benchmark: 1 Million Faces for Recognition at Scale" <https://arxiv.org/pdf/1512.00596.pdf>`_
    this implement outputs exactly the same as dev-sdk provided by the official, but with much higher speed

    Args:
        configs (dict): configuration
        facescrub (np.array): feature of facescrub
        labels (np.array): label of facescrub
        megaface (np.array): feature of megaface

    Returns:
        megaface_score (float): top1 score of megaface
    """
    facescrub = mge.tensor(facescrub, dtype="float32")
    megaface = mge.tensor(megaface, dtype="float32")

    # note: (x - y) ** 2 = x ** 2 + y ** 2 - 2 * x * y
    # facescrub_score[i][j] = l2-dist(facescrub[i], facescrub[j])
    facescrub_score = (
        (facescrub ** 2).sum(axis=-1, keepdims=True)
        + (facescrub ** 2).sum(axis=-1, keepdims=True).transpose(1, 0)
        - 2 * F.matmul(facescrub, facescrub.transpose(1, 0))
    )
    facescrub_score = facescrub_score.numpy()

    def get_score_min_megaface(x):
        distr_score = (x ** 2).sum(axis=-1) + (megaface ** 2).sum(axis=-1) - 2 * (x * megaface).sum(axis=-1)
        return distr_score.min()

    up, down = 0, 0
    for probe_i in tqdm(range(len(facescrub))):
        distr_score_min = get_score_min_megaface(facescrub[probe_i]).numpy()
        mask = (labels == labels[probe_i]) & (np.arange(len(facescrub)) != probe_i)
        for probe_j in np.where(mask)[0]:
            probe_score = facescrub_score[probe_i][probe_j]
            up += probe_score < distr_score_min
            down += 1

    megaface_score = up / down * 100
    return megaface_score


def main(args):
    configs = load_config_from_path(args.config_file)

    configs["evaluate_epoch"] = args.epoch if args.epoch is not None else configs["num_epoch"]

    # write log to worklog.txt
    os.makedirs(configs["base_dir"], exist_ok=True)
    worklog_path = os.path.join(configs["base_dir"], "worklog.txt")
    mge.set_log_file(worklog_path)

    inference_func = get_inference_func(configs)
    facescrub_feature, facescrub_label, megaface_feature = extract_feature_and_clean_noise(configs, inference_func)
    megaface_score = calculate_score(configs, facescrub_feature, facescrub_label, megaface_feature)

    logger.info("Epoch: %d", configs["evaluate_epoch"])
    logger.info("MegaFace Top1: %.2f", megaface_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file", help="path to experiment configuration", required=True)
    parser.add_argument(
        "-e", "--epoch", help="model of num epoch to evaluate (default: num_epoch)", default=None, type=int
    )
    args = parser.parse_args()

    main(args)
