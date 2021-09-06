# Copyright (c) Megvii, Inc. and its affiliates.

configs = {
    # ------------ Basic Configuration ------------
    "batch_size": 64,
    "input_size": [112, 112],
    # ------------ Training Configuration ------------
    "learning_rate": 0.1 / 8,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    # ------------ IO Configuration ------------
    "base_dir": "/home/megstudio/workspace/megengine-face-recognition/model/v0.7",
    "dataset_dir": "/home/megstudio/workspace/megengine-face-recognition/dataset",
    "log_interval": 1,
    # ------------ Dataset Configuration ------------
    "dataset": "webface",
    "num_class": 10572,
    "learning_rate_milestons": [20, 28],
    "learning_rate_gamma": 0.1,
    "num_epoch": 32,
    # ------------ Model Configuration ------------
    "backbone": "resnet50",
    "output_head": "bn_dropout_gap_fc_bn",
    "feature_dim": 512,
    # ------------ Loss Configuration ------------
    # loss function: margined_logit = s * (cos(m1 * theta + m2) - m3)
    # m1 != 1.0, m2 == 0.0, m3 == 0.0 is used in SphereFace, which is not implemented in this codebase
    # m1 == 1.0, m2 != 0.0, m3 == 0.0 is used in ArcFace
    # m1 == 1.0, m2 == 0.0, m3 != 0.0 is used in CosFace
    # other combinations of (m1, m2, m3) are also welcomed.
    "loss_type": "cosface",
    "loss_scale": 30,
    "loss_m1": 1.0,
    "loss_m2": 0.0,
    "loss_m3": 0.35,
    "loss_mat3x3_ratio": 1,
}
