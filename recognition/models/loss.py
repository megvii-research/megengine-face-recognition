import math

import megengine.functional as F
import megengine.module as M


class LogitsFullyConnected(M.Module):
    """single fully connected layer, mapping embedding to logits with normalized weight
    """

    def __init__(self, num_class, feature_dim):
        super().__init__()
        fc = M.Linear(feature_dim, num_class, bias=False)
        self.weight = fc.weight
        M.init.msra_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, embedding):
        w = F.normalize(self.weight, axis=1)
        x = embedding  # embedding has been normalized already
        logits = F.matmul(x, w.transpose(1, 0))
        return logits


class AdditiveMarginSoftmax(M.Module):
    """additive margin softmax from
    `"Additive Margin Softmax for Face Verification" <https://arxiv.org/pdf/1801.05599.pdf>`_
    and
    `"CosFace: Large Margin Cosine Loss for Deep Face Recognition" <https://arxiv.org/pdf/1801.09414.pdf>`_

    """

    def __init__(self, num_class, scale, m1, m2, m3, feature_dim=512):
        assert m1 == 1.0, f"m1 expected to be 1.0 in AdditiveMarginSoftmax, got {m1}"
        assert m2 == 0.0, f"m2 expected to be 0.0 in AdditiveMarginSoftmax, got {m2}"

        super().__init__()
        self.fc = LogitsFullyConnected(num_class, feature_dim)
        self.num_class = num_class
        self.scale = scale
        self.margin = m3

    def forward(self, embedding, target):
        origin_logits = self.fc(embedding)
        one_hot_target = F.one_hot(target, self.num_class)

        # get how much to decrease
        delta_one_hot_target = one_hot_target * self.margin

        # apply the decrease
        logits = origin_logits - delta_one_hot_target
        logits = logits * self.scale
        loss = F.loss.cross_entropy(logits, target)
        accuracy = F.topk_accuracy(origin_logits, target, topk=1)
        return loss, accuracy


class AdditiveAngularMarginSoftmax(M.Module):
    """additive angular margin softmax from
    `"ArcFace: Additive Angular Margin Loss for Deep Face Recognition" <https://arxiv.org/pdf/1801.07698.pdf>`_
    """

    def __init__(self, num_class, scale, m1, m2, m3, feature_dim=512):
        assert m1 == 1.0, f"m1 expected to be 1.0 in AdditiveAngularMarginSoftmax, got {m1}"
        assert m3 == 0.0, f"m3 expected to be 0.0 in AdditiveAngularMarginSoftmax, got {m3}"

        super().__init__()
        self.fc = LogitsFullyConnected(num_class, feature_dim)
        self.num_class = num_class
        self.scale = scale
        self.margin = m2

    def forward(self, embedding, target):
        origin_logits = self.fc(embedding)
        one_hot_target = F.one_hot(target, self.num_class).astype("bool")
        large_margined_logit = F.cos(F.acos(origin_logits) + self.margin)
        small_margined_logit = origin_logits
        margined_logit = F.where(origin_logits >= 0, large_margined_logit, small_margined_logit)
        logits = F.where(one_hot_target, margined_logit, origin_logits)
        logits = logits * self.scale
        loss = F.loss.cross_entropy(logits, target)
        accuracy = F.topk_accuracy(origin_logits, target, topk=1)
        return loss, accuracy


def get_loss(name):
    """get loss class by name

    Args:
        name (str): costum name of loss

    Returns:
        M.Module: corresponding loss class
    """
    mapping = {
        "cosface": AdditiveMarginSoftmax,
        "arcface": AdditiveAngularMarginSoftmax,
    }
    assert name in mapping, f"head {name} is not found, choose one from {mapping.keys()}"
    return mapping[name]
