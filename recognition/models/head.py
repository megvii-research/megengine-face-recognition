import megengine.functional as F
import megengine.module as M


class BNDropoutGAPFCBN(M.Module):
    """BN-Dropout-GAP-FC-BN output head used in
    """

    def __init__(self, feature_dim, channel, size=7):
        """initialzation

        Args:
            feature_dim (int): dimension number of output embedding
            channel (int): channel number of input feature map
            size (int, optional): size of input feature map. defaults to 7
        """
        super().__init__()
        self.size = size
        self.bn1 = M.BatchNorm2d(channel)
        self.dropout = M.Dropout(drop_prob=0.4)
        self.fc = M.Linear(channel, feature_dim)
        self.bn2 = M.BatchNorm1d(feature_dim, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.avg_pool2d(x, self.size)
        x = F.flatten(x, 1)
        x = self.fc(x)
        x = self.bn2(x)
        return x


class BNDropoutFCBN(M.Module):
    """BN-Dropout-FC-BN output head used in
    `"ArcFace: Additive Angular Margin Loss for Deep Face Recognition" <https://arxiv.org/pdf/1801.07698.pdf>`_
    """

    def __init__(self, feature_dim, channel, size=7):
        """initialzation

        Args:
            feature_dim (int): dimension number of output embedding
            channel (int): channel number of input feature map
            size (int, optional): size of input feature map. defaults to 7
        """
        super().__init__()
        raise NotImplementedError("please implement me!")

    def forward(self, x):
        raise NotImplementedError("please implement me!")


def get_head(name):
    """get head class by name

    Args:
        name (str): costum name of head

    Returns:
        M.Module: corresponding head class
    """
    mapping = {
        "bn_dropout_gap_fc_bn": BNDropoutGAPFCBN,
        "bn_dropout_fc_bn": BNDropoutFCBN,
    }
    assert name in mapping, f"head {name} is not found, choose one from {mapping.keys()}"
    return mapping[name]
