import megengine.functional as F
import megengine.module as M

from .head import get_head
from .loss import get_loss
from .resnet import get_backbone
from .stn import STN


class FaceRecognitionModel(M.Module):
    """combination of all building blocks, including backbone, head and loss
    """

    def __init__(self, configs):
        """initialize with configuration

        Args:
            configs (dict): configuration, required fields include:
                backbone: custom name of backbone
                output_head: custon name of output head
                feature_dim: dimension number of output embedding
                loss_type: custon name of loss function
                num_class: classification number of dataset
                loss_scale: used in loss function
                loss_m1: used in loss function
                loss_m2: used in loss function
                loss_m3: used in loss function
                use_stn: whether or not use stn
        """
        super().__init__()
        backbone_constructor = get_backbone(configs["backbone"])
        self.backbone = backbone_constructor()

        head_constructor = get_head(configs["output_head"])
        self.head = head_constructor(feature_dim=configs["feature_dim"], channel=self.backbone.output_channel)

        metric_constructor = get_loss(configs["loss_type"])
        self.metric = metric_constructor(
            num_class=configs["num_class"],
            scale=configs["loss_scale"],
            m1=configs["loss_m1"],
            m2=configs["loss_m2"],
            m3=configs["loss_m3"],
            feature_dim=configs["feature_dim"],
        )

        if configs["use_stn"]:
            self.stn = STN()
            self.use_stn = True
        else:
            self.use_stn = False

    def forward_embedding_only(self, images):
        """run forward pass without calculating loss, expected useful during evaluation.

        Args:
            images (Tensor): preprocessed images (shape: n * 3 * 112 * 112)

        Returns:
            embedding (Tensor): embedding
        """
        if self.use_stn:
            images = self.stn(images)
        feature_map = self.backbone(images)
        embedding = self.head(feature_map)
        embedding = F.normalize(embedding, axis=1)
        return embedding

    def forward(self, images, labels):
        """run forward pass and calculate loss, expected useful during training.

        Args:
            images (Tensor): preprocessed images (shape: n * 3 * 112 * 112)
            labels (Tensor): ground truth class id (shape: n)

        Returns:
            loss (Tensor): loss
            accuracy (Tensor): top1 accuracy (range: 0~1)
            embedding （Tensor): embedding
        """
        embedding = self.forward_embedding_only(images)
        loss, accuracy = self.metric(embedding, labels)
        return loss, accuracy, embedding
