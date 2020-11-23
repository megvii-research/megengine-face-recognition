import megengine.functional as F
import megengine.module as M

from .resnet import BasicBlock


class STN(M.Module):
    """spatial transformer networks from
    `"Spatial Transformer Networks" <https://arxiv.org/pdf/1506.02025.pdf>`_
    some detailed implements are highly simplified while good performance maintained
    """

    def __init__(self, input_size=112):
        assert input_size == 112, f"expected input_size == 112, got {input_size}"
        super().__init__()
        self.input_size = input_size
        self.stem = M.Sequential(
            M.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            M.BatchNorm2d(8),
            M.ReLU(),
            M.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(8, 16),
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 64, stride=2),
        )
        self.fc = M.Linear(64, 9)

    def _get_transformed_image(self, image, mat3x3):
        """apply perspective transform to the image
        note: there is NO need to guarantee the bottom right element equals 1

        Args:
            image (Tensor): input images (shape: n * 3 * 112 * 112)
            mat3x3 (Tensor): perspective matrix (shape: n * 3 * 3)

        Returns:
            transformed_image (Tensor): perspectively transformed image
        """
        s = self.input_size
        transformed_image = F.warp_perspective(image, mat3x3, [s, s])
        return transformed_image

    def _get_mat3x3(self, image):
        """get perspective matrix used in the transformation
        note: there are only 8 degrees of freedom in a perspective matrix, while the output matrix has 9 variables.

        Args:
            image (Tensor): input images (shape: n * 3 * 112 * 112)

        Returns:
            mat3x3 (Tensor): perspective matrix (shape: n * 3 * 3)
        """
        raise NotImplementedError("please implement me!")

    def forward(self, image):
        mat3x3 = self._get_mat3x3(image)
        transformed_image = self._get_transformed_image(image, mat3x3)
        return transformed_image
