# Copyright (c) Megvii, Inc. and its affiliates.

import itertools

import megengine as mge
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
        x = self.stem(image)
        x = F.avg_pool2d(x, 7)
        x = F.flatten(x, 1)
        x = self.fc(x)

        s = self.input_size
        # 0.01 here is a magic number. it aims to maintain identity transform at early stage of training
        residual = x.reshape(-1, 3, 3) * 0.01
        base = mge.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype("float32")
        base = F.broadcast_to(base, residual.shape)
        left_scale = mge.tensor([[s, 0, 0], [0, s, 0], [0, 0, 1]]).astype("float32")
        left_scale = F.broadcast_to(left_scale, residual.shape)
        right_scale = mge.tensor([[1 / s, 0, 0], [0, 1 / s, 0], [0, 0, 1]]).astype("float32")
        right_scale = F.broadcast_to(right_scale, residual.shape)
        mat3x3 = F.matmul(left_scale, F.matmul(base + residual, right_scale))
        return mat3x3

    def forward(self, image):
        mat3x3 = self._get_mat3x3(image)
        transformed_image = self._get_transformed_image(image, mat3x3)
        return transformed_image


class GridSTN(M.Module):
    """Grid STN from
    `"GridFace: Face Rectification via Learning Local Homography Transformations" <https://arxiv.org/pdf/1808.06210.pdf>`_
    some detailed implements are highly simplified while good performance maintained
    """

    def __init__(self, input_size=112, num_grid=4):
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
        self.num_grid = num_grid
        self.fc = M.Linear(64, 9 * num_grid * num_grid)

    def _get_transformed_image(self, image, mat3x3s):
        """apply the local homography transformation
        """
        n = self.num_grid
        s = self.input_size

        i = 0
        row_buff = []
        for dx in range(n):
            col_buff = []
            for dy in range(n):
                patch = F.warp_perspective(image, mat3x3s[i], [s // n, s // n])
                col_buff.append(patch)
                i += 1
            row_buff.append(F.concat(col_buff, axis=2))
        patches = F.concat(row_buff, axis=3)
        return patches

    def _get_mat3x3s(self, image):
        """a list of `num_grid * num_grid` mat3x3, used in the local homography transformation.
        """
        x = self.stem(image)
        x = F.avg_pool2d(x, 7)
        x = F.flatten(x, 1)
        x = self.fc(x)

        s = self.input_size
        n = self.num_grid
        mat3x3s = []
        for i, (dx, dy) in enumerate(itertools.product(range(n), range(n))):
            # 0.01 here is a magic number. it aims to maintain identity transform at early stage of training
            residual = x[:, 9 * i : 9 * i + 9].reshape(-1, 3, 3) * 0.01
            base = mge.tensor([[1, 0, dx / n], [0, 1, dy / n], [0, 0, 1]]).astype("float32")
            base = F.broadcast_to(base, residual.shape)
            left_scale = mge.tensor([[s, 0, 0], [0, s, 0], [0, 0, 1]]).astype("float32")
            left_scale = F.broadcast_to(left_scale, residual.shape)
            right_scale = mge.tensor([[1 / s, 0, 0], [0, 1 / s, 0], [0, 0, 1]]).astype("float32")
            right_scale = F.broadcast_to(right_scale, residual.shape)
            mat3x3 = F.matmul(left_scale, F.matmul(base + residual, right_scale))
            mat3x3s.append(mat3x3)
        return mat3x3s

    def get_mat3x3_loss(self, mat3x3s):
        """get deformable constraint supervision.
        """
        batch_size = mat3x3s[0].shape[0]
        s = self.input_size
        n = self.num_grid
        patch_size = s / n

        ori = mge.tensor([[0], [0], [1]]).astype("float32")
        ori = F.broadcast_to(ori, (batch_size, 3, 1))
        right = mge.tensor([[0], [patch_size], [1]]).astype("float32")
        right = F.broadcast_to(right, (batch_size, 3, 1))
        down = mge.tensor([[patch_size], [0], [1]]).astype("float32")
        down = F.broadcast_to(down, (batch_size, 3, 1))
        diag = mge.tensor([[patch_size], [patch_size], [1]]).astype("float32")
        diag = F.broadcast_to(diag, (batch_size, 3, 1))

        mat3x3_loss = []
        for i, (dx, dy) in enumerate(itertools.product(range(n), range(n))):
            if dy + 1 < n:
                u = F.matmul(mat3x3s[i], right)
                u = u[:, :2] / u[:, 2:] / s
                v = F.matmul(mat3x3s[i + 1], ori)
                v = v[:, :2] / v[:, 2:] / s
                mat3x3_loss.append(((u - v) ** 2).mean())

                u = F.matmul(mat3x3s[i], diag)
                u = u[:, :2] / u[:, 2:] / s
                v = F.matmul(mat3x3s[i + 1], down)
                v = v[:, :2] / v[:, 2:] / s
                mat3x3_loss.append(((u - v) ** 2).mean())
            if dx + 1 < n:
                u = F.matmul(mat3x3s[i], down)
                u = u[:, :2] / u[:, 2:] / s
                v = F.matmul(mat3x3s[i + n], ori)
                v = v[:, :2] / v[:, 2:] / s
                mat3x3_loss.append(((u - v) ** 2).mean())

                u = F.matmul(mat3x3s[i], diag)
                u = u[:, :2] / u[:, 2:] / s
                v = F.matmul(mat3x3s[i + n], right)
                v = v[:, :2] / v[:, 2:] / s
                mat3x3_loss.append(((u - v) ** 2).mean())

        mat3x3_loss = sum(mat3x3_loss)
        return mat3x3_loss

    def forward(self, image):
        mat3x3s = self._get_mat3x3s(image)
        transformed_image = self._get_transformed_image(image, mat3x3s)
        return transformed_image, mat3x3s
