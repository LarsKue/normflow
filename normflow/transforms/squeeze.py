
import torch


from .base import Transform


class SqueezeTransform(Transform):
    """
    A squeeze transform reshapes the input [c, h, w] to [4 * c, h / 2, w / 2]

    Introduced by arXiv:1605.08803
    """
    def __init__(self, subsize: int = 2):
        super().__init__()
        self.subsize = subsize

    def forward(self, x: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = x.shape

        # divide the image into subsquares of shape 2 x 2 x c by adding dimensions
        z = x.reshape(
            batch_size,
            channels,

            height // self.subsize,
            self.subsize,

            width // self.subsize,
            self.subsize
        )

        # move extra dimensions to the channel dimension
        z = torch.movedim(z, (3, 5), (2, 3))

        # merge the subsquares into the channel dimension
        z = z.reshape(
            batch_size,
            channels * self.subsize ** 2,
            height // self.subsize,
            width // self.subsize
        )

        logabsdet = z.new_zeros(batch_size)

        return z, logabsdet

    def inverse(self, z: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = z.shape

        # correct the dimensions
        channels = channels // (self.subsize ** 2)
        height = height * self.subsize
        width = width * self.subsize

        # undo merging the subsquares into the channel dimension
        z = z.reshape(
            batch_size,
            channels,
            self.subsize,
            self.subsize,
            height // self.subsize,
            width // self.subsize
        )

        # move the dimensions back
        z = torch.movedim(z, (2, 3), (3, 5))

        # merge extra dimensions back into height and width
        x = z.reshape(
            batch_size,
            channels,
            height,
            width,
        )

        logabsdet = x.new_zeros(batch_size)

        return x, logabsdet
