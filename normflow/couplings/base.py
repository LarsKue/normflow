import torch


from normflow.common import Invertible


class Coupling(Invertible):
    """ Base Class for Couplings, Which Transform a Part of Their Input Parameterized by Another Part of the Input """

    def forward(self,
                parts: tuple[torch.Tensor, ...],
                *,
                condition: torch.Tensor = None,
                ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Forward transform at least one part based on others
        @param parts: Split Input Parts
        @param condition: Tensor that specifies what the input should be conditioned on, e.g. a one-hot class labeling
        @return: transformed parts, log(|det J|) where J is the corresponding Jacobian
        """
        raise NotImplementedError

    def inverse(self,
                transformed_parts: tuple[torch.Tensor, ...],
                *,
                condition: torch.Tensor = None,
                ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Inverse Transform at least one part based on others
        @param transformed_parts: Split and Transformed Parts
        @param condition: Tensor that specifies what the input should be conditioned on, e.g. a one-hot class labeling
        @return: inverse transformed parts, log(|det J|) where J is the corresponding Jacobian
        """
        raise NotImplementedError
