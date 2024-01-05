import torch
from torch import Tensor
from torch.nn import AdaptiveAvgPool1d
from torch.nn.functional import adaptive_max_pool1d


class AdaptiveMaxPool1dWithLengths(AdaptiveAvgPool1d):

    def forward(self, input: Tensor, lengths: Tensor = None) -> Tensor:
        if lengths is None:
            return adaptive_max_pool1d(input, self.output_size)

        arange_tensor = torch.arange(input.size(2)).to(input.get_device())
        mask: Tensor = arange_tensor[None, None, :] < lengths[:, None, None]

        # get the max of the actual data
        masked_input = input.masked_fill(mask == False, float('-inf'))

        return masked_input.amax(dim=2)
