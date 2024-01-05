import torch
from torch import Tensor
from torch.nn import AdaptiveAvgPool1d
from torch.nn.functional import adaptive_avg_pool1d


class AdaptiveAvgPool1dWithLengths(AdaptiveAvgPool1d):

    def forward(self, input: Tensor, lengths: Tensor = None) -> Tensor:
        if lengths is None:
            return adaptive_avg_pool1d(input, self.output_size)

        arange_tensor = torch.arange(input.size(2)).to(input.get_device())
        mask: Tensor = arange_tensor[None, None, :] < lengths[:, None, None]

        # Ensure mask is the same type as data
        mask = mask.type_as(input)

        # Sum the actual data
        summed_data = torch.sum(input * mask, dim=2)

        # Compute the average
        avg_pooled = summed_data / lengths.float().unsqueeze(-1)

        return avg_pooled
