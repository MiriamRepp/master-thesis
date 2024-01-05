from collections import OrderedDict
from typing import Optional

import torch
from nemo.collections.asr.parts.submodules.jasper import init_weights
from nemo.core import NeuralModule, Exportable, typecheck
from nemo.core.neural_types import NeuralType, LengthsType, AcousticEncodedRepresentation, LogitsType
from torch import nn

from nemo_models.ser_nemo_models.layers.AdaptiveAvgPool1dWithLengths import AdaptiveAvgPool1dWithLengths
from nemo_models.ser_nemo_models.layers.AdaptiveMaxPool1dWithLengths import AdaptiveMaxPool1dWithLengths


class ConvASRDecoderClassificationMaskedPooling(NeuralModule, Exportable):
    """Simple ASR Decoder for use with classification models such as JasperNet and QuartzNet

     Based on these papers:
        https://arxiv.org/pdf/2005.04290.pdf
    """

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(next(self.parameters()).device)
        lengths_example = torch.randn(max_batch).to(next(self.parameters()).device)
        return tuple([input_example, lengths_example])

    @property
    def input_types(self):
        return OrderedDict({
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType())
        })

    @property
    def output_types(self):
        return OrderedDict({"logits": NeuralType(('B', 'D'), LogitsType())})

    def __init__(
            self,
            feat_in: int,
            num_classes: int,
            init_mode: Optional[str] = "xavier_uniform",
            return_logits: bool = True,
            pooling_type: str = 'avg',
            dropout: float = 0,
            gru_hidden_size: int = None,
    ):
        super().__init__()

        self._feat_in = feat_in
        self._return_logits = return_logits
        self._num_classes = num_classes

        if pooling_type == 'avg':
            self.pooling = AdaptiveAvgPool1dWithLengths(1)
        elif pooling_type == 'max':
            self.pooling = AdaptiveMaxPool1dWithLengths(1)
        else:
            raise ValueError('Pooling type chosen is not valid. Must be either `avg` or `max`')

        if gru_hidden_size is not None:
            self.gru = nn.Sequential(nn.Dropout(p=dropout), nn.GRU(feat_in, gru_hidden_size, batch_first=True))
            self._feat_in = gru_hidden_size

        self.dropout = nn.Dropout(p=dropout)
        self.decoder_layers = self._create_decoder_layers()
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def _create_decoder_layers(self):
        return torch.nn.Sequential(torch.nn.Linear(self._feat_in, self._num_classes, bias=True))

    @typecheck()
    def forward(self, encoder_output, encoded_lengths):
        pooled_encoder_output = self._pool_encoder_output(encoder_output, encoded_lengths)

        logits = self.decoder_layers(pooled_encoder_output)  # [B, num_classes]
        return self._return_logits_or_softmax(logits)

    def _pool_encoder_output(self, encoder_output, encoded_lengths):
        encoder_output_permuted = encoder_output.permute((0, 2, 1))
        if hasattr(self, 'gru') and self.gru is not None:
            encoder_output_permuted = self.gru(encoder_output_permuted)[0]

        dropped = self.dropout(encoder_output_permuted)
        dropped_permuted = dropped.permute((0, 2, 1))

        pooled = self.pooling(dropped_permuted, encoded_lengths)
        return pooled

    def _return_logits_or_softmax(self, logits):
        if self._return_logits:
            return logits

        return torch.nn.functional.softmax(logits, dim=-1)

    @property
    def num_classes(self):
        return self._num_classes
