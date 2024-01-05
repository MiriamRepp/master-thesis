from collections import OrderedDict
from typing import Optional

import torch
from nemo.core import typecheck
from nemo.core.neural_types import NeuralType, LengthsType, AcousticEncodedRepresentation, LogitsType, LabelsType
from torch import nn

from nemo_models.ser_nemo_models.decoders.ConvASRDecoderClassificationMaskedPooling import ConvASRDecoderClassificationMaskedPooling


class KeywordsEnhancedClassificationDecoder(ConvASRDecoderClassificationMaskedPooling):

    @property
    def input_types(self):
        return OrderedDict({
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "keywords": NeuralType(('B', 'T'), LabelsType()),
        })

    @property
    def output_types(self):
        return OrderedDict({"logits": NeuralType(('B', 'D'), LogitsType())})

    def __init__(
            self,
            feat_in: int,
            num_classes: int,
            num_keyword_classes: int,
            init_mode: Optional[str] = "xavier_uniform",
            return_logits: bool = True,
            pooling_type: str = 'avg',
            dropout: float = 0,
            gru_hidden_size: int = None,
    ):
        self._num_keyword_classes = num_keyword_classes
        super().__init__(feat_in, num_classes, init_mode, return_logits, pooling_type, dropout, gru_hidden_size)
        self.keyword_dropout = nn.Dropout(p=dropout)

    def _create_decoder_layers(self):
        return torch.nn.Sequential(torch.nn.Linear(self._feat_in + self._num_keyword_classes, self._num_classes, bias=True))

    @typecheck()
    def forward(self, encoder_output, encoded_lengths, keywords):
        pooled_encoder_output = self._pool_encoder_output(encoder_output, encoded_lengths)
        dropped_keywords = self.keyword_dropout(keywords)

        combined_encoder_keywords = torch.cat([pooled_encoder_output, dropped_keywords], dim=1)

        logits = self.decoder_layers(combined_encoder_keywords)  # [B, num_classes]
        return self._return_logits_or_softmax(logits)

    @property
    def num_classes(self):
        return self._num_classes
