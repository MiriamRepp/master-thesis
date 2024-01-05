from collections import OrderedDict
from typing import Optional

import torch
from nemo.collections.asr.parts.submodules.jasper import init_weights
from nemo.core import typecheck, NeuralModule, Exportable
from nemo.core.neural_types import NeuralType, LengthsType, AcousticEncodedRepresentation, LogitsType
from torch import nn


class GruKeywordsPredictingClassificationDecoderV2(NeuralModule, Exportable):

    @property
    def input_types(self):
        return OrderedDict({
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        })

    @property
    def output_types(self):
        return OrderedDict({
            "logits": NeuralType(('B', 'D'), LogitsType()),
            "keywords_logits": NeuralType(('B', 'D'), LogitsType())
        })

    def __init__(
            self,
            feat_in: int,
            num_classes: int,
            num_keyword_classes: int,
            gru_hidden_size: int,
            init_mode: Optional[str] = "xavier_uniform",
            return_logits: bool = True,
            dropout: float = 0.5,
    ):
        super().__init__()

        self._return_logits = return_logits

        self._initial_dropout = nn.Dropout(p=dropout)

        self._keyword_gru_layer = nn.GRU(feat_in, gru_hidden_size, batch_first=True)
        self._keyword_dropout = nn.Dropout(p=dropout)
        self._keyword_decoder = torch.nn.Linear(gru_hidden_size, num_keyword_classes, bias=True)

        self._emotion_gru_layer = nn.GRU(feat_in, gru_hidden_size, batch_first=True)
        self._emotion_dropout = nn.Dropout(p=dropout)
        self._emotion_decoder = torch.nn.Linear(gru_hidden_size + num_keyword_classes, num_classes, bias=True)

        self.apply(lambda x: init_weights(x, mode=init_mode))

    @typecheck()
    def forward(self, encoder_output, encoded_lengths):
        encoder_output_permuted = encoder_output.permute((0, 2, 1))
        encoder_dropped = self._initial_dropout(encoder_output_permuted)

        keyword_gru_result = self._keyword_gru_layer(encoder_dropped)[1][0]
        keyword_gru_result_dropped = self._keyword_dropout(keyword_gru_result)
        keyword_logits = self._keyword_decoder(keyword_gru_result_dropped)
        keyword_logits_softmax = torch.nn.functional.softmax(keyword_logits, dim=-1)

        emotion_gru_result = self._emotion_gru_layer(encoder_dropped)[1][0]
        emotion_gru_result_dropped = self._emotion_dropout(emotion_gru_result)

        emotion_gru_with_keywords = torch.cat([emotion_gru_result_dropped, keyword_logits_softmax], dim=1)
        emotion_logits = self._emotion_decoder(emotion_gru_with_keywords)

        return emotion_logits, keyword_logits

    @property
    def num_classes(self):
        return self._num_classes

    def _return_logits_or_softmax(self, logits):
        if self._return_logits:
            return logits

        return torch.nn.functional.softmax(logits, dim=-1)
