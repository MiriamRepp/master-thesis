from collections import OrderedDict
from typing import Optional

import torch
from nemo.core import typecheck, NeuralModule, Exportable
from nemo.core.neural_types import NeuralType, LengthsType, AcousticEncodedRepresentation, LogitsType

from nemo_models.ser_nemo_models.decoders.ConvASRDecoderClassificationMaskedPooling import ConvASRDecoderClassificationMaskedPooling
from nemo_models.ser_nemo_models.decoders.KeywordsEnhancedClassificationDecoder import KeywordsEnhancedClassificationDecoder


class KeywordsPredictingClassificationDecoder(NeuralModule, Exportable):

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
            init_mode: Optional[str] = "xavier_uniform",
            return_logits: bool = True,
            pooling_type: str = 'avg',
            dropout: float = 0,
            gru_hidden_size: int = None,
    ):
        super().__init__()
        self._keywords_decoder = ConvASRDecoderClassificationMaskedPooling(feat_in, num_keyword_classes, init_mode, return_logits, 'avg', dropout, gru_hidden_size)
        self._emotion_decoder = KeywordsEnhancedClassificationDecoder(feat_in, num_classes, num_keyword_classes, init_mode, return_logits, pooling_type, dropout, gru_hidden_size)

    @typecheck()
    def forward(self, encoder_output, encoded_lengths):
        keywords_logits = self._keywords_decoder(encoder_output=encoder_output, encoded_lengths=encoded_lengths)
        keywords = torch.sigmoid(keywords_logits)

        emotion_logits = self._emotion_decoder(encoder_output=encoder_output, encoded_lengths=encoded_lengths, keywords=keywords)

        return emotion_logits, keywords_logits

    @property
    def num_classes(self):
        return self._num_classes
