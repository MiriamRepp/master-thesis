import torch

from nemo_models.ser_nemo_models.decoders.ConvASRDecoderClassificationMaskedPooling import ConvASRDecoderClassificationMaskedPooling


class ThreeLayerConvASRDecoderClassification(ConvASRDecoderClassificationMaskedPooling):

    def _create_decoder_layers(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self._feat_in, 256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self._num_classes, bias=True)
        )