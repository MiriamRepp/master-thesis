from typing import Optional

import numpy as np
import torch
from nemo.collections.asr.models import EncDecClassificationModel
from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.common.metrics import TopKClassificationAccuracy
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo_models.ser_nemo_models.LoggingEncDecClassificationModel import LoggingEncDecClassificationModel


class WeightedEncDecClassificationModel(LoggingEncDecClassificationModel):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.class_weights = None

        super().__init__(cfg, trainer)

    def _setup_loss(self):
        return CrossEntropyLoss(weight=self.class_weights)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        super().setup_training_data(train_data_config)

        if train_data_config.class_balancing == 'weighted_loss':
            self.class_weights = self._calc_class_weights()
            self.loss = self._setup_loss()
        else:
            self.class_weights = None

    def _calc_class_weights(self):
        percentages = np.asarray([47.151867, 17.449194, 12.063270, 11.102212, 6.837521, 2.712984, 2.682951])  # TODO make it dynamic

        weights = 100 / (len(percentages) * percentages)

        return torch.tensor(weights, dtype=torch.float32)