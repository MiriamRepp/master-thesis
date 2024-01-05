from typing import List

from nemo.core import PretrainedModelInfo
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo_models.ser_nemo_models.LoggingEncDecClassificationModel import LoggingEncDecClassificationModel


class ConformerEncDecClassificationModel(LoggingEncDecClassificationModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_small",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_small",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_small/versions/1.6.0/files/stt_en_conformer_ctc_small.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_ctc_large/versions/1.0.0/files/stt_en_fastconformer_ctc_large.nemo",
        )
        results.append(model)

        return results
