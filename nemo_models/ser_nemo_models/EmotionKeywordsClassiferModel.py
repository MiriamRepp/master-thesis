import logging

import torch
from nemo.collections.asr.parts.preprocessing import process_augmentations, WaveformFeaturizer
from omegaconf import DictConfig

from nemo_models.ser_nemo_models.ConformerClassifierModel import ConformerClassifierModel
from nemo_models.ser_nemo_models.data_loaders.AudioEmotionKeywordsClassificationDataset import get_emotion_keywords_classification_dataset


class EmotionKeywordsClassifierModel(ConformerClassifierModel):

    def _setup_dataloader_from_config(self, config: DictConfig):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        featurizer = WaveformFeaturizer(sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=augmentor)
        shuffle = config['shuffle']

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` is None. Provided config : {config}")
            return None

        dataset = get_emotion_keywords_classification_dataset(featurizer=featurizer, config=config)
        batch_size = config['batch_size']

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )
