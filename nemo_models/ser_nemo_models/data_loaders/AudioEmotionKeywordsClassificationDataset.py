import logging
from typing import Optional, Dict, List, Union

import torch
from nemo.collections.asr.data.audio_to_label import _speech_collate_fn
from nemo.collections.asr.data.audio_to_text import cache_datastore_manifests
from nemo.core import Dataset
from nemo.core.neural_types import LengthsType, NeuralType, LabelsType, AudioSignal
from omegaconf import DictConfig
from torch import nn

from nemo_models.ser_nemo_models.data_loaders.AudioEmotionKeywordsCollection import AudioEmotionKeywordsCollection


class AudioEmotionKeywordsClassificationDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """

        output_types = {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate) if self is not None and hasattr(self, '_sample_rate') else AudioSignal(), ),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),

            'label': NeuralType(tuple('B'), LabelsType()),
            'label_length': NeuralType(tuple('B'), LengthsType()),

            'keywords': NeuralType(('B', 'T'), LabelsType()),
        }

        return output_types

    def __init__(
            self,
            *,
            manifest_filepath: Union[str, List[str]],
            labels: List[str],
            keyword_labels: List[str],
            featurizer,
            min_duration: Optional[float] = 0.1,
            max_duration: Optional[float] = None,
            trim: bool = False,
    ):
        super().__init__()
        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(',')

        cache_datastore_manifests(manifest_filepaths=manifest_filepath, cache_audio=True)

        self.collection = AudioEmotionKeywordsCollection(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        self.featurizer = featurizer
        self.trim = trim

        self.labels = labels if labels else self.collection.uniq_labels
        self.keyword_labels = keyword_labels

        self.num_classes = len(self.labels) if self.labels is not None else 1
        self.num_keyword_classes = len(self.keyword_labels)

        self.label2id, self.id2label = {}, {}
        for label_id, label in enumerate(self.labels):
            self.label2id[label] = label_id
            self.id2label[label_id] = label

        self.keyword_label2id, self.id2keyword_label = {}, {}
        for label_id, label in enumerate(self.keyword_labels):
            self.keyword_label2id[label] = label_id
            self.id2keyword_label[label_id] = label

        for idx in range(len(self.labels[:5])):
            logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        sample = self.collection[index]

        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim)
        f, fl = features, torch.tensor(features.shape[0]).long()

        t = torch.tensor(self.label2id[sample.label]).long()
        tl = torch.tensor(1).long()  # For compatibility with collate_fn used later

        k = self._create_keywords_multi_hot_tensor(sample.keywords)

        return f, fl, t, tl, k

    def _create_keywords_multi_hot_tensor(self, keywords):
        keyword_ids = torch.tensor([self.keyword_label2id[keyword] for keyword in keywords]).long()
        keyword_one_hots = nn.functional.one_hot(keyword_ids, num_classes=self.num_keyword_classes)
        keyword_multi_hot = keyword_one_hots.sum(dim=0).float()
        return keyword_multi_hot


    def _collate_fn(self, batch):
        orig_audio_signal, orig_audio_lengths, orig_labels, orig_tokens_lengths, orig_keywords = zip(*batch)

        audio_signal, audio_lengths, labels, labels_lengths= _speech_collate_fn(list(zip(orig_audio_signal, orig_audio_lengths, orig_labels, orig_tokens_lengths)), pad_id=0)

        keywords = torch.stack(orig_keywords)

        return audio_signal, audio_lengths, labels, labels_lengths, keywords


def get_emotion_keywords_classification_dataset(featurizer, config: DictConfig) -> AudioEmotionKeywordsClassificationDataset:
    dataset = AudioEmotionKeywordsClassificationDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        keyword_labels=config['keyword_labels'],
        featurizer=featurizer,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
    )
    return dataset
