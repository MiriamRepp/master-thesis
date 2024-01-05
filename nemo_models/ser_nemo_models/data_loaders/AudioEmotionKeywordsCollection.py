import collections
import json
import logging
from typing import *

from nemo.collections.common.parts.preprocessing import manifest
from nemo.collections.common.parts.preprocessing.collections import _Collection


class AudioEmotionKeywordsCollection(_Collection):
    """List of audio-label correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(typename='SpeechLabelEntity', field_names='audio_file duration label offset keywords', )

    def __init__(
            self,
            manifests_files: Union[str, List[str]],
            min_duration: Optional[float] = None,
            max_duration: Optional[float] = None,
            max_number: Optional[int] = None,
    ):
        audio_files, durations, labels, offsets, all_keywords = [], [], [], [], []

        for item in manifest.item_iter(manifests_files, parse_func=self.__parse_item):
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])

            labels.append(item['label'])
            offsets.append(item['offset'])
            all_keywords.append(item['keywords'])


        # Adapted from SpeechLabel
        output_type = self.OUTPUT_TYPE
        data, duration_filtered = [], 0.0
        total_duration = 0.0

        for audio_file, duration, label, offset, keywords in zip(audio_files, durations, labels, offsets, all_keywords):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                continue

            data.append(output_type(audio_file, duration, label, offset, keywords))
            total_duration += duration

            # Max number of entities filter.
            if len(data) == max_number:
                break

        logging.info(f"Filtered duration for loading collection is {duration_filtered / 3600: .2f} hours.")
        logging.info(f"Dataset loaded with {len(data)} items, total duration of {total_duration / 3600: .2f} hours.")
        self.uniq_labels = sorted(set(map(lambda x: x.label, data)))
        logging.info("# {} files loaded accounting to # {} labels".format(len(data), len(self.uniq_labels)))

        super().__init__(data)

    @staticmethod
    def __parse_item(line: str, manifest_file: str) -> Dict[str, Any]:
        item = json.loads(line)

        # Audio file
        if 'audio_filepath' in item:
            item['audio_file'] = item.pop('audio_filepath')
        else:
            raise ValueError(f"Manifest file has invalid json line structure: {line} without proper audio file key.")

        item['audio_file'] = manifest.get_full_path(audio_file=item['audio_file'], manifest_file=manifest_file)

        # Duration.
        if 'duration' not in item:
            raise ValueError(f"Manifest file has invalid json line structure: {line} without proper duration key.")

        # Label.
        if 'label' in item:
            pass
        else:
            raise ValueError(f"Manifest file has invalid json line structure: {line} without proper label key.")

        # Keywords.
        if 'emotion_keywords' in item:
            pass
        else:
            raise ValueError(f"Manifest file has invalid json line structure: {line} without proper label key.")

        item = dict(
            audio_file=item['audio_file'],
            duration=item['duration'],
            label=item['label'],
            offset=item.get('offset', None),
            keywords=item['emotion_keywords']
        )

        return item
