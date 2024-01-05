import argparse
import functools
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd

from dataconverter.data_converter_utils import find_emotion_keywords, process_dataframe, get_wav_duration


def __process_transcript(meta_data, data_folder: str, emotion_keywords: List[str]):
    file_name = meta_data[0]
    emotion = meta_data[1]
    text = meta_data[2]

    wav_file = os.path.join(data_folder, f"{file_name}.wav")

    # Get duration
    duration = get_wav_duration(wav_file)

    found_emotion_keywords = find_emotion_keywords(text, emotion_keywords)

    entry = {"audio_filepath": os.path.abspath(wav_file), "duration": duration, 'label': emotion, 'emotion_keywords': found_emotion_keywords, "text": text}

    return [entry]


def __process_data(num_classes: int, metafile_csv: str, translations_csv: str, data_folder: str, manifest_file: str, num_workers: int, balance_classes: bool,
                   emotion_keywords: List[str]):
    meta_data = pd.read_csv(metafile_csv)
    meta_info = meta_data[['main_emotion', '_id']]

    translations_data_all = pd.read_csv(translations_csv)
    translations_data = translations_data_all[['titre', 'to_translate']]

    merged_df = pd.merge(meta_info, translations_data, left_on='_id', right_on='titre', how='inner')
    cleaned_df = merged_df.rename(columns={'_id': 'file_name', 'main_emotion': 'emotion', 'to_translate': 'text'})
    cleaned_df = cleaned_df.map(lambda x: x.strip())

    # for 6 emotion classes
    if num_classes == 6:
        cleaned_df = cleaned_df[~cleaned_df['emotion'].str.contains('oth|dis|xxx|sur|fea')]

    elif num_classes == 4:
        # for 4 emotion classes
        cleaned_df = cleaned_df[~cleaned_df['emotion'].str.contains('oth|dis|xxx|sur|fea|fru')]
        cleaned_df['emotion'] = cleaned_df['emotion'].replace('exc', 'hap')

    else:
        raise ValueError("num_classes can only be 6 or 4")

    processing_func = functools.partial(__process_transcript, data_folder=data_folder, emotion_keywords=emotion_keywords)
    process_dataframe(cleaned_df, manifest_file, balance_classes, num_workers, processing_func, train_val_test_split=[0.7, 0.15, 0.15])


def main(num_classes, metafile, translations_file, data_dir, num_workers, balance_classes, emotion_keywords):
    logging.info("\n\nWorking on: {0}".format(data_dir))

    __process_data(num_classes, metafile, translations_file, data_dir, str(Path(metafile).with_suffix(".json")), num_workers, balance_classes, emotion_keywords)

    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, choices=[4, 6], required=True, help="Number of classes to be used to generate IEMOCAP-x dataset.")
    parser.add_argument("--metafile_csv", required=True, default=None, type=str)
    parser.add_argument("--translations_csv", required=True, default=None, type=str)
    parser.add_argument("--data_directory", required=True, default=None, type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--balance_classes", dest="balance_classes", action="store_true", default=False)
    parser.add_argument("--emotion_keywords", nargs="*", default=[], type=str, help="emotion keywords")
    args = parser.parse_args()
    main(args.num_classes, args.metafile_csv, args.translations_csv, args.data_directory, args.num_workers, args.balance_classes, args.emotion_keywords)
