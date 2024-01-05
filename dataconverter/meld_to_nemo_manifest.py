import argparse
import functools
import logging
import os
import subprocess
from pathlib import Path
from typing import List

import pandas as pd

from dataconverter.data_converter_utils import convert_mp4_to_wav, find_emotion_keywords, process_dataframe


def __process_transcript(meta_data, data_folder: str, dst_folder: str, emotion_keywords: List[str]):
    file_name = meta_data[0]
    emotion = meta_data[1]
    text = meta_data[2]

    mp4_file = os.path.join(data_folder, f"{file_name}.mp4")

    if not os.path.exists(mp4_file):
        return []

    # Convert mp4 file to WAV
    wav_file = os.path.join(dst_folder, f"{file_name}.wav")
    if not os.path.exists(wav_file):
        convert_mp4_to_wav(mp4_file, wav_file)

    # Get duration
    duration = float(subprocess.check_output("soxi -D {0}".format(wav_file), shell=True))

    found_emotion_keywords = find_emotion_keywords(text, emotion_keywords)

    entry = {"audio_filepath": os.path.abspath(wav_file), "duration": duration, 'label': emotion, 'emotion_keywords': found_emotion_keywords, "text": text}

    return [entry]


def __process_data(metafile_csv: str, data_folder: str, dst_folder: str, manifest_file: str, num_workers: int, balance_classes: bool, emotion_keywords: List[str]):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    meta_data = pd.read_csv(metafile_csv)

    cleaned_df = meta_data[['Dialogue_ID', 'Utterance_ID', 'Utterance', 'Emotion']]
    cleaned_df['file_name'] = 'dia' + cleaned_df['Dialogue_ID'].astype(str) + '_utt' + cleaned_df['Utterance_ID'].astype(str)
    cleaned_df = cleaned_df.rename(columns={'Emotion': 'emotion', 'Utterance': 'text'})

    processing_func = functools.partial(__process_transcript, data_folder=data_folder, dst_folder=dst_folder, emotion_keywords=emotion_keywords)
    process_dataframe(cleaned_df, manifest_file, balance_classes, num_workers, processing_func)


def main(metafile, data_dir, num_workers, log, balance_classes, emotion_keywords):
    if log:
        logging.basicConfig(level=logging.INFO)

    logging.info("\n\nWorking on: {0}".format(data_dir))

    __process_data(metafile, data_dir, data_dir + "-wav", str(Path(metafile).with_suffix(".json")), num_workers, balance_classes, emotion_keywords)

    logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metafile_csv", required=True, default=None, type=str)
    parser.add_argument("--data_directory", required=True, default=None, type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--log", dest="log", action="store_true", default=False)
    parser.add_argument("--balance_classes", dest="balance_classes", action="store_true", default=False)
    parser.add_argument("--emotion_keywords", nargs="*", default=[], type=str, help="emotion keywords")
    args = parser.parse_args()
    main(args.metafile_csv, args.data_directory, args.num_workers, args.log, args.balance_classes, args.emotion_keywords)
