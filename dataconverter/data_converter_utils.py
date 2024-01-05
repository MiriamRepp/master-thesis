import codecs
import json
import logging
import multiprocessing
import os
import random
import subprocess
from collections import Counter
from itertools import groupby

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from emotion_keyword_extraction.emotion_keyword_extractor import preprocess_text


def convert_mp4_to_wav(mp4_file, wav_file):
    command = f"ffmpeg -i {mp4_file} -ab 160k -ac 1 -vn {wav_file}"
    subprocess.call(command, shell=True)


def find_emotion_keywords(text, emotion_keywords):
    preprocessed_text = preprocess_text(text)
    return list(set(filter(lambda word: word in emotion_keywords, preprocessed_text)))


def process_entries(data_frame, num_workers, processing_func):
    selected_data = list(data_frame[['file_name', 'emotion', 'text']].values)

    log_class_distributions(data_frame)

    entries = []
    with multiprocessing.Pool(num_workers) as p:
        results = p.imap(processing_func, selected_data)
        for result in tqdm(results, total=len(selected_data)):
            entries.extend(result)

    return entries


def process_dataframe(data_frame, manifest_file, balance_classes, num_workers, processing_func, train_val_test_split=None):
    entries = process_entries(data_frame, num_workers, processing_func)

    if train_val_test_split is not None:
        entries_train, entries_val, entries_test = make_train_val_test_split(entries, train_val_test_split[0], train_val_test_split[1], train_val_test_split[2])

        if balance_classes:
            entries_train = balance_classes_by_labels(entries_train)

        base_file_name, extension = os.path.splitext(manifest_file)
        write_manifest_json(entries_train, f'{base_file_name}-train{extension}')
        write_manifest_json(entries_val, f'{base_file_name}-val{extension}')
        write_manifest_json(entries_test, f'{base_file_name}-test{extension}')

    else:
        if balance_classes:
            entries = balance_classes_by_labels(entries)

        write_manifest_json(entries, manifest_file)


def balance_classes_by_labels(entries):
    copy_factors = calculate_copy_factors_for_label(entries)

    entries_by_class = {k: list(g) for k, g in groupby(sorted(entries, key=lambda x: x['label']), lambda x: x['label'])}
    entries_with_copies_by_class = {k: e * copy_factors[k] for k, e in entries_by_class.items()}
    entries_with_copies = [item for sublist in entries_with_copies_by_class.values() for item in sublist]
    random.shuffle(entries_with_copies)

    logging.info(f'balancing classes by labels: extended {len(entries)} entries to {len(entries_with_copies)} entries')

    return entries_with_copies


def write_manifest_json(entries, manifest_file):
    num_entries_with_emotion_keywords = len(list(filter(lambda entry: len(entry['emotion_keywords']) > 0, entries)))
    print('Number of entries with emotion keywords: ', num_entries_with_emotion_keywords, ' of ', len(entries))

    random.shuffle(entries)

    with codecs.open(manifest_file, "w", "utf-8") as fout:
        for m in entries:
            fout.write(json.dumps(m, ensure_ascii=False) + "\n")


def log_class_distributions(meta_data):
    value_counts = meta_data['emotion'].value_counts()
    percentages = meta_data['emotion'].value_counts(normalize=True) * 100
    distribution = pd.DataFrame({
        'Counts': value_counts,
        'Percentage': percentages
    })
    print('Emotion Value Counts: \n', distribution)


def calculate_copy_factors_for_label(entries):
    emotions = [entry['label'] for entry in entries]
    value_counter = Counter(emotions)

    maximum_count = max(value_counter.values())
    return {emotion: round(maximum_count / count) for emotion, count in value_counter.items()}


def get_wav_duration(wav_file):
    return float(subprocess.check_output("soxi -D {0}".format(wav_file), shell=True))


def make_train_val_test_split(entries, train_split, val_split, test_split, random_state=1):
    assert train_split + val_split + test_split == 1.0

    entries_train, entries_test = train_test_split(entries, test_size=0.15, random_state=random_state, stratify=[entry['label'] for entry in entries])
    entries_train, entries_val = train_test_split(entries_train, test_size=0.15 / (1 - 0.15), random_state=random_state, stratify=[entry['label'] for entry in entries_train])

    return entries_train, entries_val, entries_test