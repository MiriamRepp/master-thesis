import argparse
import codecs
import glob
import json
import os
from pathlib import Path

from moviepy.editor import AudioFileClip
from sklearn.model_selection import train_test_split


def get_audio_duration_wav(wav_file):
    # Load the audio clip
    audio_clip = AudioFileClip(wav_file)

    # Get the audio duration
    audio_duration = audio_clip.duration

    # Close the audio clip
    audio_clip.close()

    return audio_duration


emotion_label_mapping = {
    'W': 'anger',
    'E': 'disgust',
    'T': 'sadness',
    'F': 'happiness',
    'N': 'neutral',
    'L': 'boredom',
    'A': 'fear'
}


def __process_data(data_folder: str):
    audio_file_paths = []
    durations = []
    emotions = []

    wavs = glob.glob(os.path.join(data_folder, 'wav', '*.wav'))

    for wav_file in wavs:
        duration = get_audio_duration_wav(wav_file)
        emotion = emotion_label_mapping[Path(wav_file).stem[5]]

        audio_file_paths.append(os.path.abspath(wav_file))
        durations.append(duration)
        emotions.append(emotion)

    audio_file_paths_train, audio_file_paths_test, durations_train, durations_test, emotions_train, emotions_test \
        = train_test_split(audio_file_paths, durations, emotions, test_size=0.15, random_state=1, stratify=emotions)
    audio_file_paths_train, audio_file_paths_val, durations_train, durations_val, emotions_train, emotions_val \
        = train_test_split(audio_file_paths_train, durations_train, emotions_train, test_size=0.15 / (1 - 0.15), random_state=1, stratify=emotions_train)

    write_json(audio_file_paths_train, durations_train, emotions_train, os.path.join(data_folder, 'train.json'))
    write_json(audio_file_paths_val, durations_val, emotions_val, os.path.join(data_folder, 'val.json'))
    write_json(audio_file_paths_test, durations_test, emotions_test, os.path.join(data_folder, 'test.json'))


def write_json(audio_file_paths, durations, emotions, manifest_file):
    with codecs.open(manifest_file, "w", "utf-8") as fout:
        for (audio_file_path, duration, emotion) in zip(audio_file_paths, durations, emotions):
            entry = {"audio_filepath": audio_file_path, "duration": duration, 'label': emotion}
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(data_dir):
    __process_data(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", required=True, default=None, type=str)
    args = parser.parse_args()
    main(args.data_directory)
