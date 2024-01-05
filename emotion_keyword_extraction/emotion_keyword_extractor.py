import argparse
import json

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# HYPER PARAMETERS
MIN_WORD_FREQUENCY_PER_EMOTION = .01

# Download the stopwords and wordnet resources from NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def get_unequal_tuples(lst):
    return [(x, y) for x in lst for y in lst if x != y]


# Function to clean and preprocess text data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters, punctuation, and digits
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    # Remove stopwords (common words with little meaning)
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    # Lemmatize words to their base form
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words


def count_word_frequency(series):
    # Flatten the series by concatenating all lists of words into one list
    all_words = [word for words_list in series for word in words_list]

    # Use Counter to count word frequencies
    word_counts = Counter(all_words)

    return word_counts


def sort_dict_by_values_descending(dictionary, top_n):
    sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))
    top_n_items = list(sorted_dict.items())[:top_n]
    return dict(top_n_items)


def filter_seldom_words(word_frequencies_per_emotion, samples_by_emotion):
    filtered_word_frequencies_by_emotion = {}

    for emotion, word_frequencies in word_frequencies_per_emotion.items():
        samples = samples_by_emotion[emotion]

        filtered_word_frequencies = {word: freq for word, freq in word_frequencies.items() if freq / samples >= MIN_WORD_FREQUENCY_PER_EMOTION}
        filtered_word_frequencies_by_emotion[emotion] = filtered_word_frequencies

    return filtered_word_frequencies_by_emotion


def filter_banned_words(words, banned_words):
    return [word for word in words if word not in banned_words]


def calculate_weighted_word_frequencies(filtered_word_frequencies_by_emotion, word_frequencies):
    weighted_word_frequencies_by_emotion = {}

    for emotion, emotion_word_frequencies in filtered_word_frequencies_by_emotion.items():
        weighted_word_frequencies_by_emotion[emotion] = {}

        for word, emotion_word_frequency in emotion_word_frequencies.items():
            total_word_frequency = word_frequencies[word]
            other_emotions_word_frequency = total_word_frequency - emotion_word_frequency

            score = emotion_word_frequency / other_emotions_word_frequency if other_emotions_word_frequency > 0 else emotion_word_frequency

            weighted_word_frequencies_by_emotion[emotion][word] = score

    return weighted_word_frequencies_by_emotion


def sort_word_frequencies(word_frequencies_by_emotion):
    return {emotion: dict(sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)) for emotion, word_frequencies in word_frequencies_by_emotion.items()}


def render_graphs(word_frequencies_by_emotion, max_words_per_emotion=20):
    # Create histograms for each emotion with the top 20 remaining words
    for emotion, word_frequencies in word_frequencies_by_emotion.items():
        plt.figure(figsize=(8, 6))
        top_words = list(word_frequencies.items())[:max_words_per_emotion]
        words, freqs = zip(*top_words)

        plt.bar(words, freqs)
        plt.xlabel('words')
        plt.ylabel('score')
        plt.title(f'Top {max_words_per_emotion} Highest Scoring Words for {emotion.title()} Emotion')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def legacy_evaluation(word_frequencies_per_emotion, banned_words):
    # Find the top 20 most frequent words for each emotion
    top_words_per_emotion = {}
    for emotion, word_freq in word_frequencies_per_emotion.items():
        top_words_per_emotion[emotion] = set([word for word, _ in word_freq.most_common(20)])

    # Find words that appear in the top 20 for more than one emotion
    all_combinations = get_unequal_tuples(top_words_per_emotion.values())
    intersections = [set.intersection(a, b) for (a, b) in all_combinations]
    common_words = set.union(*intersections)

    common_words = common_words.union(set(banned_words))

    # Filter out common words from the word frequencies for each emotion
    for emotion, word_freq in word_frequencies_per_emotion.items():
        word_frequencies_per_emotion[emotion] = {word: freq for word, freq in word_freq.items() if word not in common_words}

    # Create histograms for each emotion with the top 20 remaining words
    for emotion, word_freq in word_frequencies_per_emotion.items():
        plt.figure(figsize=(8, 6))
        top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])  # Get the top 20 most frequent words
        words, freqs = zip(*top_words.items())
        total_words = sum(freqs)
        normalized_freqs = [freq / total_words for freq in freqs]
        plt.bar(words, normalized_freqs)
        plt.xlabel('Words')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Top 20 Most Frequent Words for {emotion} Emotion (Excluding Common Words)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def read_json_file_as_dataframe(json_file):
    with open(json_file, 'r') as file:
        data = [json.loads(line) for line in file]

    return pd.DataFrame(data)


def main(metafile_json, words_per_emotion, banned_words):
    # Load the dataframe and display the first few rows to ensure data is correct
    dataframe = read_json_file_as_dataframe(metafile_json)

    # Apply preprocessing to the 'Utterance' column and create a new column 'processed_text'
    dataframe['pre_processed_text'] = dataframe['text'].apply(preprocess_text)
    dataframe['processed_text'] = dataframe['pre_processed_text'].apply(lambda words: filter_banned_words(words, banned_words))

    # Calculate overall word frequencies
    word_frequencies = count_word_frequency(dataframe['processed_text'])

    samples_by_emotion = dataframe.groupby('label')['label'].count()

    # Group utterances by emotion
    grouped_by_emotion = dataframe.groupby('label')['processed_text'].sum()

    # Calculate word frequencies for each emotion
    word_frequencies_per_emotion = {}
    for emotion, words in grouped_by_emotion.items():
        word_frequencies_per_emotion[emotion] = Counter(words)

    # filter words appearing in less than x percent of the samples of an emotion
    filtered_word_frequencies_by_emotion = filter_seldom_words(word_frequencies_per_emotion, samples_by_emotion)

    weighted_word_frequencies_by_emotion = calculate_weighted_word_frequencies(filtered_word_frequencies_by_emotion, word_frequencies)

    sorted_weighted_word_frequencies_by_emotion = sort_word_frequencies(weighted_word_frequencies_by_emotion)

    render_graphs(sorted_weighted_word_frequencies_by_emotion, max_words_per_emotion=words_per_emotion)

    selected_words_by_emotion = {emotion: list(word_frequencies)[:words_per_emotion] for emotion, word_frequencies in sorted_weighted_word_frequencies_by_emotion.items()}
    selected_words = [word for words in selected_words_by_emotion.values() for word in words]

    print('selected words by emotion:', selected_words_by_emotion)
    print('selected words: ', ' '.join(selected_words))
    print('selected words: ', selected_words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metafile_json", required=True, default=None, type=str)
    parser.add_argument("--words_per_emotion", required=True, default=None, type=int)
    parser.add_argument("--banned_words", nargs="*", default=[], type=str,
                        help="Words that should not be used, even though they appear a lot in the data, e.g. names.")
    args = parser.parse_args()
    main(args.metafile_json, args.words_per_emotion, args.banned_words)
