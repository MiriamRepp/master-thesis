import argparse
import json
import matplotlib.colors as mcolors

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def colors_to_float(color_strings):
    int_colors = list(map(int, color_strings))
    return np.array(int_colors) / 255.0


def read_dataset_jsons(file_names):
    dfs = [read_dataset_json(file_name) for file_name in file_names]
    return pd.concat(dfs)


def read_dataset_json(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Convert the list of dictionaries into a DataFrame
    return pd.DataFrame(data)


def draw_distribution_plot(dataset_name, emotions, counts, min_color, max_color):
    total_count = sum(counts)
    percentages = 100 * np.array(counts) / total_count

    # Function to create a green gradient colormap
    def create_gradient_colormap(N=256):
        """ Create a colormap that goes from min_color to max_color in green tones. """
        new_colors = np.zeros((N, 3))  # RGBA format
        for i in range(N):
            ratio = i / (N - 1)
            # Interpolating between min_color and max_color
            new_colors[i, :] = (1 - ratio) * np.array(min_color) + ratio * np.array(max_color)
        return mcolors.ListedColormap(new_colors)

    gradient_cmap = create_gradient_colormap()
    norm = mcolors.Normalize(vmin=min(percentages), vmax=max(percentages))

    # Creating the bar chart with the green gradient colormap
    fig, ax = plt.subplots()

    # Plotting counts with the green gradient colormap and percentages on top of each bar
    for emotion, count, percentage in zip(emotions, counts, percentages):
        bar_color = gradient_cmap(norm(percentage))
        bar = ax.bar(emotion, count, color=bar_color, alpha=.8)
        ax.text(bar[0].get_x() + bar[0].get_width() / 2, count, f'{percentage:.2f}%', ha='center', va='bottom', color='black')

    # Adding the total count to the plot, positioned inside the graph
    ax.text(0.95, 0.95, f'Total: {total_count}', ha='right', va='center',
            transform=ax.transAxes, color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Enabling grid lines
    ax.grid(True, linestyle=':')
    ax.set_axisbelow(True)

    # Setting labels and title
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Counts')
    ax.set_title(f'Label Distribution in {dataset_name} Dataset')

    # Showing the plot
    plt.show()



def draw_distributions_from_jsons(dataset_name, dataset_jsons, min_color, max_color):
    df = read_dataset_jsons(dataset_jsons)
    emotion_counts = df['label'].value_counts()

    draw_distribution_plot(dataset_name, emotion_counts.index.tolist(), emotion_counts.values.tolist(), colors_to_float(min_color), colors_to_float(max_color))

    distribution = pd.DataFrame({
        'Counts': emotion_counts,
    })
    print('Emotion Value Counts: \n', distribution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True, default=None, type=str, help="dataset name")
    parser.add_argument("--min-color", default='54,16,0', type=str, help="min color (r,g,b)")
    parser.add_argument("--max-color", default='255,86,0', type=str, help="max color (r,g,b)")
    parser.add_argument("--dataset-jsons", nargs="*", default=[], type=str, help="dataset jsons (e.g. train, test, val)")
    args = parser.parse_args()

    draw_distributions_from_jsons(args.dataset_name, args.dataset_jsons, args.min_color.split(','), args.max_color.split(','))
