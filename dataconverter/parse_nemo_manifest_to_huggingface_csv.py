import json
import csv
import os


# Function to read a JSON file and extract data
def read_json_and_extract(input_filename):
    data = []
    with open(input_filename, 'r') as json_file:
        for line in json_file:
            entry = json.loads(line)
            file_name = str(os.path.basename(entry['audio_filepath']))
            text = entry['text'].upper()
            data.append((file_name, text))
    return data


# Function to write data to a CSV file
def write_data_to_csv(output_filename, data):
    with open(output_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['file_name', 'text'])
        writer.writerows(data)


# Input JSON file
input_filename = 'train.json'

# Output CSV file
output_filename = 'train.csv'

# Read JSON and extract data
data = read_json_and_extract(input_filename)

# Write data to CSV
write_data_to_csv(output_filename, data)

print(f'Data has been successfully converted to {output_filename}')
