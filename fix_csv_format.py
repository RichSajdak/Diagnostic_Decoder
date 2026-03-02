import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('test_hex_csv.csv')

# First, get all the timestamps in order
timestamps = df['timestamp_ms'].dropna().tolist()

# Get the DDT message data (skipping the header and timestamp-only rows)
data_rows = []
with open('test_hex_csv.csv', 'r') as f:
    lines = f.readlines()
    header = lines[0].strip()
    for line in lines[1:]:
        if ',' in line and any(x in line for x in ['62', 'E#3', 'null']):
            parts = line.strip().split(',')
            # Ensure each row has the right number of columns
            while len(parts) < len(header.split(',')):
                parts.append('')
            data_rows.append(parts)

# Create new DataFrame with correct alignment
new_data = []
for i, row in enumerate(data_rows):
    if i < len(timestamps):  # Make sure we have a timestamp
        new_row = [timestamps[i]] + row[1:]  # Combine timestamp with DDT data
        new_data.append(new_row)

# Convert to DataFrame using original header
columns = header.split(',')
new_df = pd.DataFrame(new_data, columns=columns)

# Save the reformatted CSV
new_df.to_csv('test_hex_csv_reformatted.csv', index=False)
