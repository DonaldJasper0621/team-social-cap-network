import pandas as pd

# Preview a CSV file.
data_file = 'nba_2016_2017_100.csv'

# Load the dataset into a DataFrame
try:
    df = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: The file {data_file} was not found. Please ensure the file is in the current directory.")
    exit()

# Display a preview of the data (first 5 rows)
print("Data Preview (df.head()):")
print(df.head())
