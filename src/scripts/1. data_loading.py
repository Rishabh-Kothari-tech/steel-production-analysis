import pandas as pd
import os

# Set working directory to project root
project_root = r"C:\Users\risha\steel_production_analysis"
os.chdir(project_root)

def load_steel_data(file_path):
    df = pd.read_csv(file_path, engine="python")
    return df

# Debug: List files in data folder
print("Files in data folder:", os.listdir("data"))
print("Files in data/raw folder:", os.listdir("data/raw"))

# Load training dataset
train_file = "data/raw/normalized_train_data.csv"
print(f"\nLoading {train_file}...")
train_df = load_steel_data(train_file)
print("Training dataset loaded successfully!")
print(train_df.head(), "\nShape:", train_df.shape)

# Load test dataset
test_file = "data/raw/normalized_test_data.csv"
print(f"\nLoading {test_file}...")
test_df = load_steel_data(test_file)
print("Test dataset loaded successfully!")
print(test_df.head(), "\nShape:", test_df.shape)