import os
import sys
import torch
from llama_cpp import Llama
from tqdm import tqdm
import re
import pandas as pd
from datetime import datetime

FILES_COMPLETED_LOG = "/n/netscratch/cga/Lab/anasuto/immigration/logs_usa/files_complete_log.txt"
MODEL_FOLDER = "/n/netscratch/cga/Lab/anasuto/immigration/gguf/llama-32-3B-en-es-full.gguf" # change model here

# Define instruction
instruction = f"""The next text is a tweet probably about migration"""

# Define instructionSystem
instructionSystem = """You are an expert on migration. Answer the question truthfully."""  

def extract_classification_label(response):
    match = re.search(r'Answer =.*?(\d+)', response)
    if match:
        category_number = int(match.group(1))
        category_map = {
            1: "pro-immigration",
            2: "anti-immigration",
            3: "neutral",
            4: "unrelated"
        }
        return category_map.get(category_number)
    return "failed"

def classify_tweet(tweet_text):
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> {instructionSystem} <|eot_id|><|start_header_id|>user<|end_header_id|>{instruction}:
    "{tweet_text}" 
    Analyze carefully the tweet and assign it to the most relevant category among those listed below.
    Do not explain your answer and return only a number.
    Category numbers: 1 = 'pro-immigration'; 2 = 'anti-immigration'; 3 = 'neutral'; 4 = 'unrelated'.<|eot_id|><|start_header_id|>assistant<|end_header_id|>Answer = """
    output = llm(prompt, max_tokens=3, echo=True, temperature=0.01)
    answer = output['choices'][0]['text']
    classification = extract_classification_label(answer)
    return classification

def extract_year_month_day_hour_slice_from_filename(file_path):
    basename = os.path.basename(file_path)
    match = re.search(r"(\d{4})_(\d{1,2})_(\d{1,2})_(\d{1,2})-tl_\d{4}_(\d+)_tabblock20\.parquet", basename)
    if match:
        return match.group(1), match.group(2), match.group(3), match.group(4), match.group(5)
    raise ValueError(f"Unable to extract year, month, day, hour, and slice from file name: {basename}")
    
def save_checkpoint(checkpoint_file, message_id):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        new_entry = pd.DataFrame([[message_id, current_time]], columns=['message_id', 'processed_at'])
        checkpoint_df = pd.concat([checkpoint_df, new_entry], ignore_index=True)
    else:
        checkpoint_df = pd.DataFrame([[message_id, current_time]], columns=['message_id', 'processed_at'])
    checkpoint_df.to_csv(checkpoint_file, index=False)

def classify_and_save(df, checkpoint_file, output_file, parquet_file):
    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        last_processed_id = str(checkpoint_df['message_id'].iloc[-1]) if not checkpoint_df.empty else None
    else:
        last_processed_id = None
    df['message_id'] = df['message_id'].astype(str)
    start_idx = df.index[df['message_id'] == last_processed_id].tolist()[0] + 1 if last_processed_id in df['message_id'].values else 0
    
    debug_file = "/n/netscratch/cga/Lab/anasuto/immigration/logs_usa/debug_classifications.txt"

    for idx in range(start_idx, len(df)):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = df.iloc[idx]
        message_id = row['message_id']
        df.loc[idx, 'label_llama'] = classify_tweet(row['text'])
        classification = classify_tweet(row['text'])
        save_checkpoint(checkpoint_file, message_id)
        df.iloc[[idx]].to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        
        # Append results to a debug file
        with open(debug_file, "a") as f:
            f.write(f"Processed {idx}: Message ID {message_id} -> {classification}\n")
    
    if idx == len(df) - 1:
        with open(FILES_COMPLETED_LOG, "a") as log_file:
            log_file.write(f"{parquet_file}\n")

if len(sys.argv) < 2:
    print("Usage: python gpu_process_tweets.py <parquet_file>")
    sys.exit(1)

parquet_file = sys.argv[1]
log_dir = "/n/netscratch/cga/Lab/anasuto/immigration/logs_usa/"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "failed_files_read_log.csv")

try:
    df = pd.read_parquet(parquet_file)
    
    # Ensure df is a deep copy to allow modifications
    df = pd.DataFrame(df)
    
except Exception as e:
    log_df = pd.DataFrame([[parquet_file, str(e)]], columns=['file', 'error'])
    log_df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
    sys.exit(1)

# Select only required columns
expected_columns = ['message_id', 'date', 'text', 'tweet_lang', 'retweets', 'tweet_favorites', 
                    'GEOID20', 'UR20', 'UACE20', 'UATYPE20', 'latitude', 'longitude']

# Ensure missing columns don't break the script
df = df[[col for col in expected_columns if col in df.columns]]

# Drop rows with missing essential values
df = df.dropna(subset=['text', 'message_id', 'date', 'GEOID20'])

# Fill missing values for categorical columns
df.fillna(value={'UACE20': 'R', 'UATYPE20': 'R'}, inplace=True)

# Convert all columns to string to avoid type mismatches
df = df.astype(str)

df = df.reset_index(drop=True)

year, month, day, hour, slice_id = extract_year_month_day_hour_slice_from_filename(parquet_file)
save_path_geotweets = "/n/netscratch/cga/Lab/anasuto/immigration/geotweets_usa"
output_dir = os.path.join(save_path_geotweets, year, month, day)
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{year}_{month}_{day}_{hour}_{slice_id}.csv")
checkpoint_file = os.path.join(log_dir, "checkpoints", f"checkpoint_{year}_{month}_{day}_{hour}_{slice_id}.csv")
os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

llm = Llama(model_path=MODEL_FOLDER, n_ctx=2048, n_gpu_layers=-1, verbose=True)

classify_and_save(df, checkpoint_file, output_file, parquet_file)