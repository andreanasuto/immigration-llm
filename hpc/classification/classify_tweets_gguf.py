import os
import sys
import torch
from llama_cpp import Llama
from tqdm import tqdm
import re
import pandas as pd
from datetime import datetime

FILES_COMPLETED_LOG = "/n/netscratch/cga/Lab/anasuto/immigration/logs/files_complete_log.txt"
MODEL_FOLDER = "/n/netscratch/cga/Lab/anasuto/immigration/gguf/llama-32-3B-en-es-Q4_K_M.gguf" # change model here

def extract_classification_label(response):
    """
    Extracts the classification label based on the category number in the response.
    
    Args:
        response (str): The response containing the category number.
    
    Returns:
        str: The classification label corresponding to the category number.
    """
    match = re.search(r'Answer =.*?(\d+)', response)  # Extract category number
    if match:
        category_number = int(match.group(1))  # Convert to integer
        category_map = {
            1: "pro-immigration",
            2: "anti-immigration",
            3: "neutral",
            4: "unrelated"
        }
        return category_map.get(category_number)  # return label 
    return "failed"  # default return if no match is found
    
# Function to classify tweets
def classify_tweet(tweet_text):
    prompt = f"""
    <|eot_id|><|start_header_id|>user<|end_header_id|>TThe next text is a tweet probably about migration:
    {tweet_text}
    Analyze carefully the tweet and assign it to the most relevant category among those listed below. 
    Do not explain your answer and return only a number. 
    Category numbers: 1 = 'pro-immigration'; 2 = 'anti-immigration'; 3 = 'neutral'; 4 = 'unrelated'.<|eot_id|><|start_header_id|>assistant<|end_header_id|> Answer ="""
    output = llm(
        prompt,
        max_tokens=3,  # Generate up to 5 tokens
        echo=True,  # Echo the prompt back in the output
        temperature = 0.01
        )
    answer = output['choices'][0]['text']
    classification = extract_classification_label(answer)
    return classification

def extract_year_month_day_hour_from_filename(file_path):
    basename = os.path.basename(file_path)
    match = re.search(r"(\d{4})_(\d{1,2})_(\d{1,2})_(\d{1,2})", basename)
    if match:
        return match.group(1), match.group(2), match.group(3), match.group(4)
    raise ValueError(f"Unable to extract year, month, day, and hour from file name: {basename}")

# Save checkpoint function
def save_checkpoint(checkpoint_file, message_id):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        new_entry = pd.DataFrame([[message_id, current_time]], columns=['message_id', 'processed_at'])
        checkpoint_df = pd.concat([checkpoint_df, new_entry], ignore_index=True)
    else:
        checkpoint_df = pd.DataFrame([[message_id, current_time]], columns=['message_id', 'processed_at'])
    checkpoint_df.to_csv(checkpoint_file, index=False)

# Main classification and saving function
def classify_and_save(df, checkpoint_file, output_file, csv_gz_file):
    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        last_processed_id = str(checkpoint_df['message_id'].iloc[-1]) if not checkpoint_df.empty else None
    else:
        last_processed_id = None
    df['message_id'] = df['message_id'].astype(str)
    start_idx = df.index[df['message_id'] == last_processed_id].tolist()[0] + 1 if last_processed_id in df['message_id'].values else 0
    
    #batch = []  # Accumulate rows to save in batches
    #batch_size = 50
    
    for idx in range(start_idx, len(df)):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # create a timestamp for testing
        row = df.iloc[idx]
        message_id = row['message_id']
        df.at[idx, 'label_llama'] = classify_tweet(row['text'])
        #df.at[idx, 'processed_at'] = current_time # save timestamp when processed for testing
        save_checkpoint(checkpoint_file, message_id)
        df.iloc[[idx]].to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        #batch.append(df.iloc[idx])  # Add the row to the batch
        
        # Save batch when it reaches the specified size
        #if len(batch) == batch_size:
        #    pd.DataFrame(batch).to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        #    batch.clear()  # Clear the batch after saving
    
    # Save any remaining rows in the batch
    #if batch:
    #    pd.DataFrame(batch).to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    
    # Check if the dataset was fully processed
    if idx == len(df) - 1:
        # Log the file as completed
        with open(FILES_COMPLETED_LOG, "a") as log_file:
            log_file.write(f"{csv_gz_file}\n")

if len(sys.argv) < 2:
    print("Usage: python gpu_process_tweets.py <csv_gz_file>")
    sys.exit(1)

csv_gz_file = sys.argv[1]
log_dir = "/n/netscratch/cga/Lab/anasuto/immigration/logs/" # or "/n/home03/anasuto/geotweets/logs/"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "failed_files_read_log.csv")

try:
    df = pd.read_csv(csv_gz_file, sep='\t', dtype='unicode', index_col=None, low_memory=True, compression='gzip')
except Exception as e:
    log_df = pd.DataFrame([[csv_gz_file, str(e)]], columns=['file', 'error'])
    log_df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
    sys.exit(1)

# Drop unnecessary columns
df = df.drop(['geom', 'source', 'data_source', 'GPS', 'status', 'photo_url'], axis=1)
df = df.dropna(subset=['text', 'message_id', 'date'])
df['message_id'] = df['message_id'].astype(str)
numeric_cols = ['retweets', 'tweet_favorites', 'quoted_status_id', 'user_id', 'followers', 'friends', 
                'user_favorites', 'latitude', 'longitude', 'spatialerror']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.fillna(value={'retweets': 0, 'tweet_favorites': 0, 'quoted_status_id': 0, 'user_id': 0,
                 'followers': 0, 'friends': 0, 'user_favorites': 0,
                 'latitude': 0, 'longitude': 0, 'spatialerror': 0}, inplace=True)

# File setup
save_path_geotweets = "/n/netscratch/cga/Lab/anasuto/immigration/geotweets" # or /n/home03/anasuto/geotweets"
year, month, day, hour = extract_year_month_day_hour_from_filename(csv_gz_file)
output_dir = os.path.join(save_path_geotweets, year, month, day)
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{year}_{month}_{day}_{hour}.csv")
checkpoint_file = os.path.join(log_dir, "checkpoints", f"checkpoint_{year}_{month}_{day}_{hour}.csv")
os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)


llm = Llama(
    model_path=MODEL_FOLDER,
    n_ctx = 2048,
    n_gpu_layers = -1,
    verbose=True
)

# Start classification
classify_and_save(df, checkpoint_file, output_file, csv_gz_file)