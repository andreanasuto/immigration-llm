import os
import re
import pandas as pd
import time
from llama_cpp import Llama
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Dataset split selection
DATASET_SPLIT = "test"  # Change to "train" for training dataset

# Paths
GGUF_MODELS_PATH = "/n/netscratch/cga/Lab/anasuto/immigration/gguf/"
VALIDATION_RESULTS_PATH = f"/n/netscratch/cga/Lab/anasuto/immigration/validation_results/{DATASET_SPLIT}_translation_dataset"
METRICS_FILE_PATH = os.path.join(VALIDATION_RESULTS_PATH, f"validation_metrics_{DATASET_SPLIT}_translation.txt")
F1_METRICS_CSV = os.path.join(VALIDATION_RESULTS_PATH, f"f1_metrics_{DATASET_SPLIT}_translation.csv")
ACCURACY_METRICS_CSV = os.path.join(VALIDATION_RESULTS_PATH, f"accuracy_metrics_{DATASET_SPLIT}_translation.csv")

# Ensure validation results directory exists
os.makedirs(VALIDATION_RESULTS_PATH, exist_ok=True)

# List of datasets from Hugging Face
DATASETS = [
    "andreanasuto/mig-es-translation"
    #"andreanasuto/mig-en", "andreanasuto/mig-en-es", "andreanasuto/mig-es",
    #"andreanasuto/mig-multi", "andreanasuto/mig-ar", "andreanasuto/mig-de",
    #"andreanasuto/mig-fr", "andreanasuto/mig-hi", "andreanasuto/mig-hu",
    #"andreanasuto/mig-in", "andreanasuto/mig-it", "andreanasuto/mig-pt",
    #"andreanasuto/mig-tr"
]

# Load all .gguf model files in the folder
gguf_models = [os.path.join(GGUF_MODELS_PATH, f) for f in os.listdir(GGUF_MODELS_PATH) if f.endswith(".gguf")]

def extract_classification_label(response):
    match = re.search(r'Answer =.*?(\d+)', response)
    if match:
        category_map = {1: "pro-immigration", 2: "anti-immigration", 3: "neutral", 4: "unrelated"}
        return category_map.get(int(match.group(1)), "failed")
    return "failed"

def classify_tweet(llm, tweet_text):
    prompt = f"""
    <|eot_id|><|start_header_id|>user<|end_header_id|>The next text is a tweet probably about migration:
    {tweet_text}
    Analyze carefully the tweet and assign it to the most relevant category among those listed below. 
    Do not explain your answer and return only a number. 
    Category numbers: 1 = 'pro-immigration'; 2 = 'anti-immigration'; 3 = 'neutral'; 4 = 'unrelated'.<|eot_id|><|start_header_id|>assistant<|end_header_id|> Answer ="""
    
    output = llm(prompt, max_tokens=3, echo=True, temperature=0.01)
    answer = output['choices'][0]['text']
    return extract_classification_label(answer)

def append_to_metrics_file(content):
    with open(METRICS_FILE_PATH, "a") as f:
        f.write(content + "\n\n")

def save_classification_metrics(model_name, dataset_name, df, tweets_per_second):
    true_labels = df['label'].str.strip().fillna("unrelated")
    predicted_labels = df['label_llama'].str.strip().fillna("unrelated")

    accuracy = accuracy_score(true_labels, predicted_labels)
    classification_metrics = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)
    f1_weighted_macro = classification_metrics["weighted avg"]["f1-score"]

    metrics_text = f"Model: {model_name} | Dataset: {dataset_name}\n"
    metrics_text += f"Tweets per Second: {tweets_per_second:.2f}\n"
    metrics_text += f"Accuracy: {accuracy:.4f}\n"
    metrics_text += f"F1 Weighted Macro Score: {f1_weighted_macro:.4f}\n"
    
    append_to_metrics_file(metrics_text)

    # Append to F1 and Accuracy CSVs progressively
    append_to_csv(F1_METRICS_CSV, ["Model", "Dataset", "F1 Score", "Size"],
                  [model_name, dataset_name, f1_weighted_macro, len(df)])
    
    append_to_csv(ACCURACY_METRICS_CSV, ["Model", "Dataset", "Accuracy", "Size"],
                  [model_name, dataset_name, accuracy, len(df)])

def append_to_csv(filepath, columns, data):
    """Appends a single row of data to a CSV file, creating the file with a header if it does not exist."""
    file_exists = os.path.exists(filepath)
    df = pd.DataFrame([data], columns=columns)
    df.to_csv(filepath, mode='a', header=not file_exists, index=False)

# Start processing models and datasets
for model_path in gguf_models:
    model_name = os.path.basename(model_path).replace(".gguf", "")
    llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1, verbose=True)
    
    for dataset_name in DATASETS:
        print(f"\nProcessing dataset {dataset_name} with model {model_name}...")
        dataset = load_dataset(dataset_name, split=DATASET_SPLIT)
        df = pd.DataFrame(dataset)
        print(f"Length dataset {len(df)}\n")
        
        start_time = time.time()
        df['label_llama'] = [classify_tweet(llm, text) for text in tqdm(df['translation'], desc=f"Classifying {dataset_name}")]
        end_time = time.time()
        tweets_per_second = len(df) / (end_time - start_time) if (end_time - start_time) > 0 else 0.0

        # Save classification results progressively
        output_filename = f"{model_name}_{dataset_name.replace('/', '_')}_{DATASET_SPLIT}.csv"
        output_file = os.path.join(VALIDATION_RESULTS_PATH, output_filename)
        file_exists = os.path.exists(output_file)

        df.to_csv(output_file, mode='a', header=not file_exists, index=False)

        save_classification_metrics(model_name, dataset_name, df, tweets_per_second)
        print(f"Results appended in {output_file}\n")

print(f"F1 and Accuracy metrics saved progressively for {DATASET_SPLIT} dataset with translation.")