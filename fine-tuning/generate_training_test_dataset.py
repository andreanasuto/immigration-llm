# generates training-set for fine tuning and testing

import json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("/n/netscratch/cga/Lab/anasuto/immigration/tests/data/tweet_1000_classified_es.csv", 
                 sep=',', na_filter=True, dtype=str).dropna()
                 
# take a 1000 sample to build the FT model in English on 1000 tweets
#df = df.sample(n=1000, random_state=123)

# Define instruction
instruction = f"""The next text is a tweet probably about migration"""

# Define categories and mapping
categories = ["pro-immigration", "anti-immigration", "neutral", "unrelated"] 
catcodes = [1, 2, 3, 4]
catDict = dict(zip(categories, catcodes))

# Map category labels to numerical codes
df['catCode'] = df['label'].map(catDict)

# Generate hint text
hint = "; ".join(f"{v} = '{k}'" for k, v in catDict.items())

# Option 1 - do not stratify by language - use this for mono-lingual dataset

# Split dataset into train and test sets
train_set, test_set = train_test_split(df, test_size=0.25, stratify=df['catCode'], random_state=123)

############################

# Option 2 - stratify by language

# Perform stratified train-test split per language
#train_list = []
#test_list = []

#for lang in df['tweet_lang'].unique():
#    df_lang = df[df['tweet_lang'] == lang]
#    train_subset, test_subset = train_test_split(df_lang, test_size=0.25, stratify=df_lang['catCode'], random_state=123)
#    train_list.append(train_subset)
#    test_list.append(test_subset)

# Combine stratified samples
#train_set = pd.concat(train_list)
#test_set = pd.concat(test_list)

############################

train = train_set.copy()
test = test_set.copy()

# Clean text columns
for dataset in [train, test]:
    dataset.loc[:, 'text'] = dataset['text'].str.replace(r'\n+', ' ', regex=True).str.replace(r'\r+', ' ', regex=True).str.strip()

# Define functions to create formatted dataset
def create_trainset(tweet, answer):
    text = f"""<s>[INST] {instruction}:"{tweet}"\nAnalyze carefully the tweet and assign it to the most relevant category among those listed below. Do not explain your answer and return only a number.\nCategory numbers: {hint}[/INST]. Answer = [{answer}]. </s>"""
    return text

def create_testset(tweet, answer):
    text = f"""<s>[INST] {instruction}:"{tweet}"\nAnalyze carefully the tweet and assign it to the most relevant category among those listed below. Do not explain your answer and return only a number.\nCategory numbers: {hint}[/INST]. Answer = </s>"""
    return text

# Apply transformations
transformed_train = train.apply(lambda row: create_trainset(row['text'], row['catCode']), axis=1).to_frame(name='text')
transformed_test = test.apply(lambda row: create_testset(row['text'], row['catCode']), axis=1).to_frame(name='text')

# Add additional columns
for transformed_df, original_df in [(transformed_train, train), (transformed_test, test)]:
    transformed_df['newId'] = original_df['message_id']
    transformed_df['message_id'] = original_df['message_id']
    transformed_df['label'] = original_df['label']
    transformed_df['catCode'] = original_df['catCode']
    transformed_df['plaintext'] = original_df['text']
    transformed_df['lang'] = original_df['tweet_lang']  # Add language column

# Convert pandas DataFrame to Hugging Face Dataset
dataset_train = Dataset.from_pandas(transformed_train)
dataset_test = Dataset.from_pandas(transformed_test)

dataset_dict = DatasetDict({
    "train": dataset_train,
    "test": dataset_test
})

# Authenticate Hugging Face API
from huggingface_hub import HfApi, HfFolder

# Your Hugging Face token
hf_token = "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"

# Save token
HfFolder.save_token(hf_token)

# Authenticate and push dataset to Hugging Face Hub
api = HfApi()
api.whoami()  # Check authentication
dataset_dict.push_to_hub("andreanasuto/mig-es") # change name based on the source dataset