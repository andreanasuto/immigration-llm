# immigration-llm
# README 📘

## Project Overview 📂

This project is set up to help fine-tune machine learning models and run tasks on a high-performance computing (HPC) system. Below is an outline of the key parts of this repository.

### `fine-tuning` 🎯📝🔧

This folder contains scripts that help train models to perform better. It includes:

- **Creating Training and Test Datasets**: Scripts that prepare data for training and testing of the models.
- **Fine-tune Model**: Scripts that run the fine-tuning for different type of datasets and source model (eg. Llama 3.2 - 3B)  
- **Merging Model Weights**: The `merge_weights.py` script combines the weights from an existing model with the fine-tuned version to improve performance.

### `hpc` 🖥️🚀⚡

This folder has the necessary code for running model training and validation on an HPC system. We used FASRC HPC at Harvard thus the following code is compatible with that.
It is divided into two sections:

#### `classification` 📊📑🔍

- Contains SLURM (`.sbatch`) scripts for submitting jobs to the HPC system.
- Includes Python scripts that handle classification tasks on the cluster.

#### `validation` ✅🔬📉

- Similar to the classification section but focused on testing and verifying model accuracy.

## How to Use 🔄

1. **Fine-Tuning Process**

   - Generate the training and test datasets and upload them to Hugging Face.
   - Fine-tune the model to improve its performance.
   - Use `merge_weights.py` to combine the improved weights with the original model.

2. **Running Tasks on the FASRC System**

   - Submit classification jobs using SLURM `.sbatch` scripts.
   - Make sure the Python scripts are set up correctly for smooth execution on the FASRC cluster.

For more details, check the comments and explanations inside each script. 📖