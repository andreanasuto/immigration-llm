# --------------------------
# LOAD NECESSARY LIBRARIES
# --------------------------
library(caret)
library(ggplot2)
library(dplyr)

# --------------------------
# DEFINE BASE DIRECTORIES
# --------------------------
# Input data directories
base_path_test <- "/Users/Andrea/Library/CloudStorage/OneDrive-TheUniversityofLiverpool/PhD/code/global-immigration-sentiment/validation/test_dataset"
base_path_train <- "/Users/Andrea/Library/CloudStorage/OneDrive-TheUniversityofLiverpool/PhD/code/global-immigration-sentiment/validation/train_dataset"

# Output directory for confusion matrices
output_dir <- "/Users/Andrea/Library/CloudStorage/OneDrive-TheUniversityofLiverpool/PhD/code/global-immigration-sentiment/output/confusion_matrices"

# Ensure output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# --------------------------
# GLOBAL SETTINGS
# --------------------------
# TRUE -> Keeps only English, Spanish, and Multilanguage datasets
# FALSE -> Includes all available languages
filter_languages <- TRUE 

# --------------------------
# FUNCTION TO EXTRACT MODEL AND DATASET FROM FILENAME
# --------------------------

extract_model_dataset <- function(filename) {
  filename <- gsub(".csv", "", filename)
  parts <- unlist(strsplit(filename, "_"))
  model <- paste(parts[1:3], collapse = "-")  # Model name from filename
  dataset_code <- parts[length(parts)-1]  # Extract dataset code
  return(data.frame(Model = model, Dataset = dataset_code, stringsAsFactors = FALSE))
}

# --------------------------
# FUNCTION TO COMPUTE AND SAVE CONFUSION MATRIX
# --------------------------

process_conf_matrix <- function(file_path, dataset_type) {
  data <- tryCatch(read.csv(file_path), error = function(e) return(NULL))
  if (is.null(data)) return(NULL)  # Skip if file cannot be read
  
  # Convert labels to factors for confusion matrix computation
  data$label <- factor(data$label)
  data$label_llama <- factor(data$label_llama, levels = levels(data$label))
  
  # Compute confusion matrix
  cm <- confusionMatrix(table(data$label_llama, data$label), mode = "everything")
  
  # Extract model and dataset info
  dataset_info <- extract_model_dataset(basename(file_path))
  
  # Save confusion matrix as a TXT file
  model_name <- gsub("-", "_", dataset_info$Model)  # Replace "-" with "_" to avoid filename issues
  output_txt <- file.path(output_dir, paste0("conf_matrix_", dataset_type, "_", model_name, ".txt"))
  sink(output_txt)  # Redirect output to file
  print(cm)  # Print confusion matrix to file
  sink()  # Stop redirecting output
  
  # Convert confusion matrix to a data frame for visualization
  cm_table <- as.data.frame(as.table(cm$table))
  colnames(cm_table) <- c("Predicted", "Actual", "Frequency")
  
  # Create confusion matrix heatmap
  plot_cm <- ggplot(cm_table, aes(x = Actual, y = Predicted, fill = Frequency)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Frequency), size = 5) +
    scale_fill_gradient(low = "white", high = "blue") +
    labs(title = paste("Confusion Matrix for", dataset_info$Model, "(", dataset_type, ")"),
         x = "Actual", y = "Predicted") +
    theme_minimal() +
    theme(panel.background = element_rect(fill = "white", color = NA),
          plot.background = element_rect(fill = "white", color = NA),
          legend.position = "right")
  
  # Save confusion matrix as PNG file
  output_png <- file.path(output_dir, paste0("conf_matrix_", dataset_type, "_", model_name, ".png"))
  ggsave(output_png, plot_cm, width = 6, height = 5, dpi = 300)
}

# --------------------------
# LOAD AND PROCESS ALL FILES
# --------------------------

# Process test dataset files
file_list_test <- list.files(path = base_path_test, pattern = "^llama-32-3B-.*\\.csv$", full.names = TRUE)
lapply(file_list_test, process_conf_matrix, dataset_type = "Test")

# Process train dataset files
file_list_train <- list.files(path = base_path_train, pattern = "^llama-32-3B-.*\\.csv$", full.names = TRUE)
lapply(file_list_train, process_conf_matrix, dataset_type = "Train")

# Print completion message
cat("Processing complete. Confusion matrices saved in:", output_dir, "\n")