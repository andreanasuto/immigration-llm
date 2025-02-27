# load necessary libraries
library(caret)
library(ggplot2)
library(dplyr)
library(tidyr)

# define base directories
base_path_test <- "/Users/Andrea/Library/CloudStorage/OneDrive-TheUniversityofLiverpool/PhD/code/global-immigration-sentiment/validation/test_dataset"
base_path_train <- "/Users/Andrea/Library/CloudStorage/OneDrive-TheUniversityofLiverpool/PhD/code/global-immigration-sentiment/validation/train_dataset"
output_dir <- "/Users/Andrea/Library/CloudStorage/OneDrive-TheUniversityofLiverpool/PhD/code/global-immigration-sentiment/output/model_performances"

# Defines which model type to include:
# "ALL" -> Includes all models
# "F16" -> Excludes models containing "-Q4-K-M" (keeps only floating point 16-bit models)
# "4bit" -> Includes only models containing "-Q4-K-M" (keeps only 4-bit quantized models)
model_type <- "F16" 

# Language filter:
# TRUE -> Keeps only English, Spanish, and Multilanguage datasets
# FALSE -> Includes all available languages
dataset <- "Train"

# set language filter (TRUE to exclude all except English, Spanish, and Multilanguage)
filter_languages <- FALSE 

# mapping language codes to full names
language_map <- list(
  "mig-ar" = "Arabic",
  "mig-es" = "Spanish",
  "mig-en" = "English",
  "mig-pt" = "Portuguese",
  "mig-fr" = "French",
  "mig-it" = "Italian",
  "mig-de" = "German",
  "mig-hi" = "Hindi",
  "mig-en-es" = "Eng-Spa",
  "mig-multi" = "Multi",
  "mig-hu" = "Hungarian",
  "mig-in" = "Indonesia",
  "mig-es-translation" = "Spanish Translation",
  "mig-tr" = "Turkish"
)

# function to extract model and dataset from filename
extract_model_dataset <- function(filename) {
  filename <- gsub(".csv", "", filename) 
  parts <- unlist(strsplit(filename, "_"))
  model <- paste(parts[1:3], collapse = "-") 
  dataset_code <- parts[length(parts)-1] 
  dataset <- ifelse(dataset_code %in% names(language_map), language_map[[dataset_code]], dataset_code) 
  return(data.frame(Model = model, Dataset = dataset, stringsAsFactors = FALSE))
}

# function to process csv files and extract accuracy confidence intervals
process_conf_matrix <- function(file_path) {
  data <- tryCatch(read.csv(file_path), error = function(e) return(NULL))
  if (is.null(data)) return(NULL)
  
  data$label <- factor(data$label)
  data$label_llama <- factor(data$label_llama, levels = levels(data$label))
  
  cm <- confusionMatrix(table(data$label_llama, data$label), mode = "everything")
  acc_CI <- c(cm$overall["AccuracyLower"], cm$overall["AccuracyUpper"])
  
  dataset_info <- extract_model_dataset(basename(file_path))
  return(data.frame(Model = dataset_info$Model, Dataset = dataset_info$Dataset, 
                    Accuracy_Lower = acc_CI[1], Accuracy_Upper = acc_CI[2], stringsAsFactors = FALSE))
}

# load data based on dataset filter
accuracy_data <- data.frame()

if (dataset == "Test" || dataset == "both") {
  file_list_test <- list.files(path = base_path_test, pattern = "^llama-32-3B-.*\\.csv$", full.names = TRUE)
  accuracy_data_test <- do.call(rbind, lapply(file_list_test, process_conf_matrix))
  accuracy_data <- rbind(accuracy_data, accuracy_data_test)
}

if (dataset == "Train" || dataset == "both") {
  file_list_train <- list.files(path = base_path_train, pattern = "^llama-32-3B-.*\\.csv$", full.names = TRUE)
  accuracy_data_train <- do.call(rbind, lapply(file_list_train, process_conf_matrix))
  accuracy_data <- rbind(accuracy_data, accuracy_data_train)
}

# apply model type filter
if (model_type == "F16") {
  accuracy_data <- accuracy_data %>% filter(!grepl("Q4-K-M", Model))
} else if (model_type == "4bit") {
  accuracy_data <- accuracy_data %>% filter(grepl("Q4-K-M", Model))
}

# remove models with "full" in their name before renaming
accuracy_data <- accuracy_data %>% filter(!grepl("full", Model)) 

accuracy_data <- as.data.frame(accuracy_data, stringsAsFactors = FALSE)

# ensure unique values for pivot_wider
accuracy_data <- accuracy_data %>%
  group_by(Model, Dataset) %>%
  summarise(Accuracy_Lower = mean(Accuracy_Lower, na.rm = TRUE),
            Accuracy_Upper = mean(Accuracy_Upper, na.rm = TRUE), .groups = "drop")

# rename models after filtering
model_map <- function(model) {
  if (grepl("llama-32-3B-en-es", model)) {
    return("English-Spanish")
  } else if (grepl("llama-32-3B-multi", model)) {
    return("Multilanguage")
  } else if (grepl("llama-32-3B-es", model)) {
    return("Spanish")
  } else if (grepl("llama-32-3B-en", model)) {
    return("English")
  } else {
    return(model)
  }
}

accuracy_data$Model <- sapply(accuracy_data$Model, model_map)

# apply language filter
if (filter_languages) {
  allowed_languages <- c("English", "Spanish", "Multi", "Eng-Spa")
  accuracy_data <- accuracy_data %>% filter(Dataset %in% allowed_languages)
}

accuracy_table <- accuracy_data %>%
  pivot_wider(names_from = Dataset, values_from = c(Accuracy_Lower, Accuracy_Upper))

# generate dynamic output filename
filter_status <- ifelse(filter_languages, "main_languages", "all_languages")
output_filename <- paste0("accuracy_", dataset, "_", model_type, "_", filter_status, ".csv")
output_path_csv <- file.path(output_dir, output_filename)

# dynamic title for the chart
chart_title <- paste("Accuracy Confidence Intervals for", dataset, "Dataset")

# function to create accuracy confidence interval plots
plot_accuracy_intervals <- function(data, title) {
  ggplot(data, aes(x = Dataset, ymin = Accuracy_Lower, ymax = Accuracy_Upper, color = Model, fill = Model)) +
    geom_errorbar(width = 0.3, position = position_dodge(width = 0.6)) +
    geom_point(aes(y = (Accuracy_Lower + Accuracy_Upper) / 2), size = 3, position = position_dodge(width = 0.6)) +
    scale_x_discrete(labels = function(x) ifelse(x %in% names(language_map), language_map[[x]], x)) +
    labs(title = title,
         x = "Language",
         y = "Accuracy Confidence Interval",
         color = "Model",
         fill = "Model") +
    theme_minimal() +
    theme(legend.position = "bottom",
          panel.background = element_rect(fill = "white", color = NA),  # Ensure white panel background
          plot.background = element_rect(fill = "white", color = NA),   # Ensure white plot background
          legend.title = element_text(size = 12, face = "bold"),
          legend.text = element_text(size = 10),
          axis.text.x = element_text(angle = 45, hjust = 1)) # rotate x-axis labels
}

# save results
write.csv(accuracy_table, output_path_csv, row.names = FALSE)

# create and display the plot
plot <- plot_accuracy_intervals(accuracy_data, chart_title)
print(plot)  # ensure plot is displayed in RStudio

# save the plot as PNG (300 dpi)
output_png <- gsub(".csv", ".png", output_path_csv)
ggsave(output_png, plot, width = 10, height = 6, dpi = 300) # Uncomment this line to save

# print completion message
cat("Processing complete. Results saved to:", output_path_csv, "\n")