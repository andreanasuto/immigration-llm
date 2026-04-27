language_map <- c(
  "ar" = "Arabic",
  "es" = "Spanish",
  "pt" = "Portuguese",
  "fr" = "French",
  "it" = "Italian",
  "de" = "German",
  "hi" = "Hindi",
  "hu" = "Hungarian",
  "in" = "Indonesian",
  "tr" = "Turkish",
  "pl" = "Polish",
  "ko" = "Korean",
  "en" = "English"
)

lang_share_lookup <- tibble::tribble(
  ~Language,      ~LangShare,
  "English",       89.70,
  "Unknown",        8.38,
  "German",         0.17,
  "French",         0.16,
  "Swedish",        0.15,
  "Chinese",        0.13,
  "Spanish",        0.13,
  "Russian",        0.13,
  "Dutch",          0.12,
  "Italian",        0.11,
  "Japanese",       0.10,
  "Polish",         0.09,
  "Portuguese",     0.09,
  "Vietnamese",     0.08,
  "Ukrainian",      0.07,
  "Korean",         0.06,
  "Catalan",        0.04,
  "Serbian",        0.04,
  "Indonesian",     0.03,
  "Czech",          0.03,
  "Finnish",        0.03,
  "Hungarian",      0.03,
  "Norwegian",      0.03,
  "Romanian",       0.03,
  "Bulgarian",      0.02,
  "Danish",         0.02,
  "Slovene",        0.01,
  "Croatian",       0.01
)

get_language <- function(file_path, df) {
  if (str_detect(file_path, "mig-multi")) {
    return(df$lang[1])
  }

  filename <- basename(file_path)
  lang_from_name <- str_match(filename, "_([a-z]{2})\\.csv$|_([a-z]{2})_")[, 2]

  if (!is.na(lang_from_name)) {
    return(lang_from_name)
  }

  df$tweet_lang[1]
}

extract_model_dataset <- function(filename) {
  base <- basename(gsub("\\.csv$", "", filename))
  parts <- unlist(strsplit(base, "_"))
  stop_keywords <- grep("^K$|^mig-|^test$", parts)

  if (length(stop_keywords) > 0) {
    model <- paste(parts[1:(min(stop_keywords) - 1)], collapse = "_")
  } else {
    model <- base
  }

  model_map <- c(
    "llama32-3B-mig-en-es-Q4" = "English-Spanish",
    "llama32-3B-mig-en-Q4" = "English",
    "llama32-3B-mig-es-Q4" = "Spanish",
    "llama32-3B-mig-multi-Q4" = "Multilanguage"
  )

  if (model %in% names(model_map)) {
    return(model_map[model])
  }

  model
}

lang_code_to_name <- function(code) {
  ifelse(code %in% names(language_map), language_map[[code]], code)
}

is_mig_multi_or_translation <- function(file_path, path_translation_test, path_translation_train) {
  path <- tolower(normalizePath(file_path))
  translation_test <- tolower(normalizePath(path_translation_test))
  translation_train <- tolower(normalizePath(path_translation_train))

  str_detect(path, "mig-multi") |
    str_starts(path, translation_test) |
    str_starts(path, translation_train)
}

process_dataset <- function(file_path, split_type, path_translation_test, path_translation_train) {
  if (str_detect(tolower(file_path), "full")) {
    return(NULL)
  }

  if (!grepl("Q4", basename(file_path))) {
    return(NULL)
  }

  file_extension <- tools::file_ext(file_path)

  if (file_extension == "csv") {
    df <- read_csv(file_path, col_types = cols(.default = "c"))
  } else if (file_extension %in% c("xlsx", "xls")) {
    df <- read_excel(file_path)
  } else {
    return(NULL)
  }

  if ("text" %in% names(df) && !"plaintext" %in% names(df)) {
    df <- df %>% rename(plaintext = text)
  }

  mig_or_trans_flag <- is_mig_multi_or_translation(
    file_path,
    path_translation_test = path_translation_test,
    path_translation_train = path_translation_train
  )

  df <- df %>%
    mutate(lang_code = if (mig_or_trans_flag) lang else tweet_lang)

  df$Language <- map_chr(df$lang_code, lang_code_to_name)

  if (is.numeric(df$message_id)) {
    df$message_id <- formatC(df$message_id, format = "f", drop0trailing = TRUE)
  } else {
    df$message_id <- as.character(df$message_id)
  }

  model <- extract_model_dataset(file_path)

  if ("translation_check" %in% colnames(df)) {
    df$TranslationQuality <- ifelse(df$translation_check == 1, "Good", "Bad")
    df$Translation <- 1
  } else if (startsWith(file_path, path_translation_train) || startsWith(file_path, path_translation_test)) {
    df$Translation <- 1
    df$TranslationQuality <- "Unknown"
  } else {
    df$Translation <- 0
    df$TranslationQuality <- "Not Translated"
  }

  df_processed <- df %>%
    mutate(
      Model = model,
      Train_Test = split_type,
      Correct = ifelse(label_llama == label, 1, 0),
      file_path = file_path
    ) %>%
    select(
      message_id, Model, Language, Translation, TranslationQuality,
      Train_Test, Correct, label, label_llama, plaintext, file_path
    ) %>%
    mutate(
      Model = case_when(
        Model == "llama-32-3B-en-es-Q4" ~ "English-Spanish",
        Model == "llama-32-3B-en-Q4" ~ "English",
        Model == "llama-32-3B-es-Q4" ~ "Spanish",
        Model == "llama-32-3B-multi-Q4" ~ "Multilanguage",
        TRUE ~ Model
      ),
      label = case_when(
        label == "Neutral" ~ "neutral",
        TRUE ~ label
      )
    )

  df_processed
}

collect_input_files <- function(path, pattern) {
  list.files(path, pattern = pattern, full.names = TRUE)
}

build_combined_dataset <- function(raw_test_dir,
                                   raw_train_dir,
                                   translation_test_dir,
                                   translation_train_dir) {
  dataset_specs <- tibble::tribble(
    ~input_files,                                                      ~split_type,
    collect_input_files(raw_test_dir, "\\.csv$"),                      "Test",
    collect_input_files(raw_train_dir, "\\.csv$"),                     "Train",
    collect_input_files(translation_test_dir, "\\.xlsx$"),             "Test",
    collect_input_files(translation_train_dir, "\\.xlsx$"),            "Train"
  )

  purrr::map2_dfr(
    dataset_specs$input_files,
    dataset_specs$split_type,
    ~ purrr::map_dfr(
      .x,
      process_dataset,
      split_type = .y,
      path_translation_test = translation_test_dir,
      path_translation_train = translation_train_dir
    )
  )
}

assign_new_ids <- function(df) {
  df %>%
    group_by(plaintext) %>%
    mutate(newId = cur_group_id()) %>%
    ungroup()
}

prepare_modelling_dataset <- function(df) {
  df %>%
    assign_new_ids() %>%
    filter(!is.na(Language), !is.na(plaintext), !is.na(label_llama)) %>%
    distinct(Model, newId, Translation, .keep_all = TRUE) %>%
    mutate(
      message_id = as.factor(message_id),
      Model = as.factor(Model),
      Language = as.factor(Language),
      Train_Test = as.factor(Train_Test),
      Translation = as.factor(Translation),
      TranslationQuality = as.factor(TranslationQuality),
      Correct = as.factor(Correct)
    )
}

add_share_unrelated <- function(df) {
  df %>%
    group_by(Language, Train_Test) %>%
    mutate(Share_Unrelated = mean(label == "unrelated")) %>%
    ungroup()
}

add_language_share <- function(df) {
  df %>%
    left_join(lang_share_lookup, by = "Language") %>%
    mutate(LangShare = dplyr::coalesce(LangShare, 0))
}

add_dataset_size <- function(df) {
  df %>%
    group_by(Language, Train_Test, Translation, TranslationQuality) %>%
    mutate(
      DatasetSize = n_distinct(newId)
    ) %>%
    ungroup()
}

set_reference_levels <- function(df) {
  df %>%
    mutate(
      Model = relevel(factor(Model), ref = "English"),
      Language = relevel(factor(Language), ref = "English"),
      Train_Test = relevel(factor(Train_Test), ref = "Train"),
      label = relevel(factor(label), ref = "neutral"),
      TranslationQuality = relevel(factor(TranslationQuality), ref = "Not Translated"),
      Translation = factor(Translation)
    )
}

get_term_coef <- function(term_name, coef_df) {
  val <- coef_df$estimate[coef_df$term == term_name]

  if (length(val) == 0) {
    return(0)
  }

  val
}

get_named_coef <- function(term, coefficients) {
  if (term %in% names(coefficients)) {
    return(coefficients[[term]])
  }

  0
}

get_lang_coef <- function(language, coefficients) {
  get_named_coef(paste0("Language", language), coefficients)
}

get_translation_coef <- function(quality, coefficients) {
  get_named_coef(paste0("TranslationQuality", quality), coefficients)
}

get_label_coef <- function(label, coefficients) {
  if (tolower(label) == "neutral") {
    return(0)
  }

  get_named_coef(paste0("label", gsub("-", "", label)), coefficients)
}

get_model_label_interaction <- function(model, label, coefficients) {
  term <- paste0("Model", model, ":label", gsub("-", "", label))
  get_named_coef(term, coefficients)
}

add_significance_columns <- function(df) {
  df %>%
    mutate(
      significance = case_when(
        p.value < 0.001 ~ "***",
        p.value < 0.01  ~ "**",
        p.value < 0.05  ~ "*",
        TRUE            ~ ""
      ),
      abs_estimate = abs(estimate)
    )
}

tidy_baseline_terms <- function(model) {
  broom::tidy(model, conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      variable_group = case_when(
        str_starts(term, "Model")              ~ "Model",
        str_starts(term, "Language")           ~ "Language",
        str_starts(term, "TranslationQuality") ~ "Translation",
        str_starts(term, "label")              ~ "Label",
        TRUE                                   ~ "Other"
      ),
      clean_term = case_when(
        variable_group == "Model"       ~ str_remove(term, "^Model"),
        variable_group == "Language"    ~ str_remove(term, "^Language"),
        variable_group == "Translation" ~ str_remove(term, "^TranslationQuality"),
        variable_group == "Label"       ~ case_when(
          str_detect(term, "unrelated") ~ "Unrelated",
          str_detect(term, "anti")      ~ "Anti-Immigration",
          str_detect(term, "pro")       ~ "Pro-Immigration",
          TRUE                          ~ str_remove(term, "^label")
        ),
        TRUE                            ~ term
      )
    ) %>%
    add_significance_columns() %>%
    filter(!str_detect(term, "^Train_Test")) %>%
    arrange(variable_group, desc(abs_estimate))
}

tidy_model_language_terms <- function(model) {
  broom::tidy(model, conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    filter(grepl("^Model", term) | grepl("^Language", term)) %>%
    mutate(
      variable_group = case_when(
        grepl("^Model:Language", term) ~ "Model × Language",
        grepl("^Model", term)          ~ "Model Type",
        grepl("^Language", term)       ~ "Language",
        TRUE                           ~ "Other"
      ),
      clean_term = case_when(
        grepl("^Model:Language", term) ~ gsub(":Language", " × ", gsub("^Model", "", term)),
        grepl("^Model", term)          ~ gsub("^Model", "", term),
        grepl("^Language", term)       ~ gsub("^Language", "", term),
        TRUE                           ~ tools::toTitleCase(term)
      )
    ) %>%
    add_significance_columns()
}

tidy_share_unrelated_terms <- function(model) {
  broom::tidy(model, conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    filter(grepl("^Model", term) | grepl("Share_Unrelated", term)) %>%
    mutate(
      variable_group = case_when(
        grepl("^Model:.*Share_Unrelated", term) ~ "Model × Share Unrelated",
        grepl("^Model", term)                   ~ "Model Type",
        grepl("Share_Unrelated", term)          ~ "Share Unrelated",
        TRUE                                    ~ "Other"
      ),
      clean_term = case_when(
        grepl("^Model:.*Share_Unrelated", term) ~ gsub("_", " ", gsub(":Share_Unrelated", " × Share Unrelated", gsub("Model", "", term))),
        grepl("^Model", term)                   ~ gsub("_", " ", gsub("^Model", "", term)),
        TRUE                                    ~ tools::toTitleCase(gsub("_", " ", term))
      )
    ) %>%
    add_significance_columns()
}

tidy_translation_label_terms <- function(model) {
  broom::tidy(model, conf.int = TRUE) %>%
    filter(
      term != "(Intercept)",
      term != "Train_TestTest",
      !grepl("TranslationQuality", term),
      !grepl("^Language", term)
    ) %>%
    mutate(
      variable_group = case_when(
        grepl("^Model", term)    ~ "Model Type",
        grepl("^label", term)    ~ "Label Category",
        grepl("Train_Test", term) ~ "Data Split",
        TRUE                     ~ "Other"
      ),
      model_label = case_when(
        grepl("English-Spanish", term)                        ~ "English-Spanish",
        grepl("Multilanguage", term)                         ~ "Multilingual",
        grepl("Spanish", term) & variable_group == "Model Type" ~ "Spanish-Only",
        TRUE                                                 ~ NA_character_
      ),
      label_class = case_when(
        grepl("unrelated", term) ~ "Unrelated",
        grepl("anti", term)      ~ "Anti-Immigration",
        grepl("pro", term)       ~ "Pro-Immigration",
        TRUE                     ~ NA_character_
      ),
      clean_term = case_when(
        !is.na(model_label) & !is.na(label_class) ~ paste(model_label, "×", label_class),
        !is.na(label_class)                       ~ label_class,
        !is.na(model_label)                       ~ model_label,
        TRUE ~ tools::toTitleCase(gsub("^Model|^label|^Language|^Train_Test|^TranslationQuality", "", term))
      )
    ) %>%
    add_significance_columns() %>%
    arrange(variable_group, desc(abs_estimate))
}

tidy_dataset_size_terms <- function(model) {
  broom::tidy(model, conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      variable_group = case_when(
        str_starts(term, "Model")              ~ "Model",
        str_starts(term, "Language")           ~ "Language",
        str_starts(term, "TranslationQuality") ~ "Translation",
        str_starts(term, "label")              ~ "Label",
        term == "DatasetSize"                  ~ "Dataset Size",
        TRUE                                   ~ "Other"
      ),
      clean_term = case_when(
        variable_group == "Model"        ~ str_remove(term, "^Model"),
        variable_group == "Language"     ~ str_remove(term, "^Language"),
        variable_group == "Translation"  ~ str_remove(term, "^TranslationQuality"),
        variable_group == "Dataset Size" ~ "Dataset Size",
        variable_group == "Label"        ~ case_when(
          str_detect(term, "unrelated") ~ "Unrelated",
          str_detect(term, "anti")      ~ "Anti-Immigration",
          str_detect(term, "pro")       ~ "Pro-Immigration",
          TRUE                          ~ str_remove(term, "^label")
        ),
        TRUE                             ~ term
      )
    ) %>%
    add_significance_columns() %>%
    arrange(variable_group, desc(abs_estimate))
}

compute_model_language_surface <- function(model, df) {
  model_terms <- broom::tidy(model, conf.int = TRUE) %>%
    filter(term != "(Intercept)")

  combinations <- expand.grid(
    Model = levels(df$Model),
    Language = levels(df$Language),
    stringsAsFactors = FALSE
  )

  combinations$log_odds <- mapply(function(model_name, language_name) {
    get_term_coef(paste0("Model", model_name), model_terms) +
      get_term_coef(paste0("Language", language_name), model_terms) +
      get_term_coef(paste0("Model", model_name, ":Language", language_name), model_terms)
  }, combinations$Model, combinations$Language)

  combinations$accuracy <- plogis(combinations$log_odds)
  combinations
}

build_translation_heatmap_data <- function(model, languages, translation_df, translation_quality = "Unknown") {
  coefficients <- coef(model)
  labels <- c("neutral", "anti-immigration", "pro-immigration", "unrelated")

  expand.grid(Language = languages, Model = levels(translation_df$Model), Label = labels) %>%
    rowwise() %>%
    mutate(
      intercept = get_named_coef("(Intercept)", coefficients),
      model_coef = get_named_coef(paste0("Model", Model), coefficients),
      label_coef = get_label_coef(Label, coefficients),
      model_label_interaction = get_model_label_interaction(Model, Label, coefficients),
      language_coef = get_lang_coef(Language, coefficients),
      test_coef = get_named_coef("Train_TestTest", coefficients),
      translation_coef = get_translation_coef(translation_quality, coefficients),
      total_eta = intercept + model_coef + label_coef + model_label_interaction +
        language_coef + translation_coef + test_coef,
      Accuracy = plogis(total_eta)
    ) %>%
    ungroup()
}

compute_accuracy_ci <- function(data) {
  cm <- confusionMatrix(as.factor(data$label_llama), as.factor(data$label))
  accuracy <- cm$overall["Accuracy"]
  ci <- prop.test(sum(diag(cm$table)), sum(cm$table))$conf.int

  tibble(
    Accuracy = accuracy,
    Accuracy_Lower = ci[1],
    Accuracy_Upper = ci[2]
  )
}

generate_accuracy_df <- function(df_split, dataset_name) {
  df_split %>%
    group_by(Model, Language) %>%
    group_map(~ {
      acc <- compute_accuracy_ci(.x)
      acc$Model <- .y$Model
      acc$Language <- .y$Language
      acc$Dataset <- dataset_name
      acc
    }) %>%
    bind_rows()
}

generate_and_save_cm <- function(data, title, output_dir) {
  models <- unique(data$Model)
  results <- list()

  for (model in models) {
    model_data <- data[data$Model == model, ]
    cm <- confusionMatrix(as.factor(model_data$label_llama), as.factor(model_data$label))
    cm_matrix <- as.matrix(cm$table)
    filename <- file.path(output_dir, paste0("confusion_matrix_", model, "_", title, ".csv"))

    write.csv(cm_matrix, filename, row.names = TRUE)

    languages <- unique(model_data$Language)
    model_summary <- list()

    for (lang in languages) {
      lang_data <- model_data[model_data$Language == lang, ]
      total_pro <- sum(lang_data$label == "pro-immigration")
      total_anti <- sum(lang_data$label == "anti-immigration")
      total_tweets <- nrow(lang_data)
      pro_as_anti <- sum(lang_data$label == "pro-immigration" & lang_data$label_llama == "anti-immigration")
      anti_as_pro <- sum(lang_data$label == "anti-immigration" & lang_data$label_llama == "pro-immigration")

      model_summary[[lang]] <- tibble(
        Model = model,
        Dataset = title,
        Language = lang,
        Tweets = total_tweets,
        Pro_as_Anti_Count = pro_as_anti,
        Pro_as_Anti_Perc = ifelse(total_pro > 0, round(100 * pro_as_anti / total_pro, 2), NA),
        Anti_as_Pro_Count = anti_as_pro,
        Anti_as_Pro_Perc = ifelse(total_anti > 0, round(100 * anti_as_pro / total_anti, 2), NA)
      )
    }

    summary_df <- bind_rows(model_summary)

    write("\n\nLanguage-Level Misclassification Summary:\n", file = filename, append = TRUE)
    suppressMessages(
      write.table(
        summary_df,
        file = filename,
        sep = ",",
        row.names = FALSE,
        append = TRUE,
        col.names = TRUE
      )
    )

    results[[model]] <- summary_df
  }

  bind_rows(results)
}

plot_accuracy_intervals <- function(data, title, output_dir) {
  plot <- ggplot(
    data,
    aes(x = Language, ymin = Accuracy_Lower, ymax = Accuracy_Upper, color = Model, fill = Model)
  ) +
    geom_errorbar(width = 0.3, position = position_dodge(width = 0.6)) +
    geom_point(aes(y = (Accuracy_Lower + Accuracy_Upper) / 2), size = 3, position = position_dodge(width = 0.6)) +
    labs(
      title = title,
      x = "Language",
      y = "Accuracy Confidence Interval",
      color = "Model",
      fill = "Model"
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 10),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )

  ggsave(
    filename = paste0(title, ".png"),
    plot = plot,
    path = output_dir,
    width = 10,
    height = 6,
    dpi = 300
  )

  plot
}
