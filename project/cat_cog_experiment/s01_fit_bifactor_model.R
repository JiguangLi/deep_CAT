setwd("/Users/jiguangli/bayesian-cat")
library(pacman)
p_load("mirt", "tidyverse", "argparser", "feather",  "dplyr",  "plotmo", "caret",  "cvms", "magrittr", "arrow", "here")

# parse arguments
arguments <- arg_parser("FIT CAT COG Data") %>% 
  add_argument(
    "--data_input_dir", 
    help= "data_input_dir", 
    default= here("data", "cat_cog")
  ) %>%
  add_argument(
    "--model_output_dir", 
    help= "model_output_dir", 
    default= here("models", "cat_cog_models")
  ) %>%
  parse_args()

# read data
lines <- readLines(file.path(arguments[["data_input_dir"]], "COG5.def"))
max_vals <- as.numeric(unlist(strsplit(lines[10], ""))) 
cleaned_lines <- lines[11:(length(lines) - 1)]
split_lines <- lapply(cleaned_lines, function(x) as.numeric(unlist(strsplit(x, ""))))
df <- do.call(rbind, split_lines)
df <- as.data.frame(df)

# Filter out missing responses and dichotomize
df_complete <- df[!apply(df, 1, function(row) any(row == 0)), ]
thresholds = max_vals-1 # ordinal response between 2-4, threshold is max ordinal response -1, meaning you have to get everything right.
df_binary <- as.data.frame(mapply(function(column, threshold) {
  as.numeric(column > threshold)  
}, df, thresholds))

# fit bifactor
# Several options:
# 1-9 Executive Function B, 10-18 excutive function A, 19-30 episodic memory, 31-39 Processing Speed, 40-48 Semantic Memory, 49-57 working memory
# Rober excluded 49-57; the paper has 5 subdomains
loading_list <- c(rep(1, 18), rep(2, 12), rep(3, 9), rep(4, 9), rep(5, 9))
set.seed(1)
model <- bfactor(df_binary, loading_list)

# save model
saveRDS(model, file.path(arguments[["model_output_dir"]], "bifactor.rds"))
# save slopes and intercepts
factor_coefs <- coef(model)
num_items <- dim(df_binary)[2]
alphas_params <- matrix(0, num_items, 7)
for(j in 1:num_items){
  alphas_params[j, ] <- factor_coefs[[j]][1, 1:7]
}
colnames(alphas_params) <- c(paste0("dim",1:6), "intercepts")
arrow::write_feather(alphas_params%>% as.data.frame(), file.path(arguments[["model_output_dir"]], "bifactor_alphas.feather"))
# save binary responses
arrow::write_feather(df_binary, file.path(arguments[["model_output_dir"]], "binary_response_abs_right.feather"))
# save latent factors
bifactor_latent_traits <- fscores(model, rotate = "oblimin", response.pattern =df_binary, QMC=TRUE)
arrow::write_feather(bifactor_latent_traits %>% as.data.frame() , file.path(arguments[["model_output_dir"]], "latent_traits_abs_right.feather"))
