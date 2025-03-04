setwd("/Users/jiguangli/bayesian-cat/project/dese_experiment")
source("probit_em_mirt.R")
source("probit_em_util.R")
library(pacman)
p_load("tidyverse", "argparser", "feather",  "magic" , "pander", "dplyr",  "plotmo", "caret",  "cvms", "magrittr", "arrow", "here", "hash", "tictoc")

# parse arguments
arguments <- arg_parser("FIT DESE data in 2022") %>% 
  add_argument(
    "--data_input_dir", 
    help= "data_input_dir", 
    default= here("data","DESE_data", "sample_processed", "2022")
  ) %>%
  add_argument(
    "--model_output_dir", 
    help= "model_output_dir", 
    default= here("models", "dese")
  ) %>%
  add_argument(
    "--grades", 
    help= "grades_considered", 
    default= c("08")
  ) %>%
  add_argument(
    "--year", 
    help= "exam_year", 
    default= c("22")
  ) %>%
  add_argument(
    "--sample_size", 
    help= "number of students per grade", 
    default= 1000
  ) %>%
  parse_args()

# read data
data <- paste0(paste0("grade_", arguments[["grades"]], "_sampled"), ".feather") %>%
  set_names(., paste0("grade", arguments[["grades"]])) %>%
  map(function(.x) arrow::read_feather(file.path(arguments[["data_input_dir"]], .x)))
set.seed(1)
large_k <- 5
lambda1 <- 0.1

# grade 08 model
ir_grade08 <- data$grade08 %>%
  select(contains("eitem")|contains("mitem"))
print(colnames(ir_grade08))
nitems <- ncol(ir_grade08)
loading_starts_large <- hash("alphas"= matrix(runif(nitems*large_k, 0, 0.2), nitems, large_k),
                             "intercepts"= runif(nitems , -0.2,0.2), "c_params" = rep(0.5, large_k))
lambda0_path <- c(1, 5, 10, 20, 40, 60, 80, 100)
tic()
px_em <- dynamic_posterior_exploration(data = ir_grade08 %>% as.matrix() %>% unname(), k = large_k, ibp_alpha = 0.2, mc_samples =50,
                                       ssl_lambda0_path = lambda0_path, ssl_lambda1 = lambda1, pos_init =TRUE,
                                       max_iterations= 100, epsilon = 0.05, PX = TRUE, varimax = FALSE,
                                       loading_constraints= NULL, start = loading_starts_large,
                                       plot=TRUE, stop_rotation=100, random_state = 1, cores=8)
toc()
par(mfrow=c(1,1), mar=c(2, 2, 2, 2))
plot(px_em$lambda0_100$alphas, digits=1, main= "Estimated Loading(Grade 08)",  text.cell=list(cex=0.6), key=NULL)
saveRDS(px_em, file.path(arguments[["model_output_dir"]], "px_em_grade08.rds"))

# save response
alphas <- px_em$lambda0_100$alphas
# swap the third and the fourth factor for visualization
alphas[, c(3,4)] <- alphas[, c(4,3)]
# Factor number five is not identified
intercepts <- px_em$lambda0_100$intercepts
item_params = cbind(alphas[, 1:4], intercepts)
colnames(item_params) <- c(paste0("dim",1:4), "intercepts")
arrow::write_feather(item_params%>% as.data.frame(), file.path(arguments[["model_output_dir"]], "px_em_item_params.feather"))
arrow::write_feather(ir_grade08, file.path(arguments[["model_output_dir"]], "grade8_item_responses.feather"))
