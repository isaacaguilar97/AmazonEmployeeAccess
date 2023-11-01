library(doParallel)

num_cores <- parallel::detectCores() #How many cores do I have?
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

library(tidyverse)
library(embed) # for target encoding
library(vroom)
library(DataExplorer)
library(patchwork)
library(tidymodels)
library(discrim)
library(naivebayes)
library(kknn)
library(themis)


# Load the data -----------------------------------------------------------
# setwd('~/College/Stat348/AmazonEmployeeAccess')

# Load data
amazon_train <- vroom('./train.csv')
amazon_train$ACTION <- as.factor(amazon_train$ACTION)

amazon_test <- vroom('./test.csv')

# Create 2 exploratory plots ----------------------------------------------

# dplyr::glimpse(amazon_train)
# plot_bar(amazon_train) # bar charts of all discrete variables
# plot_histogram(amazon_train) # histograms of all numerical variables
# # I see there are some with more coutns than others
# plot_missing(amazon_train) # missing values

# Clean my data ####

# Set my receipe (it will be different for each model)
# my_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
#   step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
#   step_dummy(all_nominal_predictors()) # dummy variable encoding
#   # step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
# # also step_lencode_glm() and step_lencode_bayes()

# # apply the recipe to your data
# prep <- prep(my_recipe)
# train_clean <- bake(prep, new_data = amazon_train)

#
# # LOGISTIC REGRESSION -----------------------------------------------------
# 
# # Set my recipe
# my_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
#   step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding
#   step_normalize(all_numeric_predictors()) %>%
#   step_pca(all_predictors(), threshold=.9) %>% # Reduce your matrix
#   step_smote(all_outcomes(), neighbors=5)
# 
# # Create model
# lg_mod <- logistic_reg() %>% #Type of model
#   set_engine("glm")
# 
# #Define Worflow
# amazon_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(lg_mod) %>%
#   fit(data = amazon_train) # Fit the workflow
# 
# # Get Predictions
# amazon_predictions <- predict(amazon_workflow,
#                               new_data=amazon_test,
#                               type="prob") # "class" or "prob" (see doc)
# 
# amazon_test$Action <- amazon_predictions$.pred_1
# results <- amazon_test %>%
#   rename(Id = id) %>%
#   select(Id, Action)
# 
# 
# # get csv file
# vroom_write(results, 'AmazonPredlg.csv', delim = ",")
# 
# 
# # PENALIZED LOGISTIC REGRESSION-------------------------------------------
# # ROC across all possible cut offs
# 
# # Recipe
# my_recipe <- recipe(ACTION~., data=amazon_train) %>%''
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
#   step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding
#   step_normalize(all_numeric_predictors()) %>%
#   step_pca(all_predictors(), threshold=.9) %>% # Reduce your matrix
#   step_smote(all_outcomes(), neighbors=5)
# 
# # Model
# plg_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
#   set_engine("glmnet")
# 
# # Workflow
# amazon_workflow <- workflow() %>% 
#   add_recipe(my_recipe) %>% 
#   add_model(plg_mod)
# 
# ## Grid of values to tune over
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 5) ## L^2 total tuning possibilities
# 
# ## Split data for CV
# folds <- vfold_cv(amazon_train, v = 5, repeats=1)
# 
# ## Run the CV
# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#           grid=tuning_grid,
#           metrics=metric_set(roc_auc)) # they will all use a cut off of .5
# 
# ## Find best tuning parameters
# bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# ## Finalize workflow and predict
# final_wf <- amazon_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=amazon_train)
# 
# amazon_predictions <- final_wf %>%
#   predict(new_data = amazon_test, type = "prob")
# 
# # Format table
# amazon_test$Action <- amazon_predictions$.pred_1
# results <- amazon_test %>%
#   rename(Id = id) %>%
#   select(Id, Action)
# 
# # get csv file
# vroom_write(results, 'AmazonPredsplg.csv', delim = ",")
# # penalty is first and mixture is second
# 

# RANDOM FOREST  ----------------------------------------------------------

# Recipe
my_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  # step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold=.9) %>% # Reduce your matrix
  step_smote(all_outcomes(), neighbors=5)
  # step_downsample(all_outcomes(), )


# Model
rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=300) %>%
  set_engine("ranger") %>%
  set_mode("classification")

## Workflow
amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod) %>%
  fit(data = amazon_train)

## Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1,9)), # How many Variables to choose from
                            # researches have found log of total variables is enough
                            min_n(),
                            levels = 5)

# Set up K-fold CV
folds <- vfold_cv(amazon_train, v = 5, repeats=1)

# Cross Validation
CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

# Find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize workflow
final_wf <- amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

# Predict
amazon_predictions <- final_wf %>%
  predict(new_data = amazon_test, type = "prob")

# save(file="./MyFile.RData", list=c("amazon_predictions", "final_wf", "bestTune", "CV_results"))

# Format table
amazon_test$Action <- amazon_predictions$.pred_1
results <- amazon_test %>%
  rename(Id = id) %>%
  select(Id, Action)

# get csv file
vroom_write(results, 'AmazonPredsrf.csv', delim = ",")


# # Naive Bayes -------------------------------------------------------------
# 
# ## nb model3
# nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes") # install discrim library for the naivebayes eng
# 
# my_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
#   #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding
#   step_normalize(all_numeric_predictors()) %>%
#   step_pca(all_predictors(), threshold=.9) %>% # Reduce your matrix
#   step_smote(all_outcomes(), neighbors=5)
# 
# # my_recipe_prep <- prep(my_recipe, data = amazon_train)
# # baked_data <- bake(my_recipe, new_data = NULL)
# 
# nb_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(nb_model)
# 
# ## Tune smoothness and Laplace here
# tuning_grid <- grid_regular(smoothness(),
#                             Laplace(),
#                             levels = 5)
# 
# ## Set up K-fold CV
# folds <- vfold_cv(amazon_train, v = 5, repeats=1)
# 
# CV_results <- nb_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc))
# 
# ## Find best tuning parameters
# bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# ## Finalize workflow
# final_wf <- nb_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=amazon_train)
# 
# ## Predict
# amazon_predictions <- final_wf %>%
#   predict(new_data = amazon_test, type = "prob")
# 
# #save(file="./MyFile.RData", list=c("amazon_predictions", "final_wf", "bestTune", "CV_results"))
# 
# # Format table
# amazon_test$Action <- amazon_predictions$.pred_1
# results <- amazon_test %>%
#   rename(Id = id) %>%
#   select(Id, Action)
# 
# # get csv file
# vroom_write(results, 'AmazonPredsnb.csv', delim = ",")

# # KNN ---------------------------------------------------------------------
# 
# # knn model
# knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kknn")
# 
# ## Recipe
# my_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
#   # step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding
#   step_normalize(all_numeric_predictors()) %>%
#   step_pca(all_predictors(), threshold=.9) %>% # Reduce your matrix
#   step_smote(all_outcomes(), neighbors=5)
# 
# ## Workflow
# knn_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(knn_model)
# 
# ## Tune
# tuning_grid <- grid_regular(neighbors(),
#                             levels = 10)
# 
# ## Set up K-fold CV
# folds <- vfold_cv(amazon_train, v = 5, repeats=1)
# 
# CV_results <- knn_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc))
# 
# ## Find best tuning parameters
# bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# ## Finalize workflow and predict
# 
# final_wf <- knn_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=amazon_train)
# 
# ## Fit or Tune Model HERE
# amazon_predictions <- final_wf %>%
#   predict(new_data = amazon_test, type = "prob")
# 
# # save(file="./MyFile.RData", list=c("amazon_predictions", "final_wf", "bestTune", "CV_results"))
# 
# # Format table
# amazon_test$Action <- amazon_predictions$.pred_1
# results <- amazon_test %>%
#   rename(Id = id) %>%
#   select(Id, Action)
# 
# vroom_write(results, 'AmazonPredsknn.csv', delim = ",")


# Support Vector Machine --------------------------------------------------
# Each point is a vector representing the X's
# Support vector => vectors that we are goint to use to create the boundaries
# Usually just for Binary Classification (heavy computation - make reduction)

# Kernel = How you compute the boundary matters
  # Computation fast way of computing this things
    # Linear
    # Polynomial
    # Radial

# SVM models
svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmLinear <- svm_linear(cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

## Fit or Tune Model HERE

my_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold=.9) %>% # Reduce your matrix
  step_smote(all_outcomes(), neighbors=5) #Threshold is between 0 and 1

svmP_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmPoly)

svmR_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

svmL_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmLinear)

## Tune
tuning_grid <- grid_regular(degree(),
                            cost(),
                            levels = 5)

## Set up K-fold CV
folds <- vfold_cv(amazon_train, v = 5, repeats=1)

p_results <- svmP_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

r_results <- svmR_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

l_results <- svmL_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find best tuning parameters
bestTune_p <- p_results %>%
  select_best("roc_auc")

bestTune_r <- r_results %>%
  select_best("roc_auc")

bestTune_l <- l_results %>%
  select_best("roc_auc")

## Finalize workflow
p_wf <- svmP_wf %>%
  finalize_workflow(bestTune_p) %>%
  fit(data=amazon_train)

r_wf <- svmR_wf %>%
  finalize_workflow(bestTune_r) %>%
  fit(data=amazon_train)

l_wf <- svmL_wf %>%
  finalize_workflow(bestTune_l) %>%
  fit(data=amazon_train)

## Predict
p_predictions <- p_wf %>%
  predict(new_data = amazon_test, type = "prob")

r_predictions <- r_wf %>%
  predict(new_data = amazon_test, type = "prob")

l_predictions <- l_wf %>%
  predict(new_data = amazon_test, type = "prob")

# Format table
amazon_test$Action <- p_predictions$.pred_1
results <- amazon_test %>%
  rename(Id = id) %>%
  select(Id, Action)

vroom_write(results, 'AmazonPredsSVMp.csv', delim = ",")

amazon_test$Action <- r_predictions$.pred_1
results <- amazon_test %>%
  rename(Id = id) %>%
  select(Id, Action)

vroom_write(results, 'AmazonPredsSVMr.csv', delim = ",")

amazon_test$Action <- l_predictions$.pred_1
results <- amazon_test %>%
  rename(Id = id) %>%
  select(Id, Action)

vroom_write(results, 'AmazonPredsl.csv', delim = ",")


stopCluster(cl)
