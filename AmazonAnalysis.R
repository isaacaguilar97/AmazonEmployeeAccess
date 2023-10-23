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

# Clean my data ###

# 
# my_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
#   step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
#   step_dummy(all_nominal_predictors()) # dummy variable encoding
#   # step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
# # also step_lencode_glm() and step_lencode_bayes()
# 
# # # apply the recipe to your data
# # prep <- prep(my_recipe)
# # train_clean <- bake(prep, new_data = amazon_train)
# # 
# # # -------------------------------------------------------------------------
# # 
# # 
# # #Do Mosaic plot
# # #Do chi-square table
# # 
# # # 112 is the answer
# # 
# # 
# # # LOGISTIC REGRESSION -----------------------------------------------------
# # 
# # # Create model
# # my_mod <- logistic_reg() %>% #Type of model
# #   set_engine("glm")
# # 
# # #Define Worflow
# # amazon_workflow <- workflow() %>% 
# #   add_recipe(my_recipe) %>%
# #   add_model(my_mod) %>%
# #   fit(data = amazon_train) # Fit the workflow
# # 
# # # Get Predictions
# # amazon_predictions <- predict(amazon_workflow,
# #                               new_data=amazon_test,
# #                               type="prob") # "class" or "prob" (see doc)
# # 
# # # # Make histogram to determine cut off
# # # hist(amazon_predictions$.pred_1)
# # # 
# # # # Get action column with cut off
# # # action <- amazon_predictions %>%
# # #   mutate(Action = ifelse(.pred_1 > 0.99, 1, 0))
# # 
# # # Format table
# # 
# # 
# # 
# # amazon_test$Action <- amazon_predictions$.pred_1
# # results <- amazon_test %>%
# #   rename(Id = id) %>%
# #   select(Id, Action)
# # 
# # 
# # # get csv file
# # vroom_write(results, 'AmazonPredsreg.csv', delim = ",")  
# #   
# # 
# # # PENALIZED LOGISTIC REGRESSION -------------------------------------------
# # # ROC across all posible cut offs
# # 
# # my_recipe <- recipe(ACTION~., data=amazon_train) %>%
# #   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
# #   step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
# #   # step_dummy(all_nominal_predictors()) # dummy variable encoding
# #   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
# # 
# # my_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
# #   set_engine("glmnet")
# # 
# # amazon_workflow <- workflow() %>%
# # add_recipe(my_recipe) %>%
# # add_model(my_mod)
# # 
# # ## Grid of values to tune over
# # tuning_grid <- grid_regular(penalty(),
# #                             mixture(),
# #                             levels = 5) ## L^2 total tuning possibilities
# # 
# # ## Split data for CV
# # folds <- vfold_cv(amazon_train, v = 3, repeats=1)
# # 
# # ## Run the CV
# # CV_results <- amazon_workflow %>%
# # tune_grid(resamples=folds,
# #           grid=tuning_grid,
# #           metrics=metric_set(roc_auc)) # they will all use a cut off of .5
# # 
# # ## Find best tuning parameters
# # bestTune <- CV_results %>% 
# #   select_best("roc_auc")
# # 
# # ## Finalize workflow and predict
# # final_wf <- amazon_workflow %>% 
# #   finalize_workflow(bestTune) %>% 
# #   fit(data=amazon_train)
# # 
# # amazon_predictions <- final_wf %>%
# #   predict(new_data = amazon_test, type = "prob")
# # 
# # # Get action column with cut off
# # # action <- amazon_predictions %>%
# # #   mutate(Action = ifelse(.pred_class > 0.99, 1, 0))
# # 
# # # Format table
# # amazon_test$Action <- amazon_predictions$.pred_1
# # results <- amazon_test %>%
# #   rename(Id = id) %>%
# #   select(Id, Action)
# # 
# # 
# # # get csv file
# # vroom_write(results, 'AmazonPredspreg.csv', delim = ",")  
# # # penalty is first and mixture is second
# # 
# # 
# # 
# 
# 
# # RANDOM FOREST  ----------------------------------------------------------
# 
# 
# my_mod <- rand_forest(mtry = tune(),
#                       min_n=tune(),
#                       trees=250) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# my_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor)  %>% # turn all numeric features into factors
#   step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
#   step_dummy(all_nominal_predictors()) # dummy variable encoding
# # step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
# 
# ## Create a workflow with model & recipe
# 
# amazon_workflow <- workflow() %>% 
#   add_recipe(my_recipe) %>%
#   add_model(my_mod) %>%
#   fit(data = amazon_train)
# 
# ## Set up grid of tuning values
# 
# tuning_grid <- grid_regular(mtry(range = c(1,ncol(amazon_train)-1)), # How many Variables to choose from 
#                             # researches have found log of total variables is enough
#                             min_n(), # Number of observations in a leaf
#                             levels = 5)
# 
# ## Set up K-fold CV
# folds <- vfold_cv(amazon_train, v = 10, repeats=1)
# 
# CV_results <- amazon_workflow %>%
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
# final_wf <- amazon_workflow %>% 
#   finalize_workflow(bestTune) %>% 
#   fit(data=amazon_train)
# 
# amazon_predictions <- final_wf %>%
#   predict(new_data = amazon_test, type = "prob")
# 
# save(file="./MyFile.RData", list=c("amazon_predictions", "final_wf", "bestTune", "CV_results"))
# 
# # Format table
# amazon_test$Action <- amazon_predictions$.pred_1
# results <- amazon_test %>%
#   rename(Id = id) %>%
#   select(Id, Action)
# 
# 
# # get csv file
# vroom_write(results, 'AmazonPredspreg.csv', delim = ",")
# 
# 
# # Naive Bayes -------------------------------------------------------------


## nb model3
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

my_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor)  %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_pca(all_predictors(), threshold=.9) #Threshold is between 0 and 1

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here
tuning_grid <- grid_regular(smoothness(),
                            Laplace(),
                            levels = 5)

## Set up K-fold CV
folds <- vfold_cv(amazon_train, v = 10, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(MSE))

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and predict

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

## Predict
#predict(nb_wf, new_data=myNewData, type=)

amazon_predictions <- final_wf %>%
  predict(new_data = amazon_test, type = "prob")

#save(file="./MyFile.RData", list=c("amazon_predictions", "final_wf", "bestTune", "CV_results"))

# Format table
amazon_test$Action <- amazon_predictions$.pred_1
results <- amazon_test %>%
  rename(Id = id) %>%
  select(Id, Action)


# get csv file
vroom_write(results, 'AmazonPredspreg.csv', delim = ",")


# KNN ---------------------------------------------------------------------


## knn model
# knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kknn")
# 
# ## Reciepe
# my_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor)  %>% # turn all numeric features into factors
#   step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
#   step_normalize(all_numeric_predictors()) %>%
#   step_pca(all_predictors(), threshold=.9) #Threshold is between 0 and 1
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
# folds <- vfold_cv(amazon_train, v = 10, repeats=1)
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
# save(file="./MyFile.RData", list=c("amazon_predictions", "final_wf", "bestTune", "CV_results"))
# 
# # Format table
# amazon_test$Action <- amazon_predictions$.pred_1
# results <- amazon_test %>%
#   rename(Id = id) %>%
#   select(Id, Action)
# 
# vroom_write(results, 'AmazonPredspreg.csv', delim = ",")



# Dimension Reduction -----------------------------------------------------
  
stopCluster(cl)
