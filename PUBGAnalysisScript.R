## CODE HEADER: PUBGAnalysisScript
## Author: Avijay Chakravorti
## Date Created: Sep. 3 2021

###############################
### SECTION 0: LIBRARIES AND DL
###############################



# We need the following libraries:
# library(readr)
# library(caret)
# library(keras)
# library(matrixStats)
# library(dplyr)
# library(tidyr)
# library(tidyverse)
# library(corrplot)

# Install libraries if not present. This can take ~15-30 minutes, depending on internet speed.
if(!require(readr)) install.packages("readr")
if(!require(caret)) install.packages("caret")
if(!require(matrixStats)) install.packages("matrixStats")
if(!require(corrplot)) install.packages("corrplot")
if(!require(dplyr)) install.packages("dplyr")
if(!require(tidyr)) install.packages("tidyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(pls)) install.packages("pls")

# Keras install
if(!require(keras)) {
  install.packages("keras")
  library(keras)
  reticulate::install_miniconda()
  install_keras()
}

# Load libraries
library(readr)
library(caret)
library(keras)
library(matrixStats)
library(dplyr)
library(tidyr)
library(tidyverse)
library(corrplot)
library(pls)


# Set options to download pretrained models and pretrained indexes.
# Default: run all models from scratch
download_LM <- FALSE
download_PCA <- FALSE  
# download_DNN <- FALSE   # This should be downloaded with the github repo.
download_IDX <- TRUE
bool_save_models <- FALSE
bool_save_data <- FALSE

# Download main dataset
download.file("https://ahri-s3.s3.amazonaws.com/data/train_V2.csv", "train_V2.csv")

# Download previously trained model to reduce computation time.
if( download_IDX == TRUE) {
  download.file("https://ahri-s3.s3.amazonaws.com/data/small_test_idx.rds", "small_test_idx.rds")
  download.file("https://ahri-s3.s3.amazonaws.com/data/small_val_idx.rds", "small_val_idx.rds")
}
if( download_LM == TRUE) {
  download.file("https://ahri-s3.s3.amazonaws.com/data/models/m_lm1.rds", "m_lm1.rds")
}
if ( download_PCA == TRUE ) {
  download.file("https://ahri-s3.s3.amazonaws.com/data/models/m_pca.rds", "m_pca.rds")
}
# if ( download_DNN == TRUE ) {
#   
#   # Download m_dnn1 folder (create and fill)
#   dir.create('m_dnn1')
#   dir.create('m_dnn1/variables')
#   dir.create('m_dnn1/assets')
#   download.file('https://ahri-s3.s3.amazonaws.com/data/models/m_dnn1/keras_metadata.pb', 'm_dnn1/keras_metadata.pb')
#   download.file('https://ahri-s3.s3.amazonaws.com/data/models/m_dnn1/saved_model.pb', 'm_dnn1/saved_model.pb')
#   download.file('https://ahri-s3.s3.amazonaws.com/data/models/m_dnn1/variables/variables.data-00000-of-00001', 
#                 'm_dnn1/variables/variables.data-00000-of-00001')
#   download.file('https://ahri-s3.s3.amazonaws.com/data/models/m_dnn1/variables/variables.index', 'm_dnn1/variables/variables.index')
#   
#   # Download m_dnn2 folder (create and fill)
#   dir.create('m_dnn2')
#   dir.create('m_dnn2/variables')
#   dir.create('m_dnn2/assets')
#   download.file('https://ahri-s3.s3.amazonaws.com/data/models/m_dnn2/keras_metadata.pb', 'm_dnn2/keras_metadata.pb')
#   download.file('https://ahri-s3.s3.amazonaws.com/data/models/m_dnn2/saved_model.pb', 'm_dnn2/saved_model.pb')
#   download.file('https://ahri-s3.s3.amazonaws.com/data/models/m_dnn2/variables/variables.data-00000-of-00001', 
#                 'm_dnn2/variables/variables.data-00000-of-00001')
#   download.file('https://ahri-s3.s3.amazonaws.com/data/models/m_dnn2/variables/variables.index', 'm_dnn2/variables/variables.index')
#   
#   
#   download.file("https://ahri-s3.s3.amazonaws.com/data/models/history_dnn1.rds", "history_dnn1.rds")
#   download.file("https://ahri-s3.s3.amazonaws.com/data/models/history_dnn2.rds", "history_dnn2.rds")
# }




##########################
### SECTION 1: DATA IMPORT
##########################
# Let's start by loading the train set (our only dataset.)
# We will split this into train, test, and validation sets.


train_V2 <- read_csv("train_V2.csv", 
                     col_types = cols(matchType = col_factor(levels = c("solo", 
                                                                        "duo", "squad",
                                                                        "solo-fpp",
                                                                        "duo-fpp", 
                                                                        "squad-fpp"))))
train_V2 <- as_tibble(train_V2)

colnames(train_V2)
nrow(train_V2)
ncol(train_V2)

####################################
### SECTION 2: PRE-MUTATION ANALYSIS
####################################


## --- How often does an individual player appear in the dataset? ---
train_V2 %>% distinct(Id)
# Each row is a unique player. We can discard this column, it serves no purpose in analysis.



#######################################
### SECTION 3: PRE PROCESSING AND SPLIT
#######################################
# We will delete the groupId and matchId columns, to align with our row-by-row approach.
train_V2 <- train_V2[complete.cases(train_V2),] # Remove all rows with NAs
train_V2 <- na.omit(train_V2)
train_V2 <- train_V2 %>% select(-Id, -matchId, -groupId)
colnames(train_V2)
# We will attempt modeling without looking at individual player IDs. We want to focus on 
# predicting a win proportion for all players, not individual players.

# Create a small subset of this data to use with simpler R-based ML methods.
# We will only look at solo-fpp games.
small_data <- train_V2 %>% filter(matchType == "solo-fpp")
nrow(small_data)

# Select only columns of interest to ease processing
small_data <- small_data %>% select(-DBNOs, -revives, -teamKills, -matchType)
# Filter out NAs
small_data <- small_data[complete.cases(small_data),] # Remove all rows with NAs
colnames(small_data)

## ============================
## === TRAIN TEST VAL SPLIT ===
##=============================

# Do this only if there isn't already a train / test / val dataset stored in the data folder.
if( file.exists("small_val_idx.rds")==FALSE |
    file.exists("small_test_idx.rds")==FALSE
    ) {

  # Validation - trainall split from small_data. Val is 15% of small_data
  small_val_idx <- createDataPartition(small_data$winPlacePerc, times = 1, p = 0.15, 
                                       list = FALSE)
  small_val_set <- small_data[small_val_idx, ]
  small_trainall_set <- small_data[-small_val_idx, ]
  
  # Train - test split from trainall. Test is 20% of trainall.
  small_test_idx <- createDataPartition(small_trainall_set$winPlacePerc, times = 1, p = 0.20, 
                                        list = FALSE)
  small_test_set <- small_trainall_set[small_test_idx, ]
  small_train_set <- small_trainall_set[-small_test_idx, ]
  
  # Seeing lengths of all sets
  nrow(small_val_set)
  nrow(small_test_set)
  nrow(small_train_set)
  
  if(bool_save_data==TRUE) {
  # Save train test val indexes
  saveRDS(small_val_idx, file="small_val_idx.rds")
  saveRDS(small_test_idx, file="small_test_idx.rds")
  }
  
  rm(small_trainall_set)
  gc()
  
} else {
  # Load downloaded indexes
  small_val_idx <- readRDS("small_val_idx.rds")
  small_test_idx <- readRDS("small_test_idx.rds")
  
  # Create train test val sets using given indices
  small_val_set <- small_data[small_val_idx, ]
  small_trainall_set <- small_data[-small_val_idx, ]
  
  small_test_set <- small_trainall_set[small_test_idx, ]
  small_train_set <- small_trainall_set[-small_test_idx, ]
  
  rm(small_trainall_set)
  gc()
  
  # Seeing lengths of all sets
  nrow(small_val_set)
  nrow(small_test_set)
  nrow(small_train_set)
  
  gc()
}

## ==================================
## Normalization
## ==================================
# Acquire means and stds of train set columns.
train_means <- colMeans(small_train_set)
train_sds <- colSds(as.matrix(small_train_set))

# Normalize train, test, val set using train means and sds
small_train_set <- as_tibble(scale(small_train_set, center = train_means, scale = train_sds))
small_test_set <- as_tibble(scale(small_test_set, center = train_means, scale = train_sds))
small_val_set <- as_tibble(scale(small_val_set, center = train_means, scale = train_sds))

# Assign x and y for each dataset
# Splitting into x and y in case needed
small_train_x <- as.matrix(small_train_set %>% select(-winPlacePerc))
small_test_x <- as.matrix(small_test_set %>% select(-winPlacePerc))
small_val_x <- as.matrix(small_val_set %>% select(-winPlacePerc))

small_train_y <- as.matrix(small_train_set %>% select(winPlacePerc))
small_test_y <- as.matrix(small_test_set %>% select(winPlacePerc))
small_val_y <- as.matrix(small_val_set %>% select(winPlacePerc))



################################################
### SECTION 4: POSTMUTATION ANALYSIS - SMALL SET
################################################

# Get correlation matrix and plot
small_cormat <- cor(small_train_set)
corrplot::corrplot(small_cormat)

# Plot damage vs. kills. Probably a correlation there.
small_train_set %>% sample_n(10000) %>%
  ggplot(aes(damageDealt, kills)) + geom_point() + ggtitle("Kills vs. Damage Dealt")

# Plot heals vs. damageDealt
small_train_set %>% sample_n(10000) %>%
  ggplot(aes(damageDealt, heals)) + geom_point() + ggtitle("Heals vs. Damage Dealt")


################################
### SECTION 5: LINEAR REGRESSION
################################

# Read the stored linear model, if not present make a new one.
if( file.exists("m_lm1.rds")==FALSE
) {

  # Train a linear model using every variable to predict winPlacePerc column.
  m_lm1 <- train(winPlacePerc ~ ., method = "lm", data = small_train_set)
  summary(m_lm1)
  
  # Save model to disk
  if(bool_save_models==TRUE) saveRDS(m_lm1, file="m_lm1.rds")
  
} else {
  
  # Load model from disk
  m_lm1 <- readRDS("m_lm1.rds")
  
}

# Make predictions
pred_lm1 <- predict(m_lm1, small_test_set)

# Plot predictions vs actual and calculate RMSE
plot(pred_lm1, small_test_y, xlab = "Predicted Normalized Rank", 
     ylab = "Actual Normalized Rank") + title("LinearModel Actual vs. Predicted")
rmse_lm1 <- RMSE(pred_lm1, small_test_y)
rmse_lm1

# We see a simple linear model gets us, on average, within 10% of the 
# actual win place percentage.

#################################################
### SECTION 6: Principal Component Analysis (PCA)
#################################################
# Load PCA model from disk if present, else generate a new one.

if( file.exists("m_pca.rds")==FALSE
) {

  # Train pca model
  m_pca <- train(winPlacePerc ~ ., method = "pcr", data = small_train_set)
  summary(m_pca)
  
  # Save model to disk
  if(bool_save_models==TRUE) saveRDS(m_pca, file="m_pca.rds")
  
} else {
  # Load pca model if present
  m_pca <- readRDS("m_pca.rds")
}
  
  
  
  
# Make predictions
pred_pca <- predict(m_pca, small_test_set)

# Plot predictions vs actual / calculate rmse
plot(pred_pca, small_test_y, xlab = "Predicted Normalized Rank", 
     ylab = "Actual Normalized Rank") +  title("PCA Model Actual vs. Predicted")
rmse_pca <- RMSE(pred_pca, small_test_y)
rmse_pca

# This is worse than the linear model. It seems to almost never over-predict someone's 
# relative rank, but often underpredicts it.

#############################################
### SECTION 7: BASELINE PREDICTION of AVERAGE
#############################################
# Let's see what predicting the average does.

pred_avg <- matrix(mean(small_train_y), nrow = nrow(small_test_y), ncol = 1)
rmse_avg <- RMSE(pred_avg, small_test_y)
rmse_avg


###############################
### SECTION 8: KERAS DNNS
###############################
# Will save these models if they didn't previously exist. This is to ensure that others 
# can run the code without having to have an NVIDIA gpu for reasonable runtimes.
# To run this code from scratch, just delete the model in the models folder.

# Run this code if models aren't found in folder data/models
if( 
    file.exists("m_dnn1")==FALSE |
    file.exists("history_dnn1.rds")==FALSE
) {

  ## =====================
  ## === MODEL 1: DNN1 ===
  ## =====================
  
  # Now we're going to use tensorflow deep neural nets to achieve hopefully more 
  # specific results.
  
  # Constructing a deep neural network with 5 layers.
  # - Layers 1-4: 512-unit fully connected layers
  # - Layer 5: Output layer, 1 node, outputs a single number per input row.
  m_dnn1 <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = 'relu', input_shape = ncol(small_train_x)) %>%
    layer_dense(units = 512, activation = 'relu') %>%
    layer_dense(units = 512, activation = 'relu') %>%
    layer_dense(units = 512, activation = 'relu') %>%
    layer_dense(units = ncol(small_train_y), activation = "linear")
  m_dnn1
  
  # Compile the model
  compile(m_dnn1, loss = "mse" , optimizer = "adam", 
          metrics = list("mse"))
  
  # Train the model, record history of training.
  history_dnn1 <- fit(m_dnn1, small_train_x, small_train_y, epochs = 30, batch_size = 8192, 
                      validation_data = list(small_test_x, small_test_y))
  
  # Save the model
  if(bool_save_models==TRUE) {
  m_dnn1 %>% save_model_tf(filepath = 'm_dnn1')
  m_dnn1 %>% save_model_weights_tf(filepath = 'm_dnn1_weights')
  saveRDS(history_dnn1, file="history_dnn1.rds")
  }
  
} else {
  # Load model from disk if avaliable
  m_dnn1 <- load_model_tf('m_dnn1')
  history_dnn1 <- readRDS("history_dnn1.rds")
}

# Make predictions
pred_dnn1 <- m_dnn1 %>% predict(small_test_x)

# Plot predictions / calculate RMSE
plot(pred_dnn1, small_test_y, xlab = "Predicted Normalized Rank", 
     ylab = "Actual Normalized Rank") +  title("DNN1 Model Actual vs. Predicted")
rmse_dnn1 <- RMSE(pred_dnn1, small_test_y)
rmse_dnn1

## ======================
## === MODEL 2: DNN2 ====
## ======================

# New DNN, less complex, lower batch size
layersize_dnn2 <- 128

# Check if model is avaliable on disk. If not, run the training code.
if( file.exists("m_dnn2")==FALSE | 
    file.exists("history_dnn2.rds")==FALSE
) {

  # Constructing a deep neural network with 5 layers.
  # - Layers 1-4: layersize_dnn2-unit fully connected layers
  # - Layer 5: Output layer, 1 node, outputs a single number per input row.
  m_dnn2 <- keras_model_sequential() %>%
    layer_dense(units = layersize_dnn2, activation = 'relu', 
                input_shape = ncol(small_train_x)) %>%
    layer_dense(units = layersize_dnn2, activation = 'relu') %>%
    layer_dense(units = layersize_dnn2, activation = 'relu') %>%
    layer_dense(units = layersize_dnn2, activation = 'relu') %>%
  #              kernel_regularizer = regularizer_l2(0.01), 
  #              bias_regularizer = regularizer_l2(0.01))     %>%
    layer_dense(units = ncol(small_test_y), activation = "linear")
    
  m_dnn2
  
  # Compiling the model
  compile(m_dnn2, loss = "mse" , optimizer = "adam", 
          metrics = list("mse"))
  
  # Training the model, storing history as training progresses.
  history_dnn2 <- fit(m_dnn2, small_train_x, small_train_y, epochs = 18, batch_size = 128, 
                      validation_data = list(small_test_x, small_test_y) )
  
  # Save model to disk
  if(bool_save_models==TRUE) {
  m_dnn2 %>% save_model_tf(filepath = 'm_dnn2')
  m_dnn2 %>% save_model_weights_tf(filepath = 'm_dnn2_weights')
  saveRDS(history_dnn2, file="history_dnn2.rds")
  }
  
} else {
  # Load model from disk if avaliable
  m_dnn2 <- load_model_tf("m_dnn2")
  history_dnn2 <- readRDS("history_dnn2.rds")
}

# Make predictions
pred_dnn2 <- m_dnn2 %>% predict(small_test_x)

# Plot predictions / calculate RMSE
plot(pred_dnn2, small_test_y, xlab = "Predicted Normalized Rank", 
     ylab = "Actual Normalized Rank") +  title("dnn2 Model Actual vs. Predicted")
rmse_dnn2 <- RMSE(pred_dnn2, small_test_y)
rmse_dnn2


##############################
### SECTION 9: LM DNN ENSEMBLE
##############################

# Join both prediction vectors, take row-wise average.
pred_lm_dnn_ensemble <- rowMeans(cbind(pred_lm1, pred_dnn2))

# Plot prediction and calc rmse
plot(pred_lm_dnn_ensemble, small_test_y, xlab = "Predicted Normalized Rank", 
     ylab = "Actual Normalized Rank") +  title("LM+DNN2 Ensemble Actual vs. Predicted")
rmse_lm_dnn_ensemble <- RMSE(pred_lm_dnn_ensemble, small_test_y)
rmse_lm_dnn_ensemble


########################################
### SECTION 10: UN-NORMALIZED ERROR PLOTS
########################################
# Get SD of winPlacePerc to un-normalize predictions and actual.
winPlacePerc_sd <- train_sds[length(train_sds)]
winPlacePerc_mean <- train_means[length(train_means)]

# Create dataframe of all predictions, with a column for true values.
test_pred_df <- tibble(actual = small_test_y*winPlacePerc_sd + winPlacePerc_mean, 
                       avg = pred_avg*winPlacePerc_sd + winPlacePerc_mean,
                       lm1 = pred_lm1*winPlacePerc_sd + winPlacePerc_mean,
                       pca = pred_pca*winPlacePerc_sd + winPlacePerc_mean,
                       dnn1 = pred_dnn1*winPlacePerc_sd + winPlacePerc_mean,
                       dnn2 = pred_dnn2*winPlacePerc_sd + winPlacePerc_mean,
                       ens = pred_lm_dnn_ensemble*winPlacePerc_sd + winPlacePerc_mean
                       )
head(test_pred_df)

# Create a sample of 40000 rows so plotting is easy.
test_pred_df_sample <- test_pred_df %>% sample_n(40000)


## PLOTTING AVERAGE PREDS
test_pred_df_sample %>% ggplot(aes(x=avg, y=actual)) +
  geom_point(aes(color = abs(avg - actual)) ) + ggtitle("Average Model")

## PLOTTING LINEARMODEL PREDS
test_pred_df_sample %>% ggplot(aes(x=lm1, y=actual)) +
  geom_point(aes(color = abs(lm1 - actual)) ) + ggtitle("Linear Regression Model")

## PLOTTING PCA PREDS 
test_pred_df_sample %>% ggplot(aes(x=pca, y=actual)) +
  geom_point(aes(color = abs(pca - actual)) ) + ggtitle("PCA Model")

## PLOTTING DNN1 PREDS
test_pred_df_sample %>% ggplot(aes(x=dnn1, y=actual)) +
  geom_point(aes(color = abs(dnn1 - actual)) ) + ggtitle("DNN1 Model")

## PLOTTING DNN2 PREDS
test_pred_df_sample %>% ggplot(aes(x=dnn2, y=actual)) +
  geom_point(aes(color = abs(dnn2 - actual)) ) + ggtitle("DNN2 Model")

## PLOTTING ENSEMBLE PREDS
test_pred_df_sample %>% ggplot(aes(x=ens, y=actual)) +
  geom_point(aes(color = abs(ens - actual)) ) + ggtitle("Ensemble Model")


############################################
### SECTION 11: EXPLORING RESULTS OF LM1 ###
############################################
plot(varImp(m_lm1), title="Linear Model Variable Importance")

############################################
### SECTION 12: FINAL RESULTS ON VAL SET ###
############################################

## FINAL RMSE ##
pred_final <- m_dnn2 %>% predict(small_val_x)
final_rmse <- RMSE(pred_final, small_val_y)
final_rmse

## FINAL RMSE SCALED ##
final_rmse * winPlacePerc_sd
