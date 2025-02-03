################################################################################################
####################################### TREE MODELS ############################################
################################################################################################

# Remove all objects in the environment
rm(list=ls())

# Load dataset and assign to variable
data <- Dataset2_Companies

# Create a copy of the dataset
datacart <- data

# Remove the 'ID' column (which is not used for the analysis)
data$ID <- NULL

# Rename the first column to 'Default' (target variable)
colnames(data)[1] <- "Default"

# Set random seed for reproducibility
set.seed(387)

# Split the data into training (70%) and testing (30%) sets
trainIndex <- createDataPartition(datacart$Default, p=0.7, list = FALSE, times = 1)
train.data.cart <- datacart[trainIndex, ]
test.data.cart  <- datacart[-trainIndex, ]

# Check the distribution of 'Default' in the training and test sets
table(train.data.cart$Default)  # Count of defaults in training set
table(test.data.cart$Default)   # Count of defaults in test set

# Display the proportions of 'Default' in both training and test sets
prop.table(table(train.data.cart$Default))
prop.table(table(test.data.cart$Default))

# Plot pie charts to visualize the distribution of 'Default' vs 'Non Default' in both sets
par(mfrow=c(1,2))
pie3D(prop.table(table(train.data.cart$Default)),
      main="Default Vs Non Default in Training Dataset",
      labels=c("No_Default", "Default"),
      col = c("Turquoise", "Medium Sea Green"))

pie3D(prop.table(table(test.data.cart$Default)),
      main="Default Vs Non Default in Test Dataset",
      labels=c("No_Default", "Default"),
      col = c("Aquamarine", "Dark Sea Green"))

# Set control parameters for the rpart (CART) model
r.ctrl <- rpart.control(minsplit = 100,      # Minimum number of observations for a split
                        minbucket = 10,     # Minimum number of observations in each leaf node
                        cp = 0,             # Complexity parameter to control tree pruning
                        xval = 10)          # Number of cross-validation folds

# Exclude columns that are not needed (not used in this code directly)
cart.train <- train.data.cart

# Build the CART model using rpart function
set.seed(3)
m1 <- rpart(formula = Default ~ .,  # Formula specifying 'Default' as the target variable and all others as predictors
            data = cart.train,       # Training data
            method = "class",        # Classification task
            control = r.ctrl)        # Control parameters

# Display the tree using a fancy plot
library(rattle)  # Load rattle package for tree visualization
library(RColorBrewer)
fancyRpartPlot(m1)  # Visualize the initial tree

# Print and plot complexity parameter (cp) table
printcp(m1)  # Display the complexity parameter table
plotcp(m1)   # Plot the cp table to visualize the tree's performance at different levels of pruning

# Prune the tree using a specific cp value to avoid overfitting
ptree <- prune(m1, cp = 0.0017)

# Print and visualize the pruned tree
printcp(ptree)  # Print the pruned tree's complexity parameter table
fancyRpartPlot(ptree, uniform = TRUE, main = "Final Tree", palettes = c("Blues", "Oranges"))

# Make predictions on the training set
cart.train$predict.class <- predict(ptree, cart.train, type = "class")  # Predicted classes
cart.train$predict.score <- predict(ptree, cart.train, type = "prob")  # Predicted probabilities

# Create deciles for predictions (divide predictions into 10 groups based on the predicted probability)
decile <- function(x) {
  deciles <- vector(length = 10)
  for (i in seq(0.1, 1, 0.1)) {
    deciles[i * 10] <- quantile(x, i, na.rm = TRUE)
  }
  return(ifelse(x < deciles[1], 1,
                ifelse(x < deciles[2], 2,
                       ifelse(x < deciles[3], 3,
                              ifelse(x < deciles[4], 4,
                                     ifelse(x < deciles[5], 5,
                                            ifelse(x < deciles[6], 6,
                                                   ifelse(x < deciles[7], 7,
                                                          ifelse(x < deciles[8], 8,
                                                                 ifelse(x < deciles[9], 9, 10)))))))))))
}

# Apply deciling to the predicted probabilities
cart.train$deciles <- decile(cart.train$predict.score[,2])

# Load libraries for further analysis
library(data.table)
library(scales)

# Rank the data by deciles and calculate various classification metrics
tmp_DT <- data.table(cart.train)
rank <- tmp_DT[, list(cnt = length(Default),
                      cnt_resp = sum(Default == 1),
                      cnt_non_resp = sum(Default == 0)),
               by = deciles][order(-deciles)]

# Calculate response rate and cumulative response and non-response rates
rank$rrate <- round(rank$cnt_resp / rank$cnt, 4)
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp), 4)
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp), 4)

# Calculate Kolmogorov-Smirnov statistic and update classification metrics
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp) * 100
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

# Load packages for performance evaluation
library(ROCR)  # For ROC curve
library(ineq)  # For Gini index calculation

# Create a prediction object
pred <- prediction(cart.train$predict.score[,2], cart.train$Default)

# Calculate the ROC performance and other evaluation metrics
perf <- performance(pred, "tpr", "fpr")  # True positive rate vs false positive rate
KS <- max(attr(perf, 'y.values')[[1]] - attr(perf, 'x.values')[[1]])  # Kolmogorov-Smirnov statistic

# Calculate AUC (Area Under the Curve)
auc <- performance(pred, "auc")
auc <- as.numeric(auc@y.values)

# Calculate Gini index
gini = ineq(cart.train$predict.score[,2], type="Gini")

# Print confusion matrix for training data
with(cart.train, table(Default, predict.class))

# Predict on test data
cart.test <- test.data.cart
cart.test$predict.class <- predict(ptree, cart.test, type = "class")
cart.test$predict.score <- predict(ptree, cart.test, type = "prob")
cart.test$deciles <- decile(cart.test$predict.score[,2])

# Rank the test data and calculate classification metrics
tmp_DT <- data.table(cart.test)
rank <- tmp_DT[, list(cnt = length(Default),
                      cnt_resp = sum(Default == 1),
                      cnt_non_resp = sum(Default == 0)),
               by = deciles][order(-deciles)]

rank$rrate <- round(rank$cnt_resp / rank$cnt, 4)
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp), 4)
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp), 4)

# Calculate Kolmogorov-Smirnov statistic and update classification metrics for test data
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp) * 100
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

# Evaluate performance on test data
pred <- prediction(cart.test$predict.score[,2], cart.test$Default)
perf <- performance(pred, "tpr", "fpr")

# Plot ROC curve
plot(perf, main = 'ROC', col = 'blue', lwd = 2)

# Output metrics for the test set
KS
auc
gini

# Calculate accuracy and classification error rate for the test set
Accuracy = (563 + 26) / (563 + 16 + 9 + 26)
Accuracy

# Calculate the classification error rate
Classification_Error_Rate = 1 - Accuracy
Classification_Error_Rate



##############################################################################################
#========================= CART with Centralities Measures ====================#
##############################################################################################
# Prepare and clean the dataset
data.ce <- data_with_centralities_

# Convert relevant columns to factors
data.ce$Default <- as.factor(data.ce$Default)
data.ce$Industry <- as.factor(data.ce$Industry)
data.ce$Region <- as.factor(data.ce$Region)
data.ce$Corporate_network <- as.factor(data.ce$Corporate_network)

# Standardize numerical columns
standardized_df <- data.ce %>%
  mutate(across(where(is.numeric), ~ (.-mean(.))/sd(.)))

data.ce <- standardized_df

# One-hot encode categorical variables
industry_encoded <- model.matrix(~ 0 + as.factor(data.ce$Industry))
region_encoded <- model.matrix(~ 0 + as.factor(data.ce$Region))
corporate_dummy <- model.matrix(~ 0 + as.factor(data.ce$Corporate_network))

# Assign custom names to the encoded variables
colnames(industry_encoded) <- c("Industry_1", "Industry_2", "Industry_3")
colnames(region_encoded) <- c("Region_1", "Region_2", "Region_3")
colnames(corporate_dummy) <- c("Corporate_no", "Corporate_yes")

# Combine the encoded variables with the rest of the dataset
data.ce <- cbind(data.ce[, !names(data.ce) %in% c("Industry", "Region", "Corporate_network")],
                 industry_encoded, region_encoded, corporate_dummy)

# Convert Default column to numeric
data.ce$Default <- as.numeric(as.character(data.ce$Default))

# Split the data into training and testing sets
set.seed(3767)
trainIndex.ce <- createDataPartition(data.ce$Default, p=0.7, list=FALSE, times=1)
train.data.ce <- data.ce[trainIndex.ce, ]
test.data.ce  <- data.ce[-trainIndex.ce, ]

# Check the distribution of the target variable in training and testing sets
table(train.data.ce$Default)
table(test.data.ce$Default)
prop.table(table(train.data.ce$Default))
prop.table(table(test.data.ce$Default))

# Visualize the distribution of the target variable in both sets
par(mfrow=c(1,2))
pie3D(prop.table(table(train.data.ce$Default)), main="Default Vs Non Default in Training Data",
      labels=c("No_Default", "Default"), col = c("Turquoise", "Medium Sea Green"))
pie3D(prop.table(table(test.data.ce$Default)), main='Default Vs Non Default in Test Data',
      labels=c("No_Default", "Default"), col = c("Aquamarine", "Dark Sea Green"))

# Set the control parameters for rpart (CART model)
r.ctrl <- rpart.control(minsplit = 100, minbucket = 10, cp = 0, xval = 10)

# Build the CART model
m2 <- rpart(Default ~ ., data = train.data.ce, method = "class", control = r.ctrl)

# Plot the decision tree
fancyRpartPlot(m2)

# Print complexity parameter table for pruning decisions
printcp(m2)
plotcp(m2)

# Prune the decision tree based on the chosen complexity parameter (cp)
ptree.ce <- prune(m2, cp=0.0017)

# Plot the pruned tree
fancyRpartPlot(ptree.ce, uniform = TRUE, main = "Final Tree", palettes = c("Blues", "Oranges"))

# Make predictions on the training set
train.data.ce$predict.class <- predict(ptree.ce, train.data.ce, type = "class")
train.data.ce$predict.score <- predict(ptree.ce, train.data.ce, type = "prob")

# Define a function to calculate deciles
decile <- function(x) {
  deciles <- vector(length=10)
  for (i in seq(0.1, 1, .1)) {
    deciles[i*10] <- quantile(x, i, na.rm = TRUE)
  }
  return(ifelse(x < deciles[1], 1,
                ifelse(x < deciles[2], 2,
                       ifelse(x < deciles[3], 3,
                              ifelse(x < deciles[4], 4,
                                     ifelse(x < deciles[5], 5,
                                            ifelse(x < deciles[6], 6,
                                                   ifelse(x < deciles[7], 7,
                                                          ifelse(x < deciles[8], 8,
                                                                 ifelse(x < deciles[9], 9, 10)))))))))))
}

# Apply decile calculation to the training data
train.data.ce$deciles <- decile(train.data.ce$predict.score[,2])

# Rank the deciles and calculate cumulative values
tmp_DT <- data.table(train.data.ce)
rank <- tmp_DT[, list(cnt = length(Default),
                      cnt_resp = sum(Default == 1),
                      cnt_non_resp = sum(Default == 0)), by = deciles][order(-deciles)]

rank$rrate <- round(rank$cnt_resp / rank$cnt, 4)
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp), 4)
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp), 4)
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp) * 100
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

# Calculate performance metrics
pred.ce <- prediction(train.data.ce$predict.score[,2], train.data.ce$Default)
perf.ce <- performance(pred.ce, "tpr", "fpr")
KS.ce <- max(attr(perf.ce, 'y.values')[[1]] - attr(perf.ce, 'x.values')[[1]])
auc.ce <- performance(pred.ce, "auc")
auc.ce <- as.numeric(auc.ce@y.values)
gini.ce <- ineq(train.data.ce$predict.score[,2], type = "Gini")
with(train.data.ce, table(Default, predict.class))

# Predict on the test set
test.data.ce$predict.class <- predict(ptree.ce, test.data.ce, type = "class")
test.data.ce$predict.score <- predict(ptree.ce, test.data.ce, type = "prob")
test.data.ce$deciles <- decile(test.data.ce$predict.score[,2])

# Calculate performance metrics for the test set
tmp_DT <- data.table(test.data.ce)
rank <- tmp_DT[, list(cnt = length(Default),
                      cnt_resp = sum(Default == 1),
                      cnt_non_resp = sum(Default == 0)), by = deciles][order(-deciles)]

rank$rrate <- round(rank$cnt_resp / rank$cnt, 4)
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp), 4)
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp), 4)
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp) * 100
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

# Calculate performance metrics for the test set
pred.ce <- prediction(test.data.ce$predict.score[,2], test.data.ce$Default)
perf.ce <- performance(pred.ce, "tpr", "fpr")
KS.ce <- max(attr(perf.ce, 'y.values')[[1]] - attr(perf.ce, 'x.values')[[1]])
auc.ce <- performance(pred.ce, "auc")
auc.ce <- as.numeric(auc.ce@y.values)
gini.ce <- ineq(test.data.ce$predict.score[,2], type = "Gini")
with(test.data.ce, table(Default, predict.class))

# Plot ROC curve
plot(perf.ce, main = 'ROC', col = 'blue', lwd = 2)

# Print key metrics
KS.ce
auc.ce
gini.ce

# Calculate classification accuracy and error rate
Accuracy.ce <- (562 + 30) / (562 + 12 + 10 + 30)
Accuracy.ce
Tasso_di_errore_di_classificazione_ce <- 1 - Accuracy.ce
Tasso_di_errore_di_classificazione_ce


###############################################################################
#========================= CART with cluster===============================#
###############################################################################

data.c <- data_clusters

# One-hot encoding for the variables "Industry", "Region", and "Corporate_network"
industry_encoded <- model.matrix(~ 0 + as.factor(data.c$Industry))
region_encoded <- model.matrix(~ 0 + as.factor(data.c$Region))
corporate_dummy <- model.matrix(~ 0 + as.factor(data.c$Corporate_network))

# Assigning custom names to the encoded variables
colnames(industry_encoded) <- c("Industry_1", "Industry_2", "Industry_3")
colnames(region_encoded) <- c("Region_1", "Region_2", "Region_3")
colnames(corporate_dummy) <- c("Corporate_no", "Corporate_yes")

# Creating the training dataframe with the encoded variables and custom names
data.c <- cbind(data.c[, !names(data.c) %in% c("Industry", "Region", "Corporate_network")],
                industry_encoded, region_encoded, corporate_dummy)

# Converting target variables and feature columns
data.c$Default <- as.factor(data.c$Default)
data.c$Industry <- as.factor(data.c$Industry)
data.c$Region <- as.factor(data.c$Region)
data.c$Corporate_network <- as.factor(data.c$Corporate_network)
data.c$cluster <- as.factor(data.c$cluster)

# Splitting the data into training and testing sets
set.seed(15)
trainIndex.c <- createDataPartition(data.c$Default, p = 0.7, list = FALSE, times = 1)
train.data.c <- data.c[trainIndex.c, ]
test.data.c <- data.c[-trainIndex.c, ]

# Checking the distribution of the data
table(train.data.c$Default)
table(test.data.c$Default)
prop.table(table(train.data.c$Default))
prop.table(table(test.data.c$Default))

# Displaying pie charts for the class distribution
par(mfrow = c(1, 2))
pie3D(prop.table(table(train.data.c$Default)), main="Default vs Non Default in Training Dataset",
      labels=c("No_Default", "Default"), col = c("Turquoise", "Medium Sea Green"))
pie3D(prop.table(table(test.data.c$Default)), main="Default vs Non Default in Testing Dataset",
      labels=c("No_Default", "Default"), col = c("Aquamarine", "Dark Sea Green"))

# Setting control parameters for rpart
r.ctrl <- rpart.control(minsplit = 100, minbucket = 10, cp = 0, xval = 10)

# Decision tree model (CART)
set.seed(26)
m3 <- rpart(Default ~ ., data = train.data.c, method = "class", control = r.ctrl)
fancyRpartPlot(m3)

# Pruning the model
ptree3 <- prune(m3, cp = 0.0017)
fancyRpartPlot(ptree3, uniform = TRUE, main = "Final Tree", palettes = c("Blues", "Oranges"))

# Predictions and probability calculation for the training set
train.data.c$predict.class <- predict(ptree3, train.data.c, type = "class")
train.data.c$predict.score <- predict(ptree3, train.data.c, type = "prob")

# Creating deciles
train.data.c$deciles <- decile(train.data.c$predict.score[, 2])

# Encoding the ranking of deciles
tmp_DT <- data.table(train.data.c)
rank <- tmp_DT[, .(cnt = length(Default), cnt_resp = sum(Default == 1), cnt_non_resp = sum(Default == 0)),
               by = deciles][order(-deciles)]
rank$rrate <- round(rank$cnt_resp / rank$cnt, 4)
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp), 4)
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp), 4)
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp) * 100
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

# Calculating the ROC curve and Gini Index for the training set
library(ROCR)
library(ineq)
pred.clu <- prediction(train.data.c$predict.score[, 2], train.data.c$Default)
perf.clu <- performance(pred.clu, "tpr", "fpr")
plot(perf.clu, main = 'ROC Curve', col = 'red', lwd = 2)
KS.c <- max(attr(perf.clu, 'y.values')[[1]] - attr(perf.clu, 'x.values')[[1]])
auc.c <- performance(pred.clu, "auc")
auc.c <- as.numeric(auc.c@y.values)
gini.c <- ineq(train.data.c$predict.score[, 2], type = "Gini")

# Model accuracy
accuracy_c <- (1320 + 70) / (1320 + 28 + 17 + 70)

# Prediction on the test set
test.data.c$predict.class <- predict(ptree3, test.data.c, type = "class")
test.data.c$predict.score <- predict(ptree3, test.data.c, type = "prob")
test.data.c$deciles <- decile(test.data.c$predict.score[, 2])

# Classification and ranking for the test data
tmp_DT <- data.table(test.data.c)
rank <- tmp_DT[, .(cnt = length(Default), cnt_resp = sum(Default == 1), cnt_non_resp = sum(Default == 0)),
               by = deciles][order(-deciles)]
rank$rrate <- round(rank$cnt_resp / rank$cnt, 4)
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp), 4)
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp), 4)
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp) * 100
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

# Calculating the ROC curve for the test set
pred.c <- prediction(test.data.c$predict.score[, 2], test.data.c$Default)
perf.c <- performance(pred.c, "tpr", "fpr")
plot(perf.c, main = 'ROC Curve for Test Data', col = 'blue', lwd = 2)

KS.c <- max(attr(perf.c, 'y.values')[[1]] - attr(perf.c, 'x.values')[[1]])
auc.c <- performance(pred.c, "auc")
auc.c <- as.numeric(auc.c@y.values)
gini.c <- ineq(test.data.c$predict.score[, 2], type = "Gini")

# Accuracy and classification error for the test data
accuracy_c_test <- (567 + 28) / (567 + 15 + 5 + 28)
classification_error <- 1 - accuracy_c_test

# Comparing ROC curves
plot(perf.clu, col = 'blue', lwd = 2)
plot(perf.c, add = TRUE, col = 'green', lwd = 2)
legend("right", legend = c("Training Model", "Test Model"), lty = 1, col = c("blue", "green"))
