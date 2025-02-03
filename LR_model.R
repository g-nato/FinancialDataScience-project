################################################################################################
#========================= Logistic Regression ===============================#
################################################################################################


# Split the dataset into training and testing samples using stratified sampling
perc <- 0.7  # Set the percentage for the training data
set.seed(122)  # Set seed for reproducibility

# Create stratified partition for training and testing
div <- createDataPartition(y = data$Default, p = perc, list = F)

# Training Sample: 70% of the data
data.train <- data[div,]  # Training dataset
percentage(data.train$Default)  # Show the proportion of defaults in training set

# Test Sample: Remaining 30% of the data
data.test <- data[-div,]  # Test dataset
percentage(data.test$Default)  # Show the proportion of defaults in test set

# Build the initial logistic regression model using all predictors
fit1 <- glm(Default ~ ., data = data.train, family = binomial())  # Logistic regression
summary(fit1)  # Display the summary of the initial model

# Perform stepwise regression to select the best predictors
fit_step <- step(fit1, direction = 'both')  # Both forward and backward stepwise selection
summary(fit_step)  # Display the summary of the stepwise regression model

# Fit the final model based on the selected predictors
fit_step <- glm(Default ~ Total_assets + Net_income + Profit_Loss_after_tax + Net_income_on_Total_Assets,
                data = data.train, family = binomial())  # Logistic regression with selected variables
summary(fit_step)  # Display the summary of the final model

# Compare the stepwise model with the initial model using ANOVA
anova(fit_step, fit1, test = "Chisq")  # Chi-squared test for model comparison

# Get the coefficients of the final model
coef(fit_step)  # Display the coefficients of the model

# Calculate and interpret the odds ratios (exp of coefficients)
exp(coefficients(fit_step))  # Exponentiate the coefficients to get odds ratios

# Get predicted probabilities of default for the test set using the final model
data.test$score <- predict(fit_step, type = 'response', data.test)  # Predicted probabilities using final model
data.test$score1 <- predict(fit1, type = 'response', data.test)  # Predicted probabilities using initial model

# Set a threshold (cut-off) for classification into Default or Non-Default
cut_off <- def_perc  # Define the threshold for prediction (default percentage)
data.test$pred <- ifelse(data.test$score <= cut_off, 0, 1)  # Assign predicted class based on the threshold

# Plot ROC curve to visualize the model's performance
perf_auroc <- performance(prediction(data.test$score, data.test$Default), "auc")  # Calculate AUC
auroc <- as.numeric(perf_auroc@y.values)  # Extract the AUC value

# Plot ROC curve (True Positive Rate vs. False Positive Rate)
perf_plot <- performance(prediction(data.test$score, data.test$Default), "tpr", "fpr")
plot(perf_plot, main = 'ROC Curve', col = 'blue', lwd = 2)  # Plot ROC curve with True Positive Rate vs. False Positive Rate

# Create confusion matrix to evaluate the classification results
confusion_matrix <- table(data.test$Default, data.test$pred)  # Confusion matrix showing predicted vs actual outcomes

# Calculate Gini Index based on AUC (2 * AUC - 1)
gini_index <- 2 * auroc - 1  # Gini Index: a measure of model's discriminatory power

# Calculate the KS (Kolmogorov-Smirnov) statistic, which measures the separation between classes
ks_value <- max(perf_plot@y.values[[1]] - perf_plot@x.values[[1]])  # Maximum difference between TPR and FPR

# Calculate overall accuracy (percentage of correct predictions)
overall_accuracy <- mean(data.test$pred == data.test$Default)  # Accuracy: Proportion of correct predictions

# Print the evaluation results
cat("Gini Index:", gini_index, "\n")  # Print Gini Index
cat("KS Value:", ks_value, "\n")  # Print KS Value
cat("Overall Accuracy:", overall_accuracy, "\n")  # Print Overall Accuracy



##############################################################################################
#========================= Logistic Regression with Centralities Measures ====================#
##############################################################################################


# Prepare data
data_network <- data_with_centralities_

# Calculate the default rate
def_perc_centra <- sum(data_network$Default) / length(data_network$Default)
print(def_perc_centra)

# Convert categorical variables to factors
data_network$Region <- as.factor(data_network$Region)
data_network$Industry <- as.factor(data_network$Industry)
data_network$Corporate_network <- as.factor(data_network$Corporate_network)
data_network$Default <- as.factor(data_network$Default)

# Apply one-hot encoding to categorical variables (Industry, Region, Corporate_network)
industry_encoded <- model.matrix(~ 0 + as.factor(data_network$Industry))
region_encoded <- model.matrix(~ 0 + as.factor(data_network$Region))
corporate_dummy <- model.matrix(~ 0 + as.factor(data_network$Corporate_network))

# Assign custom names to the encoded variables for clarity
colnames(industry_encoded) <- c("Industry_1", "Industry_2", "Industry_3")
colnames(region_encoded) <- c("Region_1", "Region_2", "Region_3")
colnames(corporate_dummy) <- c("Corporate_no", "Corporate_yes")

# Combine the original dataset with the encoded variables
data_network <- cbind(data_network[, !names(data_network) %in% c("Industry", "Region", "Corporate_network")], industry_encoded, region_encoded, corporate_dummy)

# Set up stratified sampling for splitting the dataset into training and test samples
perc <- 0.7
set.seed(1707)  # Set random seed for reproducibility
div_c <- createDataPartition(y = data_network$Default, p = perc, list = FALSE)

# Training Sample
data_network.train <- data_network[div_c, ]

# Test Sample
data_network.test <- data_network[-div_c, ]

# Fit logistic regression model including all predictors
fit_full <- glm(Default ~ ., data = data_network.train, family = binomial())
summary(fit_full)

# Perform stepwise regression to select the best features
fit_centralities <- step(fit_full, direction = 'both')
summary(fit_centralities)

# Perform ANOVA to compare the full model and the stepwise model
anova(fit_centralities, fit_full, test = "Chisq")

# Get predicted default probabilities for the test dataset
data_network.test$score_centralities <- predict(fit_centralities, type = 'response', newdata = data_network.test)

# Set a cutoff based on the default rate and classify the predictions
cut_off <- def_perc_centra
data_network.test$pred <- ifelse(data_network.test$score_centralities <= cut_off, 0, 1)

# False Positive Rate (FPR) calculation
n_neg <- sum(data_network.test$Default == '0')
data_network.test$fp_flag <- ifelse(data_network.test$pred == 1 & data_network.test$Default == '0', 1, 0)
fpr <- sum(data_network.test$fp_flag) / n_neg  # False Positive Rate

# False Negative Rate (FNR) calculation
n_pos <- sum(data_network.test$Default == '1')
data_network.test$fn_flag <- ifelse(data_network.test$pred == 0 & data_network.test$Default == '1', 1, 0)
fnr <- sum(data_network.test$fn_flag) / n_pos  # False Negative Rate

# Sensitivity and Specificity calculation
sensitivity <- 1 - fnr
specificity <- 1 - fpr

# Plot AUROC (Area Under the Receiver Operating Characteristic curve)
perf_auroc_centralities <- performance(prediction(data_network.test$score_centralities, data_network.test$Default), "auc")
auroc_centralities <- as.numeric(perf_auroc_centralities@y.values)

# ROC Curve plot
perf_plot_c <- performance(prediction(data_network.test$score_centralities, data_network.test$Default), "tpr", "fpr")
plot(perf_plot_c, main = 'ROC Curve', col = 'blue', lwd = 2)

# Calculate Gini Index based on AUROC
gini_index_cen <- 2 * auroc_centralities - 1

# Calculate the KS statistic
ks_value_ce <- max(perf_plot_c@y.values[[1]] - perf_plot_c@x.values[[1]])

# Calculate overall accuracy of the model
overall_accuracy_ce <- mean(data_network.test$pred == data_network.test$Default)

# Print evaluation metrics
cat("Gini Index:", gini_index_cen, "\n")
cat("KS Value:", ks_value_ce, "\n")
cat("Overall Accuracy:", overall_accuracy_ce, "\n")


###############################################################################
#========================= Logistic Regression with cluster===============================#
###############################################################################

# Copy the original data to a new variable
data_clus <- data_clusters
str(data_clus)  # Check the structure of the data

# Calculate the percentage of defaults in the dataset
def_perc <- sum(data_clus$Default) / length(data_clus$Default)
print(def_perc)

# Convert categorical variables to factors
data_clus$Default <- as.factor(data_clus$Default)
data_clus$Region <- as.factor(data_clus$Region)
data_clus$Industry <- as.factor(data_clus$Industry)
data_clus$Corporate_network <- as.factor(data_clus$Corporate_network)
data_clus$cluster <- as.factor(data_clus$cluster)


# Apply one-hot encoding to categorical variables: "Industry", "Region", and "Corporate_network"
industry_encoded <- model.matrix(~ 0 + as.factor(data_clus$Industry))
region_encoded <- model.matrix(~ 0 + as.factor(data_clus$Region))
corporate_dummy <- model.matrix(~ 0 + as.factor(data_clus$Corporate_network))
cluster_dummy <- model.matrix(~ 0 + as.factor(data_clus$cluster))

# Assign custom column names to the encoded variables
colnames(industry_encoded) <- c("Industry_1", "Industry_2", "Industry_3")
colnames(region_encoded) <- c("Region_1", "Region_2", "Region_3")
colnames(corporate_dummy) <- c("Corporate_no", "Corporate_yes")
colnames(cluster_dummy) <- c("cluster_1", "cluster_2", "cluster_3", "cluster_4", "cluster_5", "cluster_6")


# Remove the original categorical variables and add the encoded variables to the dataset
data_clus <- cbind(data_clus[, !names(data_clus) %in% c("Industry", "Region", "Corporate_network", "cluster")],
                   industry_encoded, region_encoded, corporate_dummy, cluster_dummy)


# Perform stratified sampling to create a 70% training set and a 30% testing set
perc <- 0.7
set.seed(190)  # Set seed for reproducibility
div <- createDataPartition(y = data_clus$Default, p = perc, list = FALSE)

# Training Sample (70% of the data)
data_clus.train <- data_clus[div,]
percentage(data_clus.train$Default)  # Check the proportion of defaults in the training set

# Test Sample (30% of the data)
data_clus.test <- data_clus[-div,]
percentage(data_clus.test$Default)  # Check the proportion of defaults in the test set


# Fit a logistic regression model using all predictors
fit_cluster <- glm(Default ~ ., data = data_clus.train, family = binomial())
summary(fit_cluster)  # Display the summary of the initial model


# Perform stepwise regression to select the best predictors
fit_step_cluster <- step(fit_cluster, direction = 'both')
summary(fit_step_cluster)  # Display the summary of the stepwise regression model


# Compare the stepwise model with the initial model using the Chi-squared test
anova(fit_step_cluster, fit_cluster, test = "Chisq")


# Calculate and display the odds ratios by exponentiating the coefficients
exp(coefficients(fit_step_cluster))


# Get predicted probabilities of default for the test set using the stepwise model
data_clus.test$score_clus <- predict(fit_step_cluster, type = 'response', newdata = data_clus.test)

# Get predicted probabilities of default for the test set using the initial model
data_clus.test$score1_clus <- predict(fit_cluster, type = 'response', newdata = data_clus.test)


# Classify predictions as 0 or 1 based on the default percentage threshold
cut_off <- def_perc
data_clus.test$pred_clus <- ifelse(data_clus.test$score_clus <= cut_off, 0, 1)


# Calculate AUROC (Area Under the ROC Curve)
perf_auroc <- performance(prediction(data_clus.test$score_clus, data_clus.test$Default), "auc")
auroc_cluster <- as.numeric(perf_auroc@y.values)  # Extract AUROC value

# Plot the ROC curve (True Positive Rate vs. False Positive Rate)
perf_plot_cluster <- performance(prediction(data_clus.test$score_clus, data_clus.test$Default), "tpr", "fpr")
plot(perf_plot_cluster, main = 'ROC', col = 'blue', lwd = 2)

# Display the AUROC value
auroc_cluster


# Create a confusion matrix to compare predicted vs. actual values
confusion_matrix <- table(data_clus.test$Default, data_clus.test$pred_clus)
confusion_matrix

# Calculate the Gini Index (2 * AUC - 1)
gini_index_clu <- 2 * auroc_cluster - 1

# Calculate the KS (Kolmogorov-Smirnov) statistic
ks_value_clu <- max(perf_plot_cluster@y.values[[1]] - perf_plot_cluster@x.values[[1]])

# Calculate overall accuracy (percentage of correct predictions)
overall_accuracy_clu <- mean(data_clus.test$pred_clus == data_clus.test$Default)

# Print the performance metrics
cat("Gini Index:", gini_index_clu, "\n")
cat("KS Value:", ks_value_clu, "\n")
cat("Overall Accuracy:", overall_accuracy_clu, "\n")


# Calculate accuracy using a threshold of 0.5 for binary classification
predictions <- ifelse(data_clus.test$score_clus >= 0.5, 1, 0)
accuracy_clu <- mean(predictions == data_clus.test$Default)

# Print the accuracy result
cat("Accuracy:", accuracy_clu, "\n")


# Plot the ROC curves for the base model, the network model, and the cluster model
plot(perf_plot, col = 'blue', lwd = 2)  # Base model
plot(perf_plot_cluster, add = TRUE, col = 'green', lwd = 2)  # Cluster model
legend("right", legend = c("Base Model", "Cluster Model"), lty = 1, col = c("blue", "green"))
