#========================= Random Forest ===============================#
library(randomForest)

# Prepare training and testing data
data.rf <- data

set.seed(150)

# Split the data into training (70%) and testing (30%)
trainIndex <- createDataPartition(data.rf$Default, p=0.7, list = FALSE, times = 1)
data.train.rf <- data.rf[trainIndex, ]
data.test.rf  <- data.rf[-trainIndex,]

# Fit a Random Forest model
fit3 <- randomForest(as.factor(Default) ~ ., data = data.train.rf, na.action=na.roughfix)

# Get predicted probabilities on training data
fit3_fitForest_t <- predict(fit3, newdata = data.train.rf, type="prob")[,2]

# Combine actual values and predicted probabilities
fit3_fitForest.na_t <- as.data.frame(cbind(data.train.rf$Default, fit3_fitForest_t))
colnames(fit3_fitForest.na_t) <- c('Default','pred')

# Remove NA values
fit3_fitForest.narm_t <- as.data.frame(na.omit(fit3_fitForest.na_t))

# Calculate model performance
fit3_pred_t <- prediction(fit3_fitForest.narm_t$pred, fit3_fitForest.narm_t$Default)
fit3_perf_t <- performance(fit3_pred_t, "tpr", "fpr")

# ROC curve plot for model performance
plot(fit3_perf_t, colorize=TRUE, lwd=2, main = "fit3 ROC: Random Forest", col = "blue")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3)

# Calculate AUROC, KS Statistic, and Gini Coefficient
fit3_AUROC_t <- round(performance(fit3_pred_t, measure = "auc")@y.values[[1]] * 100, 2)
fit3_KS_t <- round(max(attr(fit3_perf_t, 'y.values')[[1]] - attr(fit3_perf_t, 'x.values')[[1]]) * 100, 2)
fit3_Gini_t <- (2 * fit3_AUROC_t - 100)

# Print AUROC, KS, and Gini results
cat("AUROC: ", fit3_AUROC_t, "\tKS: ", fit3_KS_t, "\tGini:", fit3_Gini_t, "\n")

# Convert predicted probabilities to binary classes (0 or 1) with cutoff 0.5
predicted_classes_t <- ifelse(fit3_fitForest.narm_t$pred >= 0.5, 1, 0)

# Calculate confusion matrix for training data
confusion_matrix_t <- table(Actual = fit3_fitForest.narm_t$Default, Predicted = predicted_classes_t)
print(confusion_matrix_t)

# Accuracy calculation
Accuracy = (571 + 17) / (571 + 25 + 1 + 17)
Accuracy


#========================= Test Performance ===============================#
# Get predicted probabilities on testing data
fit3_fitForest <- predict(fit3, newdata = data.test.rf, type="prob")[,2]

# Combine actual values and predicted probabilities for testing data
fit3_fitForest.na <- as.data.frame(cbind(data.test.rf$Default, fit3_fitForest))
colnames(fit3_fitForest.na) <- c('Default','pred')

# Remove NA values
fit3_fitForest.narm <- as.data.frame(na.omit(fit3_fitForest.na))

# Calculate model performance on test data
fit3_pred <- prediction(fit3_fitForest.narm$pred, fit3_fitForest.narm$Default)
fit3_perf <- performance(fit3_pred, "tpr", "fpr")

# Plot Variable Importance for Random Forest
varImpPlot(fit3, main="Random Forest: Variable Importance")

# ROC curve plot for test data performance
plot(fit3_perf, colorize=TRUE, lwd=2, main = "fit3 ROC: Random Forest", col = "blue")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3)

# Calculate AUROC, KS Statistic, and Gini Coefficient for test data
fit3_AUROC <- round(performance(fit3_pred, measure = "auc")@y.values[[1]] * 100, 2)
fit3_KS <- round(max(attr(fit3_perf, 'y.values')[[1]] - attr(fit3_perf, 'x.values')[[1]]) * 100, 2)
fit3_Gini <- (2 * fit3_AUROC - 100)

# Print AUROC, KS, and Gini results for test data
cat("AUROC: ", fit3_AUROC, "\tKS: ", fit3_KS, "\tGini:", fit3_Gini, "\n")

# Convert predicted probabilities to binary classes (0 or 1) with cutoff 0.5
predicted_classes <- ifelse(fit3_fitForest.narm$pred >= 0.5, 1, 0)

# Calculate confusion matrix for test data
confusion_matrix <- table(Actual = fit3_fitForest.narm$Default, Predicted = predicted_classes)
print(confusion_matrix)

# Accuracy calculation for test data
Accuracy = (571 + 17) / (571 + 25 + 1 + 17)
Accuracy

###########################################################################################
#========================= Random Forest with cluster ===============================#
###########################################################################################

data.c <- data_clusters

# Codifica one-hot encoding per le variabili "Industry", "Region", e "Corporate_network"
industry_encoded <- model.matrix(~ 0 + as.factor(data.c$Industry))
region_encoded <- model.matrix(~ 0 + as.factor(data.c$Region))
corporate_dummy <- model.matrix(~ 0 + as.factor(data.c$Corporate_network))

# Assegnazione di nomi personalizzati alle variabili codificate
colnames(industry_encoded) <- c("Industry_1", "Industry_2", "Industry_3")
colnames(region_encoded) <- c("Region_1", "Region_2", "Region_3")
colnames(corporate_dummy) <- c("Corporate_no", "Corporate_yes")

# Creazione del dataframe di addestramento con le variabili codificate e nomi personalizzati
data.c <- cbind(data.c[, !names(data.c) %in% c("Industry", "Region", "Corporate_network")], industry_encoded, region_encoded, corporate_dummy)

# Converti la variabile di target in un fattore
data.c$Default <- as.factor(data.c$Default)

# Suddividi il dataset in dati di addestramento (70%) e test (30%)
set.seed(153)
trainIndex <- createDataPartition(data.c$Default, p = 0.7, list = FALSE, times = 1)
data.train.c <- data.c[trainIndex, ]
data.test.c  <- data.c[-trainIndex,]

# Allena il modello Random Forest
fit1 <- randomForest(Default ~ ., data = data.train.c, na.action = na.roughfix)

# Predizione sui dati di test
fit1_fitForest <- predict(fit1, newdata = data.test.c, type = "prob")[, 2]
fit1_fitForest.na <- as.data.frame(cbind(data.test.c$Default, fit1_fitForest))
colnames(fit1_fitForest.na) <- c('Default', 'pred')

# Rimuovi i valori NA
fit1_fitForest.narm <- na.omit(fit1_fitForest.na)

# Calcola la performance del modello
fit1_pred <- prediction(fit1_fitForest.narm$pred, fit1_fitForest.narm$Default)
fit1_perf <- performance(fit1_pred, "tpr", "fpr")

# Plot variabili di importanza
varImpPlot(fit1, main = "Random Forest: Variable Importance")

# Plot delle performance del modello (curva ROC)
plot(fit1_perf, colorize = TRUE, lwd = 2, main = "fit1 ROC: Random Forest", col = "blue")
lines(x = c(0, 1), y = c(0, 1), col = "red", lwd = 1, lty = 3)

# Calcolo AUROC, KS e GINI
fit1_AUROC <- round(performance(fit1_pred, measure = "auc")@y.values[[1]] * 100, 2)
fit1_KS <- round(max(attr(fit1_perf, 'y.values')[[1]] - attr(fit1_perf, 'x.values')[[1]]) * 100, 2)
fit1_Gini <- (2 * fit1_AUROC - 100)
cat("AUROC: ", fit1_AUROC, "\tKS: ", fit1_KS, "\tGini:", fit1_Gini, "\n")

# Converti i valori previsti in classi binarie (0 o 1) utilizzando una soglia di cutoff (ad esempio 0.5)
predicted_classes <- ifelse(fit1_fitForest.narm$pred >= 0.5, 1, 0)

# Calcola la matrice di confusione
confusion_matrix <- table(Actual = fit1_fitForest.narm$Default, Predicted = predicted_classes)

# Visualizza la matrice di confusione
print(confusion_matrix)

# Calcolo dell'accuratezza
Accuracy.c = (567 + 22) / (567 + 20 + 5 + 22)
Accuracy.c


# Confronto tra le curve ROC dei diversi modelli

plot(fit3_perf, col = 'blue', lwd = 2)
plot(fit2_perf, add = TRUE, col = 'red', lwd = 2)
plot(fit1_perf, add = TRUE, col = 'green', lwd = 2)
legend("right", legend = c("Model_base", "Model_network", "Model_clusters"), lty = (1:1), col = c("blue", "red", "green"))



###########################################################################################
#========================= Random Forest with Centralities ===============================#
###########################################################################################


# Prepare the data, including encoding categorical variables and merging with centralities
data.ce <- data_with_centralities_

# One-hot encoding for "Industry", "Region", and "Corporate_network" variables
industry_encoded <- model.matrix(~ 0 + as.factor(data.ce$Industry))
region_encoded <- model.matrix(~ 0 + as.factor(data.ce$Region))
corporate_dummy <- model.matrix(~ 0 + as.factor(data.ce$Corporate_network))

# Assign custom column names to the encoded variables
colnames(industry_encoded) <- c("Industry_1", "Industry_2", "Industry_3")
colnames(region_encoded) <- c("Region_1", "Region_2", "Region_3")
colnames(corporate_dummy) <- c("Corporate_no", "Corporate_yes")

# Create the final training dataset with encoded variables and custom names
data.ce <- cbind(data.ce[, !names(data.ce) %in% c("Industry", "Region", "Corporate_network")],
                 industry_encoded, region_encoded, corporate_dummy)

# Convert relevant columns to factors for modeling
data.ce$Default    <- as.factor(data.ce$Default)
data.ce$Industry    <- as.factor(data.ce$Industry)
data.ce$Region      <- as.factor(data.ce$Region)
data.ce$Corporate_network <- as.factor(data.ce$Corporate_network)

# Set a seed for reproducibility and split the dataset into training (70%) and testing (30%)
set.seed(151)
trainIndex <- createDataPartition(data.ce$Default, p=0.7, list = FALSE, times = 1)
data.train.ce <- data.ce[trainIndex, ]
data.test.ce  <- data.ce[-trainIndex,]

# Fit a Random Forest model
fit2 <- randomForest(Default ~ ., data = data.train.ce, na.action=na.roughfix)

# Get predicted probabilities for the test data
fit2_fitForest <- predict(fit2, newdata = data.test.ce, type="prob")[,2]

# Combine actual values and predicted probabilities for test data
fit2_fitForest.na <- as.data.frame(cbind(data.test.ce$Default, fit2_fitForest))
colnames(fit2_fitForest.na) <- c('Default','pred')

# Remove NA values from the dataset
fit2_fitForest.narm <- as.data.frame(na.omit(fit2_fitForest.na))

# Create the prediction object for performance analysis
fit2_pred <- prediction(fit2_fitForest.narm$pred, fit2_fitForest.narm$Default)

# Calculate the performance metrics: True Positive Rate (tpr) vs False Positive Rate (fpr)
fit2_perf <- performance(fit2_pred, "tpr", "fpr")

# Plot variable importance for Random Forest
varImpPlot(fit2, main="Random Forest: Variable Importance")

# Plot the ROC curve for model performance
plot(fit2_perf, colorize=TRUE, lwd=2, main = "fit2 ROC: Random Forest", col = "blue")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3)

# Calculate AUROC, KS Statistic, and Gini Coefficient
fit2_AUROC <- round(performance(fit2_pred, measure = "auc")@y.values[[1]] * 100, 2)
fit2_KS <- round(max(attr(fit2_perf, 'y.values')[[1]] - attr(fit2_perf, 'x.values')[[1]]) * 100, 2)
fit2_Gini <- (2 * fit2_AUROC - 100)

# Print the results: AUROC, KS, and Gini
cat("AUROC: ", fit2_AUROC, "\tKS: ", fit2_KS, "\tGini:", fit2_Gini, "\n")

# Convert predicted probabilities to binary classes (0 or 1) with a cutoff threshold of 0.5
predicted_classes <- ifelse(fit2_fitForest.narm$pred >= 0.5, 1, 0)

# Create the confusion matrix
confusion_matrix <- table(Actual = fit2_fitForest.narm$Default, Predicted = predicted_classes)

# Print the confusion matrix
print(confusion_matrix)

# Calculate accuracy
Accuracy.ce <- (566 + 25) / (566 + 25 + 6 + 17)
Accuracy.ce


