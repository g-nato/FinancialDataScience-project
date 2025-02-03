# Install necessary packages
install.packages("readxl")
install.packages("ggplot2")
install.packages("readr")
install.packages("nnet")
install.packages("neuralnet")
install.packages("plotrix")
install.packages("data.table")
install.packages("ROCR")
install.packages("ineq")

# Load required libraries
library(readxl)
library(readr)
library(ggplot2)
library(nnet)
library(neuralnet)
library(plotrix)
library(data.table)
library(scales)
library(ROCR)
library(ineq)

# Create a copy of the dataset for analysis
data <- Dataset2_Companies
data1 <- data

# Summary of the dataset
summary(data)

# One-hot encoding for categorical variables
industry_encoded <- model.matrix(~ 0 + as.factor(data$Industry))
region_encoded <- model.matrix(~ 0 + as.factor(data$Region))
corporate_dummy <- model.matrix(~ 0 + as.factor(data$Corporate_network))

# Assign customized names to encoded variables
colnames(industry_encoded) <- c("Industry_1", "Industry_2", "Industry_3")
colnames(region_encoded) <- c("Region_1", "Region_2", "Region_3")
colnames(corporate_dummy) <- c("Corporate_no", "Corporate_yes")

# Create the final dataset with encoded variables
data <- cbind(data[, !names(data) %in% c("Industry", "Region", "Corporate_network")],
              industry_encoded, region_encoded, corporate_dummy)

# Check the distribution of the target variable
table(data$Default)

# ---------------------------------- DATA SPLITTING ----------------------------------
set.seed(233)
trainIndex <- createDataPartition(data$Default, p=0.7, list=FALSE, times=1)
NN.train.data <- data[trainIndex, ]
NN.test.data  <- data[-trainIndex, ]

# Check distribution of training and testing datasets
table(NN.train.data$Default)
table(NN.test.data$Default)
prop.table(table(NN.train.data$Default))
prop.table(table(NN.test.data$Default))

# ---------------------------------- DATA VISUALIZATION ----------------------------------
par(mfrow=c(1,2))

pie3D(prop.table(table(NN.train.data$Default)),
      main="Default vs Non-Default in Training Dataset",
      labels=c("No_Default", "Default"),
      col = c("Turquoise", "Medium Sea Green"))

pie3D(prop.table(table(NN.test.data$Default)),
      main="Default vs Non-Default in Testing Dataset",
      labels=c("No_Default", "Default"),
      col = c("Aquamarine", "Dark Sea Green"))

# ---------------------------------- DATA SCALING ----------------------------------
selected_features <- c("Total_assets", "Shareholders_funds", "Long_term_debt", "Loans", "Turnover",
                       "EBITDA", "Net_income", "Tangible_fixed_assets", "Profit_Loss_after_tax",
                       "Current_liabilities", "Current_assets", "Net_income_on_Total_Assets", "Leverage",
                       "Tangible_on_total_assets", "Industry_1", "Industry_2", "Industry_3", "Region_1",
                       "Region_2", "Region_3", "Corporate_no", "Corporate_yes")

x <- subset(NN.train.data, select = selected_features)
nn.devscaled <- scale(x)
nn.devscaled <- cbind(NN.train.data[1], nn.devscaled)

# ---------------------------------- NEURAL NETWORK MODEL ----------------------------------
set.seed(143)
nn2 <- neuralnet(Default ~ ., data=nn.devscaled, hidden=c(10), act.fct="logistic",
                 err.fct='ce', linear.output=FALSE, lifesign="minimal")

# Plot the neural network
plot(nn2)

# ---------------------------------- MODEL PERFORMANCE ----------------------------------
NN.train.data$Prob <- nn2$net.result[[1]]

# Probability distribution in training set
hist(NN.train.data$Prob)

# Function to calculate deciles
decile <- function(x) {
  deciles <- quantile(x, probs = seq(0.1, 1, 0.1), na.rm = TRUE)
  return(findInterval(x, deciles) + 1)
}

NN.train.data$deciles <- decile(NN.train.data$Prob)

# Ranking model results
tmp_DT <- data.table(NN.train.data)
rank <- tmp_DT[, .(cnt = .N, cnt_resp = sum(Default == 1), cnt_non_resp = sum(Default == 0)), by = deciles]
rank <- rank[order(-deciles)]
rank$rrate <- percent(rank$cnt_resp / rank$cnt)
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- percent(rank$cum_resp / sum(rank$cnt_resp))
rank$cum_rel_non_resp <- percent(rank$cum_non_resp / sum(rank$cnt_non_resp))
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp)

print(rank)

# Assign class 0/1 based on probability threshold
NN.train.data$Class <- ifelse(NN.train.data$Prob > 0.5, 1, 0)
with(NN.train.data, table(Default, as.factor(Class)))

# Calculate error
error <- sum((NN.train.data$Default - NN.train.data$Prob)^2) / 2
print(error)

# ---------------------------------- MODEL VALIDATION ----------------------------------
pred3 <- prediction(NN.train.data$Prob, NN.train.data$Default)
perf3 <- performance(pred3, "tpr", "fpr")
plot(perf3)

KS3 <- max(attr(perf3, 'y.values')[[1]] - attr(perf3, 'x.values')[[1]])
auc3 <- as.numeric(performance(pred3, "auc")@y.values)
gini3 <- ineq(NN.train.data$Prob, type="Gini")

print(list(AUC = auc3, KS = KS3, Gini = gini3))

# ---------------------------------- TEST SET SCORING ----------------------------------
x_test <- subset(NN.test.data, select = selected_features)
x_test_scaled <- scale(x_test)
compute.output <- compute(nn2, x_test_scaled)
NN.test.data$Predict.score <- compute.output$net.result



########################################################################################
#-----------------------------------------------NEURAL NETWORK WITH CLUSTER-------------
########################################################################################

# Copy dataset
processed_data <- data_clusters

# Convert categorical variables to factors
processed_data$Region <- as.factor(processed_data$Region)
processed_data$Industry <- as.factor(processed_data$Industry)
processed_data$Corporate_network <- as.factor(processed_data$Corporate_network)
processed_data$cluster <- as.factor(processed_data$cluster)

# One-hot encoding for categorical variables
industry_encoded <- model.matrix(~ 0 + as.factor(processed_data$Industry))
region_encoded <- model.matrix(~ 0 + as.factor(processed_data$Region))
corporate_dummy <- model.matrix(~ 0 + as.factor(processed_data$Corporate_network))
cluster_dummy <- model.matrix(~ 0 + as.factor(processed_data$cluster))

# Assign custom names to encoded variables
colnames(industry_encoded) <- c("Industry_1", "Industry_2", "Industry_3")
colnames(region_encoded) <- c("Region_1", "Region_2", "Region_3")
colnames(corporate_dummy) <- c("Corporate_no", "Corporate_yes")
colnames(cluster_dummy) <- c("Cluster1", "Cluster2", "Cluster3", "Cluster4", "Cluster5", "Cluster6")

# Create final dataset with encoded variables
processed_data <- cbind(processed_data[, !names(processed_data) %in% c("Industry", "Region", "Corporate_network", "cluster")],
                         industry_encoded, region_encoded, corporate_dummy, cluster_dummy)

# Split dataset into training and testing sets
set.seed(190)
trainIndex <- createDataPartition(processed_data$Default, p=0.7, list=FALSE, times=1)
train_data <- processed_data[trainIndex, ]
test_data  <- processed_data[-trainIndex,]

# Check data distribution
print(table(train_data$Default))
print(table(test_data$Default))
print(prop.table(table(train_data$Default)))
print(prop.table(table(test_data$Default)))

# Plot pie charts for data distribution
par(mfrow=c(1,2))
pie3D(prop.table(table(train_data$Default)), main="Default Vs Non Default in Training Set",
      labels=c("No_Default", "Default"), col = c("Turquoise", "Medium Sea Green"))
pie3D(prop.table(table(test_data$Default)), main='Default Vs Non Default in Testing Set',
      labels=c("No_Default", "Default"), col = c("Aquamarine", "Dark Sea Green"))

# Feature scaling
selected_features <- c("Total_assets", "Shareholders_funds", "Long_term_debt", "Loans", "Turnover",
                        "EBITDA", "Net_income", "Tangible_fixed_assets", "Profit_Loss_after_tax",
                        "Current_liabilities", "Current_assets", "Net_income_on_Total_Assets",
                        "Leverage", "Tangible_on_total_assets", "Industry_1", "Industry_2", "Industry_3",
                        "Region_1", "Region_2", "Region_3", "Corporate_no", "Corporate_yes",
                        "Cluster1", "Cluster2", "Cluster3", "Cluster4", "Cluster5", "Cluster6")

scaled_data <- scale(subset(train_data, select = selected_features))
nn_train_scaled <- cbind(train_data[1], scaled_data)

# Build Neural Network Model
nn_model <- neuralnet(Default ~ ., data = nn_train_scaled, hidden = c(10),
                      act.fct = "logistic", err.fct = 'ce', linear.output = FALSE,
                      lifesign = "minimal")

# Plot Neural Network
plot(nn_model)

# Model Performance Metrics
train_data$Prob = nn_model$net.result[[1]]
print(quantile(train_data$Prob, c(0,1,5,10,25,50,75,90,95,98,99,100)/100))

# Histogram of Probabilities
hist(train_data$Prob)

# Decile Calculation Function
decile <- function(x) {
  deciles <- quantile(x, seq(0.1, 1, 0.1), na.rm=TRUE)
  return(findInterval(x, deciles) + 1)
}

# Apply Decile Function
train_data$deciles <- decile(train_data$Prob)

# Ranking Analysis
library(data.table)
library(scales)
rank_data <- data.table(train_data)[, .(cnt = .N, cnt_resp = sum(Default==1), cnt_non_resp = sum(Default==0)), by=deciles]
rank_data <- rank_data[order(-deciles)]
rank_data[, rrate := round(cnt_resp / cnt, 2)]
rank_data[, `:=`(cum_resp = cumsum(cnt_resp), cum_non_resp = cumsum(cnt_non_resp))]
rank_data[, `:=`(cum_rel_resp = round(cum_resp / sum(cnt_resp), 2), cum_rel_non_resp = round(cum_non_resp / sum(cnt_non_resp), 2))]
rank_data[, ks := abs(cum_rel_resp - cum_rel_non_resp)]
rank_data[, rrate := percent(rrate)]
rank_data[, cum_rel_resp := percent(cum_rel_resp)]
rank_data[, cum_rel_non_resp := percent(cum_rel_non_resp)]
print(rank_data)

# Assign Class based on Probability Threshold
train_data$Class <- ifelse(train_data$Prob > 0.5, 1, 0)
print(table(train_data$Default, as.factor(train_data$Class)))

# Calculate Model Error
error <- sum((train_data$Default - train_data$Prob)^2)/2
print(error)

# Model Evaluation Metrics
library(ROCR)
pred <- prediction(train_data$Prob, train_data$Default)
perf <- performance(pred, "tpr", "fpr")
plot(perf)

KS_stat <- max(attr(perf, 'y.values')[[1]] - attr(perf, 'x.values')[[1]])
auc_value <- performance(pred, "auc")@y.values[[1]]

library(ineq)
gini_value <- ineq(train_data$Prob, type="Gini")

# Print Performance Metrics
print(list(AUC = auc_value, KS = KS_stat, Gini = gini_value))



########################################################################################
#----------------------------------NEURAL NETWORK WITH CENTRALITIES--------------------
########################################################################################

# Load data

data_centralities <- data_with_centralities_

# Convert categorical variables to factors
data_centralities$Default <- as.factor(data_centralities$Default)
data_centralities$Region <- as.factor(data_centralities$Region)
data_centralities$Industry <- as.factor(data_centralities$Industry)
data_centralities$Corporate_network <- as.factor(data_centralities$Corporate_network)

# Standardize numerical features
standardized_df <- data_centralities %>%
  mutate(across(where(is.numeric), ~ (.-mean(.))/sd(.)))
data_centralities <- standardized_df

# One-hot encoding for categorical variables
industry_encoded <- model.matrix(~ 0 + as.factor(data_centralities$Industry))
region_encoded <- model.matrix(~ 0 + as.factor(data_centralities$Region))
corporate_dummy <- model.matrix(~ 0 + as.factor(data_centralities$Corporate_network))

# Assign custom column names
colnames(industry_encoded) <- c("Industry_1", "Industry_2", "Industry_3")
colnames(region_encoded) <- c("Region_1", "Region_2", "Region_3")
colnames(corporate_dummy) <- c("Corporate_no", "Corporate_yes")

# Create training dataframe with encoded variables
data_centralities <- cbind(data_centralities[, !names(data_centralities) %in% c("Industry", "Region", "Corporate_network")],
                           industry_encoded, region_encoded, corporate_dummy)

# Convert target variable to numeric
data_centralities$Default  <- as.numeric(as.character(data_centralities$Default))

# Split into training and testing sets
set.seed(295)
trainIndex_cen <- createDataPartition(data_centralities$Default, p=0.7, list = FALSE, times = 1)
NN.train.data.cen <- data_centralities[trainIndex_cen, ]
NN.test.data.cen  <- data_centralities[-trainIndex_cen,]

# Check data distribution
table(NN.train.data.cen$Default)
table(NN.test.data.cen$Default)
prop.table(table(NN.train.data.cen$Default))
prop.table(table(NN.test.data.cen$Default))

# Visualize data distribution
par(mfrow=c(1,2))

pie3D(prop.table(table(NN.train.data.cen$Default)),
      main="Default Vs Non Default in Training Data Set",
      labels=c("No_Default", "Default"),
      col = c("Turquoise", "Medium Sea Green")
)

pie3D(prop.table(table(NN.test.data.cen$Default)),
      main='Default Vs Non Default in Testing Data Set',
      labels=c("No_Default", "Default"),
      col = c("Aquamarine", "Dark Sea Green")
)

# Scale selected features
x <- subset(NN.train.data.cen, select = c(
  "Total_assets", "Shareholders_funds", "Long_term_debt", "Loans", "Turnover", "EBITDA",
  "Net_income", "Tangible_fixed_assets", "Profit_Loss_after_tax", "Current_liabilities",
  "Current_assets", "Net_income_on_Total_Assets", "Leverage", "Tangible_on_total_assets",
  "degree_centrality", "closeness_centrality", "betweenness_centrality",
  "Industry_1", "Industry_2", "Industry_3", "Region_1", "Region_2", "Region_3",
  "Corporate_no", "Corporate_yes"
))

nn.devscaled.cen <- scale(x)
nn.devscaled.cen <- cbind(NN.train.data.cen[1], nn.devscaled.cen)

# Build the Neural Network Model
set.seed(334)
nn2 <- neuralnet(formula = Default ~
                   Total_assets + Shareholders_funds + Long_term_debt + Loans + Turnover + EBITDA +
                   Net_income + Tangible_fixed_assets + Profit_Loss_after_tax + Current_liabilities +
                   Current_assets + Net_income_on_Total_Assets + Leverage + Tangible_on_total_assets +
                   degree_centrality + closeness_centrality + betweenness_centrality + Industry_1 +
                   Industry_2 + Industry_3 + Region_1 + Region_2 + Region_3 + Corporate_no + Corporate_yes,
                 data = nn.devscaled.cen,
                 hidden = c(10),
                 act.fct = "logistic",
                 err.fct = 'ce',
                 linear.output = FALSE,
                 lifesign = "minimal"
)

# Plot the neural network structure
plot(nn2)

# Evaluate model performance
NN.train.data.cen$Prob = nn2$net.result[[1]]

# Probability distribution
quantile(NN.train.data.cen$Prob, c(0,1,5,10,25,50,75,90,95,98,99,100)/100)

# Histogram of probabilities
hist(NN.train.data.cen$Prob)

# Compute deciles
decile <- function(x) {
  deciles <- vector(length=10)
  for (i in seq(0.1,1,.1)){
    deciles[i*10] <- quantile(x, i, na.rm=T)
  }
  return (ifelse(x<deciles[1], 1,
                  ifelse(x<deciles[2], 2,
                         ifelse(x<deciles[3], 3,
                                ifelse(x<deciles[4], 4,
                                       ifelse(x<deciles[5], 5,
                                              ifelse(x<deciles[6], 6,
                                                     ifelse(x<deciles[7], 7,
                                                            ifelse(x<deciles[8], 8,
                                                                   ifelse(x<deciles[9], 9, 10)))))))))))
}

NN.train.data.cen$deciles <- decile(NN.train.data.cen$Prob)

# Install required packages
install.packages("data.table")
library(data.table)
library(scales)

# Compute ranking metrics
tmp_DT = data.table(NN.train.data.cen)
rank <- tmp_DT[, list(
  cnt = length(Default),
  cnt_resp = sum(Default==1),
  cnt_non_resp = sum(Default == 0)) ,
  by=deciles][order(-deciles)]
rank$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp)
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)
rank

# Assign class based on probability threshold
NN.train.data.cen$Class = ifelse(NN.train.data.cen$Prob>0.5,1,0)
with( NN.train.data.cen, table(Default, as.factor(Class) ))

# Compute model error
sum((NN.train.data.cen$Default - NN.train.data.cen$Prob)^2)/2

# Other performance measures
library(ROCR)
pred.cen <- prediction(NN.train.data.cen$Prob, NN.train.data.cen$Default)
perf.cen <- performance(pred.cen, "tpr", "fpr")
plot(perf.cen)

KS.cen <- max(attr(perf.cen, 'y.values')[[1]]-attr(perf.cen, 'x.values')[[1]])
KS.cen

auc.cen <- performance(pred.cen,"auc")
auc.cen <- as.numeric(auc.cen@y.values)
auc.cen
