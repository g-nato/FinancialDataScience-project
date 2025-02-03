#======================= Install and Load Packages ===============================#

# Install necessary packages if not yet installed
install.packages("readxl")
install.packages("caret")
install.packages("ROCR")
install.packages("ppcor")
install.packages("Hmisc")
install.packages("corrplot")
install.packages("tidymodels")

# Load the required libraries
library(readxl)
library(caret)
library(ROCR)
library(ggplot2)
library(corrplot)
library(ppcor)
library(dplyr)
library(tidymodels)

#======================= Clean Environment ===============================#

# Clean up the environment by removing all existing objects
rm(list=ls())

#======================= Load and Prepare Data ===============================#

# Load the dataset 
data <- Dataset2_Companies

# Remove the 'ID' column as it's not needed for the analysis
data$ID <- NULL

# Rename the first column to 'Default', which is the target variable
colnames(data)[1] <- "Default"

# Calculate the default rate (percentage of defaults)
def_perc <- sum(data$Default) / length(data$Default)
print(def_perc)

#======================= Region Aggregation ===============================#

# Create a new variable 'Region' based on Italian regions, assigning numeric values for North, Center, and South
Region <- function(data) {
  data$Region <- ifelse(data$Region %in% c("Valle D'Aosta", "Piemonte", "Liguria", "Lombardia",
                                           "Trentino-Alto Adige", "Veneto", "Friuli-Venezia Giulia",
                                           "Emilia-Romagna"), 1,
                        ifelse(data$Region %in% c("Puglia", "Campania", "Basilicata", "Calabria",
                                                  "Sicilia", "Sardegna"), 3,
                               ifelse(data$Region %in% c("Toscana", "Umbria", "Marche", "Lazio",
                                                         "Abruzzo", "Molise"), 2, NA)))
  
  data$Region <- as.factor(data$Region)  # Convert 'Region' to a factor variable
  return(data)
}

# Apply the Region function to the data
data <- Region(data)

#======================= Industry Aggregation ===============================#

# Create a new 'Industry' variable by mapping the sector codes to numbers
Industry <- function(data) {
  data$Industry <- ifelse(data$Industry %in% c("A", "B"), 1,   # Primary sector -> 1
                          ifelse(data$Industry %in% c("C", "D", "E", "F"), 2,  # Secondary sector -> 2
                                 ifelse(data$Industry %in% c("G", "H", "I", "J", "K", "L", "M", "N", "O",
                                                              "P", "Q", "R", "S", "T", "U"), 3, NA)))  # Tertiary sector -> 3
  
  data$Industry <- as.factor(data$Industry)  # Convert 'Industry' to a factor variable
  return(data)
}

# Apply the Industry function to the data
data <- Industry(data)

#======================= Corporate Network Variable ===============================#

# Create a new variable 'Corporate_network' to identify companies in the same network (1 if duplicated, 0 otherwise)
data$Corporate_network <- ifelse(duplicated(data$Corporate_network) | duplicated(data$Corporate_network, fromLast = TRUE), 1, 0)

#======================= Convert to Factor Variables ===============================#

# Convert 'Default' and 'Corporate_network' columns to factors
data$Default <- as.factor(data$Default)
data$Corporate_network <- as.factor(data$Corporate_network)



#======================= Data Analysis ===============================#

# Assign the processed data to a new variable 'data_c'
data_c <- data

# Check for missing values in each variable
missing_values <- colSums(is.na(data))

# Create a dataframe to store the variables and their respective missing values count
missing_data <- data.frame(variable = names(missing_values), missing_count = missing_values)

# Create a bar plot for the missing values distribution
ggplot(missing_data, aes(x = variable, y = missing_count, fill = variable)) +
  geom_bar(stat = "identity", color = "black") +
  labs(x = "Variable", y = "Number of Missing Values") +
  ggtitle("Distribution of Missing Values by Variable") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme_minimal()


### Bar Plot for 'Default' Variable
ggplot(data, aes(x = factor(Default))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Default",
       x = "Default",
       y = "Frequency") +
  scale_x_discrete(labels = c("0" = "Non-Default", "1" = "Default")) +
  geom_text(stat = "count", aes(label = paste0(sprintf("%.1f", (..count..)/sum(..count..) * 100), "%")),
            vjust = -0.5) +
  theme(text = element_text(size = 22))


### Distribution of Values for 'Industry'
ggplot(data, aes(x = Industry, fill = factor(Default))) +
  geom_bar() +
  labs(x = "Industry", y = "Count", fill = "Default") +
  ggtitle("Distribution of Data by Industry and Default") +
  theme(text = element_text(size = 22))


### Distribution of Values for 'Region'
ggplot(data, aes(x = Region, fill = factor(Default))) +
  geom_bar() +
  labs(x = "Region", y = "Count", fill = "Default") +
  ggtitle("Distribution of Data by Region and Default") +
  theme(text = element_text(size = 22))


### Distribution of Values for 'Corporate Network'
ggplot(data, aes(x = Corporate_network, fill = factor(Default))) +
  geom_bar() +
  labs(x = "Corporate Network", y = "Count", fill = "Default") +
  ggtitle("Distribution of Data by Corporate Network and Default") +
  theme(text = element_text(size = 22))


### Calculate Correlations for Numerical Variables
correlations <- cor(data[, -c(1, 16, 17, 18)]) # Exclude non-numeric variables

# Display correlation plot
corrplot(correlations, method="circle")
corrplot(correlations, method="number")


### Calculate Partial Correlations
p_correlations <- pcor(data[,-c(1, 16, 17, 18)])
pcor_mat       <- p_correlations$estimate

# Display partial correlation plot
corrplot(pcor_mat, method="circle")
corrplot(pcor_mat, method="number")


### Histogram Plots for Each Numeric Variable

# Select only numeric variables
numeric_vars <- sapply(data, is.numeric)

# Filter out non-numeric variables
numeric_data <- data[, numeric_vars]

# Create histogram plots for each numeric variable
histograms <- lapply(names(numeric_data), function(var) {
  ggplot(data, aes(x = .data[[var]])) +
    geom_histogram(binwidth = 1, fill = "steelblue", color = "white") +
    labs(title = paste("Distribution of", var),
         x = var,
         y = "Frequency") +
    theme_minimal() +
    scale_x_continuous(breaks = seq(floor(min(data[[var]])), ceiling(max(data[[var]])), by = 1))
})

# Arrange histograms in a grid layout
grid.arrange(grobs = histograms, ncol = 3)


### Box Plots for Financial Variables (First 15 variables)

boxplots <- lapply(names(data)[1:15], function(var) {
  ggplot(data, aes(x = factor(Default), y = data[[var]])) +
    geom_boxplot(fill = "steelblue") +
    labs(title = paste("Box Plot of", var),
         x = "Default",
         y = var) +
    scale_x_discrete(labels = c("0" = "Active", "1" = "Defaulted"))
})

# Arrange box plots in a grid layout
grid.arrange(grobs = boxplots, ncol = 3)

