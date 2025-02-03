# Use sapply to list the type of each variable
types <- sapply(data_c, class)
type_table <- as.data.frame(cbind(names, types))

# Select only the columns to be considered as numeric
data_ca <- data_c[, 2:15]
names_num <- colnames(data_ca)

# Descriptive statistics
descr_stat <- as.data.frame(stat.desc(data_ca))

# Use sapply to calculate the standard deviation of all variables
stdev <- sapply(data_ca, sd)
sd_data <- as.data.frame(cbind(names_num, stdev))

# Use sapply to display the skewness
skwn <- sapply(data_ca, skewness)
skwn_data <- as.data.frame(cbind(stdev, skwn))


# Standardize the variables
data.scale <- data.frame(scale(data_ca))

# Display summary to confirm zero mean after standardization
summary(data.scale)

# Check variance, should be 1 after standardization
var(data.scale)

# K-means clustering

set.seed(90)
km_fit <- kmeans(data.scale, 6)  # 6 clusters

# Cluster assignments
km_fit$cluster

# Cluster sizes
km_fit$size

# Function to plot the within-group sum of squares (WGSS) against the number of clusters
wgssplot <- function(data_ca, nc, seed) {
  wgss <- rep(0, nc)
  wgss[1] <- (nrow(data_ca) - 1) * sum(apply(data_ca, 2, var))
  for (i in 2:nc) {
    set.seed(seed)
    wgss[i] <- sum(kmeans(data_ca, centers = i)$withinss)
  }
  plot(1:nc, wgss, type = "b", xlab = "Number of Clusters",
       ylab = "Within-group sum of squares")
}

# How to determine the optimal number of clusters?
# We can observe the explained variance percentage as a function of the number of clusters.
# Plot the WGSS as a function of the number of clusters.

wgssplot(data.scale, nc = 1, seed = 99)

# Is there a relationship between clusters and default probability of companies?

data_clus <- as.data.frame(cbind(data_c, km_fit$cluster))
colnames(data_clus)[19] <- 'cluster'

# Create a contingency table of clusters and default probabilities
clus_def <- table(data_clus$cluster, data_clus$Default)

# Calculate default rates within each cluster
def_table <- clus_def / km_fit$size
def_rates <- def_table[, 2]
def_rates

# Plot clusters
autoplot(km_fit, data.scale, frame = TRUE)

# Save dataset to desktop
# Set the path for saving the dataset
percorso_salvataggio2 <- "~/Desktop/data_clusters.csv"

# Save the dataset as a CSV file
write.csv(data_clus, file = percorso_salvataggio2, row.names = FALSE)

# Print confirmation message
cat("The file has been successfully saved on the desktop.\n")
