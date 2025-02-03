# Standardize the variables
data.scale_c <- data.frame(scale(data_c[2:15]))  # Standardizing columns 2 to 15 of the dataset

# Summary of standardized data (Check if mean is 0 after scaling)
summary(data.scale_c)

# Variance of standardized data (Check if diagonal is 1 after scaling)
var(data.scale_c)

# Define a metric to compute the relative distance between companies
# Applying the Euclidean distance between each pair of companies' feature vectors
dist = as.matrix(dist(scale(data.scale_c)))

# Define the graph based on the distance matrix (Undirected, weighted graph)
g2 = graph_from_adjacency_matrix(dist, mode = "undirected", weighted = TRUE)

# Find the Minimum Spanning Tree (MST) of the graph g
g2_mst = mst(g2)

# Plot the MST
V(g2_mst)$Default = data$Default  # Add default status as a vertex attribute
V(g2_mst)[Default == 1]$color = "firebrick1"  # Color defaulted companies in red
V(g2_mst)[Default == 0]$color = "lightgreen"  # Color active companies in green

# Set random seed for reproducibility and plot the MST
set.seed(424)
plot(g2_mst, graph = "nsca",  # Plotting MST using the "nsca" layout
     vertex.label = NA,
     vertex.size = 5,
     main = "MST of Italian Small-Medium Enterprises Network")

# Step 1: Calculate the correlations between variables
cor_network <- cor_auto(data.scale_c)

# Plot Partial Correlation Network (Visualize relationships between variables)
Graph_2 <- qgraph(cor_network, graph = "pcor", layout = "spring", edge.width = 1)
summary(Graph_2)

# Partial Correlation Network with significance threshold (alpha = 0.05)
Graph_3 <- qgraph(cor_network, graph = "pcor", layout = "spring", edge.width = 1, threshold = "sig",
                  sampleSize = nrow(data.scale_c), alpha = 0.05)
summary(Graph_3)

# Investigate centrality measures of the graph
centralities_Graph3 <- centrality(Graph_3)

# Plot centrality measures (Strength and Closeness) using z-scores
centralityPlot(Graph_3, include = c("Strength", "Closeness"), scale = "z-scores")

# Add centrality measures to the dataset
data_with_centralities <- cbind(data.scale_c,
                                centralities_Graph3$degree,
                                centralities_Graph3$closeness,
                                centralities_Graph3$betweenness)

# Compute degree centrality, closeness centrality, and betweenness centrality for MST
degree_centrality <- degree(g2_mst)
closeness_centrality <- closeness(g2_mst)
betweenness_centrality <- betweenness(g2_mst)

# Create a dataframe with centrality measures
centralities <- data.frame(degree_centrality, closeness_centrality, betweenness_centrality)

# Add the centrality measures to the original dataset
data_with_centralities <- cbind(data_c, centralities)

# Save the dataset to desktop
# Set the path for saving the dataset on the desktop
percorso_salvataggio <- "~/Desktop/nome_file.csv"

# Save the dataset as a CSV file
write.csv(data_with_centralities, file = percorso_salvataggio, row.names = FALSE)

# Print a confirmation message
cat("The file has been successfully saved on the desktop.\n")
