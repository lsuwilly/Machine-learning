import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import *

# Use columns 0 up to and including 6 (Nitrogen to Rainfall)
crop_data = pandas.read_csv("../data/Crop_recommendation.csv", header = 0, usecols = range(8))

# Get different selections with and without label, and also one 
# that is just y
crop_data_no_classes = crop_data.iloc[:, 0:7]
crop_data_with_classes = crop_data.iloc[:, :]
crop_data_classes = crop_data.iloc[:,7]

# Convert crop data to matrix for further processing
matrix = crop_data_no_classes.values
matrix_classes = crop_data_classes.values

# print(crop_data_no_classes)

kmeans = KMeans(n_clusters = 5, random_state=0)
kmeans.fit(matrix)

kmeans_labels = kmeans.labels_
# for label in labels:
#     print(label)


crop_data_with_labels = crop_data
crop_data_with_labels['kmeans clusters'] = kmeans_labels
#print(crop_data_with_labels)

# Now, we visualize this new labeled data

x_col = 0
y_col = 1

plt.scatter(matrix[:, x_col], matrix[:, y_col], c=kmeans_labels, cmap='viridis')
plt.xlabel(f'{crop_data.columns[x_col]}')
plt.ylabel(f'{crop_data.columns[y_col]}')
plt.title(f'KMeans Clustering Results')
plt.show()

# Finally, we add the classes back on, so now we can graph coloring labels
# from kmeans, dbscan, and the classes
crop_data_with_labels["class"] = crop_data_classes

crop_data_with_labels.to_csv("../data/output.csv")

print(crop_data_with_labels)



# We need to group by each label, and then see what classes compose each
# label

classes_by_label = crop_data_with_labels.groupby(['kmeans clusters', 'class']).size().reset_index(name = 'number of samples')

# Let's get the fractions of each class that compose each label, and graph
# them.
grouped_clusters = classes_by_label.groupby(["kmeans clusters", "class"])["number of samples"].sum().reset_index()

total_samples_per_cluster = grouped_clusters.groupby('kmeans clusters')['number of samples'].transform('sum')

# The fraction here is the total fraction of each class which composes each
# cluster / label
grouped_clusters["fraction"] = grouped_clusters["number of samples"] / total_samples_per_cluster

# print(grouped_clusters)

fig, axes = plt.subplots(len(grouped_clusters['kmeans clusters'].unique()), 1, figsize=(40, 20))

for i, cluster in enumerate(grouped_clusters['kmeans clusters'].unique()):
    # Get the data for the current cluster
    cluster_data = grouped_clusters[grouped_clusters['kmeans clusters'] == cluster]
    
    # Plot pie chart
    axes[i].pie(cluster_data['fraction'], labels=cluster_data['class'], autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'Cluster {cluster}')
    
plt.tight_layout()
plt.show()