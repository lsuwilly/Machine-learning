import pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import *

from src.knn import *

def find_cluster(new_data):
    # Use columns 0 up to and including 6 (Nitrogen to Rainfall)
    crop_data = pandas.read_csv("./data/Crop_recommendation.csv", header = 0, usecols = range(8))

    # Get different selections with and without label for clustering
    # and also one that is just the label
    crop_data_no_classes = crop_data.iloc[:, 0:7]

    # Convert crop data to matrix for further processing
    matrix = crop_data_no_classes.values



    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)

    # Experimentally, we have discovered that the optimal
    # number of clusters for our data is around 5

    kmeans = KMeans(n_clusters = 5, random_state=0)
    kmeans.fit(matrix_scaled) # Fit crop data without classes onto k means

    kmeans_labels = kmeans.labels_



    # Make new crop data df that also includes clustering labels
    crop_data_with_labels = crop_data
    crop_data_with_labels['kmeans clusters'] = kmeans_labels

    # Now, we visualize this new clustered data
    x_col = 0
    y_col = 1

    # Plot N x P , with the kmeans labels as the colors to show relationship
    plt.scatter(matrix[:, x_col], matrix[:, y_col], c=kmeans_labels, cmap='viridis')
    plt.xlabel(f'{crop_data.columns[x_col]}')
    plt.ylabel(f'{crop_data.columns[y_col]}')
    plt.title(f'KMeans Clustering Results')
    plt.show()

    # We predict the class of some experimental data with new clustering
    # information

    # Predict and show cluster
    new_data_label = kmeans.predict(new_data)

    print("CLUSTER :",new_data_label[0]) # New label for our sample

    # Predict class with knn with added k means filtering
    sample_prediction = knn_predict(crop_data_with_labels, new_data, new_data_label[0])



    # We need to group by each label, and then see what classes compose each
    # label

    classes_by_label = crop_data_with_labels.groupby(['kmeans clusters', 'label']).size().reset_index(name = 'number of samples')

    # Let's get the fractions of each class that compose each label, and graph
    # them.
    grouped_clusters = classes_by_label.groupby(["kmeans clusters", "label"])["number of samples"].sum().reset_index()

    total_samples_per_cluster = grouped_clusters.groupby('kmeans clusters')['number of samples'].transform('sum')

    # The fraction here is the total fraction of each class which composes each
    # cluster / label
    grouped_clusters["fraction"] = grouped_clusters["number of samples"] / total_samples_per_cluster



    # Now, display pie chart using fractions data
    fig, axes = plt.subplots(len(grouped_clusters['kmeans clusters'].unique()), 1, figsize=(40, 20))

    for i, cluster in enumerate(grouped_clusters['kmeans clusters'].unique()):
        # Get the data for the current cluster
        cluster_data = grouped_clusters[grouped_clusters['kmeans clusters'] == cluster]
        
        # Formatting code for displaying labels around pie chart with arrows
        wedges, texts, autotexts = axes[i].pie(cluster_data["fraction"],
                                              labels=cluster_data['label'],
                                              autopct='%1.1f%%',
                                              startangle=90,
                                              pctdistance=0.85,
                                              wedgeprops={'width':0.3, 'edgecolor':'black'})

        for text in autotexts:
            text.set(size=12, color='black')

        for j, wedge in enumerate(wedges):
            angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
            x = wedge.r * 0.75 * plt.np.cos(plt.np.radians(angle))
            y = wedge.r * 0.75 * plt.np.sin(plt.np.radians(angle))
            axes[i].annotate(cluster_data['label'].iloc[i],
                        xy=(x,y),
                        xytext=(1.1 * x, 1.1 * y),
                        arrowprops=dict(facecolor='black', arrowstyle='->', lw=1))

        axes[i].set_title(f'Cluster {cluster}')

    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots()

    cluster_data = grouped_clusters[grouped_clusters['kmeans clusters'] == new_data_label[0]]

    # Plot pie chart specifically for selected chart
    ax.pie(cluster_data['fraction'], labels=cluster_data['label'], autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Distributions of Crops in Similar Fields :')

    return sample_prediction, fig

if __name__ == '__main__':
    new_data = pandas.DataFrame({
        'N': [85],
        'P': [45],
        'K': [40],
        'temperature': [22.0],
        'humidity': [75.0],
        'ph': [6.4],
        'rainfall': [210.0]
    })

    sample_prediction, pie = find_cluster(new_data)

    pie.show()