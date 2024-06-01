import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Rastgele bir veri seti oluşturma
# np.random.seed(42)
# n_customers = 200
#
# data = {
#     'CustomerID': range(1, n_customers + 1),
#     'Age': np.random.randint(18, 70, size=n_customers),
#     'AnnualIncome': np.random.randint(20000, 150000, size=n_customers),
#     'SpendingScore': np.random.randint(1, 100, size=n_customers)
# }
#
# df = pd.DataFrame(data)
#
# # Veri setini CSV dosyasına kaydetme
# df.to_csv('customer_data.csv', index=False)

# Veri setini yükleme
df = pd.read_csv('customer_data.csv')

# Özellikleri seçme ve standardize etme
features = df[['Age', 'AnnualIncome', 'SpendingScore']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# K-means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_features)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=5)
hierarchical_labels = hierarchical.fit_predict(scaled_features)

# DBSCAN uygulama
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_features)

# Kümeleme sonuçlarını görselleştirme
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=kmeans_labels, cmap='viridis')
plt.title('K-means Clustering')

plt.subplot(1, 3, 2)
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=hierarchical_labels, cmap='viridis')
plt.title('Hierarchical Clustering')

plt.subplot(1, 3, 3)
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')

plt.show()

# Dendrogram oluşturma
linked = linkage(scaled_features, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
