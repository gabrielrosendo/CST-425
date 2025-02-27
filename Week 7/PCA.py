import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Generate synthetic simulated data
np.random.seed(42)
n_samples = 100
n_features = 10
data = np.random.rand(n_samples, n_features) * 100  # Simulated sensor data

# Introduce anomalies
n_anomalies = 5
anomalies = np.random.rand(n_anomalies, n_features) * 300  # Extreme values
data[:n_anomalies] = anomalies

# Convert to DataFrame
df = pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(n_features)])

# Preprocessing: Standardizing the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Applying PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
df_pca = pca.fit_transform(df_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

# Anomaly detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_labels = iso_forest.fit_predict(df_pca)

# Plotting the PCA results with anomalies highlighted
plt.figure(figsize=(8,6))
plt.scatter(df_pca[:,0], df_pca[:,1], c=anomaly_labels, cmap='coolwarm', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection with Anomaly Detection")
plt.colorbar(label="Anomaly Score")
plt.show()
