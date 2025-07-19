import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Loading data from a file
df = pd.read_excel('licencjat.xlsx', sheet_name='dane_standaryzowane')

print(df.head())  # Wyświetli pierwsze kilka wierszy danych
print(df.columns)  # Wyświetli nazwy kolumn

# Selection of variables for clustering (standardized)
features = [
    'płeć', 'standaryzowany śr. Koszyk', 'standaryzowana intensywność',
    'subs. kawowa', 'subs. Maszyn.', 'original', 'vertuo', 'hybryd'
]
X = df[features]

# Elbow method and silhouette score
inertia = []
silhouette_scores = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    
    if k > 1:
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(None)

# Elbow method chart
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Liczba klastrów (k)')
plt.ylabel('Inercja')
plt.title('Wykres łokcia')
plt.grid(True)
plt.show()

# Silhouette score chart
plt.figure(figsize=(8, 5))
plt.plot(k_range[1:], silhouette_scores[1:], marker='o', color='orange')
plt.xlabel('Liczba klastrów (k)')
plt.ylabel('Silhouette Score')
plt.title('Wykres Silhouette Score')
plt.grid(True)
plt.show()

# Stability test: multiple k-means actuation and repeatability assessment
def test_stabilnosci(X, k, n_runs=30):
    labels_runs = []
    for i in range(n_runs):
        kmeans = KMeans(n_clusters=k, random_state=None, n_init=10)
        labels = kmeans.fit_predict(X)
        labels_runs.append(labels)
    # Calculation of pairs of matching indices between subsequent runs
    ari_scores = []
    for i in range(n_runs - 1):
        score = adjusted_rand_score(labels_runs[i], labels_runs[i+1])
        ari_scores.append(score)
    print(f"Stabilność klasteryzacji dla k={k}:")
    print(f"Średni Adjusted Rand Index (30 par kolejnych uruchomień): {np.mean(ari_scores):.3f}")
    print(f"Min ARI: {np.min(ari_scores):.3f}, Max ARI: {np.max(ari_scores):.3f}")

# Stability test example for k=3
test_stabilnosci(X, k=3)
# Stability test example for k=4
test_stabilnosci(X, k=4)


# Final model with k=3
k_opt = 3
kmeans_final = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
df['segment'] = kmeans_final.fit_predict(X)

# Cluster profiling
print("Profilowanie klastrów:")
numerical_features = ['standaryzowany śr. Koszyk', 'standaryzowana intensywność']
cluster_means = df.groupby('segment')[numerical_features].mean()
print(cluster_means)

# Gender distribution (in %)
print("\nProcentowa dystrybucja płci w klastrach:")
gender_dist = df.groupby('segment')['płeć'].value_counts(normalize=True).unstack() * 100
print(gender_dist)

# Distribution of binary/subscription features
binary_features = ['subs. kawowa', 'subs. Maszyn.', 'original', 'vertuo', 'hybryd']
print("\nŚrednia wartość cech binarnych (procent w klastrze):")
print(df.groupby('segment')[binary_features].mean() * 100)

# Cluster visualization using PCA
pca = PCA(n_components=2)
components = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
colors = sns.color_palette("Set1", k_opt)
for i in range(k_opt):
    subset = components[df['segment'] == i]
    plt.scatter(subset[:, 0], subset[:, 1],
                label=f'Segment {i}',
                color=colors[i], alpha=0.7)

plt.title('Separowalność segmentów klientów – PCA (k=4)')
plt.xlabel('Główna składowa 1')
plt.ylabel('Główna składowa 2')
plt.legend(title='Segment')
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplots for numerical features
for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='segment', y=feature, palette='pastel')
    plt.title(f'{feature} wg segmentu')
    plt.show()

# Comparison of average feature values between segments
numerical_features = ['standaryzowany śr. Koszyk']
binary_features = ['płeć', 'subs. kawowa', 'subs. Maszyn.', 'original', 'vertuo', 'hybryd']

# Average values of numerical characteristics
mean_numerical = df.groupby('segment')[numerical_features].mean()
print("Średnie wartości cech numerycznych wg segmentów:")
print(mean_numerical)

# Percentage distribution of binary variables and categories
mean_binary = df.groupby('segment')[binary_features].mean() * 100
print("\nProcentowa wartość zmiennych binarnych wg segmentów:")
print(mean_binary)

# Visualization of average numerical features
mean_numerical.plot(kind='bar', figsize=(10,6))
plt.title('Średnie wartości cech numerycznych wg segmentów')
plt.ylabel('Średnia wartość')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

# Update segment labels and the "gender" -> "men" variable
segment_labels = ['original', 'vertuo', 'hybryd']
category_labels = ['mężczyźni', 'subs. kawowa', 'subs. Maszyn.', 'original', 'vertuo', 'hybryd']

# Creating two charts with new labels
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# First chart: men, subs. coffee, subs. Machines.
for i, kat in enumerate(category_labels[:3]):
    axs[0].bar(x + i * width, wartosci[i], width, label=kat)

axs[0].set_title('Płeć i subskrypcje (%)')
axs[0].set_xticks(x + width)
axs[0].set_xticklabels(segment_labels)
axs[0].set_xlabel('Segment')
axs[0].set_ylabel('Procent (%)')
axs[0].legend()

# Second chart: original, vertuo, hybrid
for i, kat in enumerate(category_labels[3:]):
    axs[1].bar(x + i * width, wartosci[i + 3], width, label=kat)

axs[1].set_title('Typ systemu (%)')
axs[1].set_xticks(x + width)
axs[1].set_xticklabels(segment_labels)
axs[1].set_xlabel('Segment')
axs[1].legend()

fig.suptitle('Przeciętny udział zmiennych binarnych wg segmentów (%)')
plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.show()

# Exact percentage sex distribution by segment:
gender_dist = df.groupby('segment')['płeć'].value_counts(normalize=True).unstack() * 100
print("\nDokładny procentowy rozkład płci wg segmentów:")
print(gender_dist)

# Gender visualization on a bar chart
gender_dist.plot(kind='bar', stacked=True, figsize=(8,6), colormap='Pastel1')
plt.title('Procentowy rozkład płci wg segmentów')
plt.ylabel('Procent (%)')
plt.xlabel('Segment')
plt.legend(title='Płeć', labels=['Kobieta', 'Mężczyzna'])
plt.grid(axis='y')
plt.show()

# Dendrogram for tuning the number of clusters using hierarchical grouping
linked = linkage(X, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=df.index.values,
           distance_sort='descending',
           show_leaf_counts=False)
plt.title('Dendrogram - Hierarchiczne grupowanie klientów')
plt.xlabel('Indeksy klientów')
plt.ylabel('Odległość')
plt.show()

