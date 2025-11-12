import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_excel("argrupamento\Online Retail.xlsx")
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['Total'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (ref_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'Total': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

sil_scores = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(rfm_scaled)
    sil = silhouette_score(rfm_scaled, labels)
    sil_scores.append(sil)

plt.plot(K, sil_scores, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette vs Número de Clusters')
plt.show()

best_k = K[sil_scores.index(max(sil_scores))]
print(f'Melhor número de clusters: {best_k}')
print(f'Melhor Silhouette Score: {max(sil_scores):.3f}')

kmeans_final = KMeans(n_clusters=best_k, random_state=42)
rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled)

print(rfm.groupby('Cluster').mean())
