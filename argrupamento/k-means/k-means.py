import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

base_path = r'C:\Users\jorge\OneDrive\Documentos\GitHub\Sistemas-Inteligentes\argrupamento\k-means'

customers = pd.read_csv(base_path + r'\olist_customers_dataset.csv')
orders = pd.read_csv(base_path + r'\olist_orders_dataset.csv')
order_items = pd.read_csv(base_path + r'\olist_order_items_dataset.csv')
payments = pd.read_csv(base_path + r'\olist_order_payments_dataset.csv')
reviews = pd.read_csv(base_path + r'\olist_order_reviews_dataset.csv')

df = orders.merge(payments, on='order_id', how='left')
df = df.merge(order_items[['order_id', 'price', 'freight_value']], on='order_id', how='left')
df = df.merge(reviews[['order_id', 'review_score']], on='order_id', how='left')
df = df.merge(customers[['customer_id','customer_city','customer_state']], on='customer_id', how='left')

X = df[['payment_value', 'price', 'freight_value', 'review_score']].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

cluster_mean = df.groupby('cluster')[['payment_value','price','freight_value','review_score']].mean().round(2)
print(cluster_mean)

def nomear_grupo(row):
    if row['review_score'] < 2:
        return 'Pedidos Problematicos'
    elif row['payment_value'] > 1000:
        return 'Pedidos Premium'
    elif row['price'] < 120 and row['review_score'] > 4:
        return 'Pedidos Simples'
    else:
        return 'Pedidos Intermediarios'

nome_clusters = {}
for i in cluster_mean.index:
    temp = cluster_mean.loc[i]
    nome_clusters[i] = nomear_grupo(temp)

df['perfil'] = df['cluster'].map(nome_clusters)

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='payment_value', y='price', hue='perfil', palette='viridis', s=50)
plt.title('Clusters de Pedidos - K-Means')
plt.xlabel('Valor pago (R$)')
plt.ylabel('Preço do produto (R$)')
plt.legend(title='Perfil')
plt.show()

print(df[['order_id','payment_value','price','review_score','cluster','perfil']].head(10))
