#!/usr/bin/env python
# coding: utf-8

import datetime as dt
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

retail = pd.read_csv('data 2.csv', sep=',', encoding='ISO-8859-1', header=0)

retail['CustomerID'] = retail['CustomerID'].astype(str)

retail['Amount'] = retail['Quantity'] * retail['UnitPrice']
rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()
rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
rfm_f.columns = ['CustomerID', 'Frequency']
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%m/%d/%Y %H:%M')

max_date = max(retail['InvoiceDate'])
retail['Diff'] = max_date - retail['InvoiceDate']
rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()
rfm_p['Diff'] = rfm_p['Diff'].dt.days
rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

attributes = ['Amount', 'Frequency', 'Recency']
plt.rcParams['figure.figsize'] = [10, 8]
sns.boxplot(data=rfm[attributes], palette="Set2", whis=1.5, saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize=14, fontweight='bold')
plt.ylabel("Range", fontweight='bold')
plt.xlabel("Attributes", fontweight='bold')

Q1 = rfm.Amount.quantile(0.05)
Q3 = rfm.Amount.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Amount >= Q1 - 1.5 * IQR) & (rfm.Amount <= Q3 + 1.5 * IQR)]

Q1 = rfm.Recency.quantile(0.05)
Q3 = rfm.Recency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Recency >= Q1 - 1.5 * IQR) & (rfm.Recency <= Q3 + 1.5 * IQR)]

Q1 = rfm.Frequency.quantile(0.05)
Q3 = rfm.Frequency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Frequency >= Q1 - 1.5 * IQR) & (rfm.Frequency <= Q3 + 1.5 * IQR)]

rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
scaler = StandardScaler()
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']

kmeans = KMeans(n_clusters=3, max_iter=300)
kmeans.fit(rfm_df_scaled)

rfm['Cluster_Id'] = kmeans.predict(rfm_df_scaled)

filename = 'kmeans_model.pkl'
with open('kmeans_saved_model', 'wb') as file:
    pickle.dump(kmeans, file)
file.close()
pickle.dump(kmeans, open('kmeans_model.pkl', 'wb'))

sns.stripplot(x='Cluster_Id', y='Amount', data=rfm)
plt.savefig("Cluster_Id_Amount.png")
plt.clf()

sns.stripplot(x='Cluster_Id', y='Recency', data=rfm)
plt.savefig("Cluster_Id_Recency.png")
plt.clf()

# Evaluation Metrics
silhouette_avg = silhouette_score(rfm_df_scaled, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)

calinski_harabasz_score_value = calinski_harabasz_score(rfm_df_scaled, kmeans.labels_)
print("Calinski-Harabasz Score:", calinski_harabasz_score_value)

davies_bouldin_score_value = davies_bouldin_score(rfm_df_scaled, kmeans.labels_)
print("Davies-Bouldin Score:", davies_bouldin_score_value)
