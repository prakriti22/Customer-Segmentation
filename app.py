import os
import pickle

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, render_template, request, url_for
from joblib import load
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')

import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='')

model = load('kmeans_model.joblib')

def load_and_clean_data(file_path):
    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)
    retail['CustomerID'] = retail['CustomerID'].astype(str)

    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']
    rfm = retail.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%m/%d/%Y %H:%M')

    max_date = max(retail['InvoiceDate'])
    retail['Diff'] = max_date - retail['InvoiceDate']
    rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days
    rfm = pd.merge(rfm, rfm_f, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

    numeric_columns = ['Amount', 'Frequency', 'Recency']
    try:
        rfm[numeric_columns] = rfm[numeric_columns].apply(pd.to_numeric, errors='coerce')
    except pd.errors.OverflowError as e:
        print(f"Error converting columns to numeric: {e}")

    rfm = rfm.dropna(subset=numeric_columns)

    if rfm[numeric_columns].dtypes.all() != 'object':
        print("All columns successfully converted to numeric.")

    for column in numeric_columns:
        Q1 = rfm[column].quantile(0.05)
        Q3 = rfm[column].quantile(0.95)
        IQR = Q3 - Q1

        rfm = rfm[(rfm[column] >= Q1 - 1.5 * IQR) & (rfm[column] <= Q3 + 1.5 * IQR)]

    return rfm

def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
    scaler = StandardScaler()
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
    return rfm, rfm_df_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)
    df = preprocess_data(file_path)[1]
    results_df = model.predict(df)
    df_with_id = preprocess_data(file_path)[0]
    df_with_id['Cluster_Id'] = results_df
    sns.stripplot(x='Cluster_Id', y='Amount', data=df_with_id)
    amount_img_path = 'static/ClusterId_Amount.png'
    plt.savefig(amount_img_path)
    plt.close()
    sns.stripplot(x='Cluster_Id', y='Frequency', data=df_with_id)
    freq_img_path = 'static/ClusterId_Frequency.png'
    plt.savefig(freq_img_path)
    plt.close()
    sns.stripplot(x='Cluster_Id', y='Recency', data=df_with_id)
    recen_img_path = 'static/ClusterId_Recency.png'
    plt.savefig(recen_img_path)
    plt.close()
    try:
        response = {
            'amount_img': amount_img_path,
            'freq_img': freq_img_path,
            'recency_img': recen_img_path
        }
        return render_template('result.html', **response)
    except Exception as e:
        return str(e)

@app.route('/get_images')
def get_images():
    images = {
        "amount_img": url_for('static', filename='ClusterId_Amount.png'),
        "freq_img": url_for('static', filename='ClusterId_Frequency.png'),
        "recency_img": url_for('static', filename='ClusterId_Recency.png')
    }
    return images

if __name__ == "__main__":
    app.run(debug=True)
