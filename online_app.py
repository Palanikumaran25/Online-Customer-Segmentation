import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Online Customer Segmentation App:")

def load_data():
    df = pd.read_csv("C:/Python/Online Customer Segmentation/Online Retail.xlsx - Online Retail.csv", encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0] 
    return df 

df = load_data()  

with st.expander("📄 Raw Data"):
    st.write(df) 
    
df['TotalPrice'] = df["Quantity"] * df["UnitPrice"]
 
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) 

snapshot_date = df['InvoiceDate'].max() 

rfm = df.groupby("CustomerID").agg({ 
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, 
    'InvoiceNo': 'nunique', 
    'TotalPrice': 'sum'
})
rfm.columns = ["Recency", "Frequency", "Monetary"] 

with st.expander("📊 RFM Table"): 
    st.dataframe(rfm.describe())  
    
#Normalize RFM Data
scaler = StandardScaler() 
rfm_scaled = scaler.fit_transform(rfm)  

with st.expander("📊 Data Visualization"):
    st.header("Bar_chart")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=["Recency", "Frequency", "Monetary"]
    )
    st.bar_chart(chart_data)
    
    st.header("Line_chart")
    chart_data = pd.DataFrame( 
        np.random.randn(20,3),
        columns = ["Recency", "Frequency", "Monetary"]
    ) 
    st.line_chart(chart_data) 
    
    st.header("Scatter_chart")
    chart_data = pd.DataFrame( 
        np.random.rand(20,3),
        columns = ["Recency", "Frequency", "Monetary"]
    )
    st.scatter_chart(chart_data) 
    
    st.header("Area_chart")
    chart_data = pd.DataFrame( 
        np.random.rand(20, 3), 
        columns = ["Recency", "Frequency", "Monetary"]
    )
    st.area_chart(chart_data)
    
     
#Choose number of clusters 
with st.sidebar:
    k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=4)

kmeans = KMeans(n_clusters=k, random_state= 42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)   


with st.expander("📈 Cluster Summary"):
    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Cluster': 'count'
    }).rename(columns={'Cluster': 'Count'})
    
    st.dataframe(cluster_summary)





                                    




