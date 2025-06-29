import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

#  1. Load Data 
def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up one level from 'scripts'
    file_path = os.path.join(base_dir, 'data', 'raw', 'data.csv')
    df = pd.read_csv(file_path)
    return df

    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

#  2. RFM Feature Generation 
def create_rfm_features(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    }).reset_index()
    
    return rfm

# 3. KMeans Clustering
def scale_features(df: pd.DataFrame, features: list) -> np.ndarray:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    return scaled, scaler

def assign_risk_clusters(rfm_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    features = ['Recency', 'Frequency', 'Monetary']
    scaled_features, scaler = scale_features(rfm_df, features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['Cluster'] = kmeans.fit_predict(scaled_features)

    return rfm_df, kmeans, scaler

def define_high_risk_cluster(rfm_df: pd.DataFrame) -> int:
    summary = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()

    summary['RiskScore'] = (summary['Recency'].rank(ascending=False) +
                            summary['Frequency'].rank(ascending=True) +
                            summary['Monetary'].rank(ascending=True))

    return int(summary.sort_values('RiskScore', ascending=False).iloc[0]['Cluster'])

def add_proxy_target(df: pd.DataFrame, rfm_df: pd.DataFrame, high_risk_cluster: int) -> pd.DataFrame:
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    df = df.merge(rfm_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
    df['is_high_risk'] = df['is_high_risk'].fillna(0).astype(int)
    return df

# 4. Feature Engineering 
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
    df['Transaction_Day'] = df['TransactionStartTime'].dt.day
    df['Transaction_Month'] = df['TransactionStartTime'].dt.month
    df['Transaction_Year'] = df['TransactionStartTime'].dt.year

    agg_df = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'std', 'count'],
        'Transaction_Hour': 'nunique'
    }).reset_index()

    agg_df.columns = ['CustomerId', 'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Num_Transactions', 'Unique_Hours']

    df = pd.merge(df, agg_df, on='CustomerId', how='left')
    return df

#  5. Preprocessing Pipeline 
def create_pipeline(df: pd.DataFrame):
    num_cols = ['Total_Amount', 'Avg_Amount', 'Std_Amount', 'Num_Transactions', 'Transaction_Hour']
    cat_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    return full_pipeline

#  6. Save Processed Data 
def save_processed_data(df: pd.DataFrame, path='data/processed/processed_data.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

#  7. Main Execution 
if __name__ == "__main__":
    df = load_data("data/raw/data.csv")

    rfm = create_rfm_features(df)
    rfm_clustered, _, _ = assign_risk_clusters(rfm)
    high_risk = define_high_risk_cluster(rfm_clustered)
    df = add_proxy_target(df, rfm_clustered, high_risk)

    df = feature_engineering(df)

    pipeline = create_pipeline(df)
    X = pipeline.fit_transform(df)
    y = df['is_high_risk']