import pandas as pd
from scripts.data_processing import create_rfm_features, cluster_customers
import pytest

# Sample test data
sample_data = pd.DataFrame({
    'CustomerId': ['C1', 'C1', 'C2', 'C3', 'C3', 'C3'],
    'TransactionStartTime': pd.to_datetime([
        '2025-06-01', '2025-06-10', '2025-05-20',
        '2025-04-01', '2025-04-15', '2025-04-30'
    ]),
    'Amount': [100, 200, 150, 50, 75, 60]
})

def test_create_rfm_features():
    rfm = create_rfm_features(sample_data, snapshot_date=pd.to_datetime('2025-06-15'))
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns
    assert rfm.shape[0] == 3  # 3 customers

def test_cluster_customers():
    rfm = create_rfm_features(sample_data, snapshot_date=pd.to_datetime('2025-06-15'))
    clustered = cluster_customers(rfm)
    assert 'cluster' in clustered.columns
    assert clustered['cluster'].nunique() <= 3
