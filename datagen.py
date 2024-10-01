import pandas as pd
import numpy as np

# Create synthetic customer data
np.random.seed(42)
data = {
    'CustomerID': np.arange(1, 101),
    'Recency': np.random.randint(1, 100, 100),
    'Frequency': np.random.randint(1, 20, 100),
    'Monetary': np.random.uniform(100, 1000, 100),
    'CLV': np.random.uniform(500, 20000, 100)  # Customer Lifetime Value
}

df = pd.DataFrame(data)
df.to_csv('clv_dataset.csv', index=False)
