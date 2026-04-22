import io
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Kaggle Cancer Dataset
def load_and_prep_cancer_arff(filepath):
    """Loads and standardizes ARFF biological datasets."""
    with open(filepath, 'r') as f:
            lines = [line.replace("'ALL '", "'ALL'").replace("ALL ", "ALL") for line in f]
    clean_content = io.StringIO("".join(lines))
    data, _ = arff.loadarff(clean_content)
    df = pd.DataFrame(data)
    
    target_col = df.columns[-1]
    
    # Handle byte decoding if necessary
    if isinstance(df[target_col].iloc[0], bytes):
        target_series = df[target_col].str.decode('utf-8').str.strip()
    else:
        target_series = df[target_col]
        
    y = LabelEncoder().fit_transform(target_series)
    y = y - np.mean(y) # Center target
    
    X = StandardScaler().fit_transform(df.drop(columns=[target_col]))
    
    return X, y 