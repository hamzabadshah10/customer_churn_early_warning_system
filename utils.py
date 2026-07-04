import pandas as pd
import kagglehub
import os

def get_clean_data():
    # Download and load
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    df = pd.read_csv(os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    
    # Drop irrelevant column
    df = df.drop(columns=['customerID'])
    
    # Fix missing numbers in TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Convert Yes/No to 1/0 for Churn and other columns
    # We will use simple mapping for the target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df