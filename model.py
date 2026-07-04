import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_and_predict(df):
    """
    Trains a RandomForest model on the dataset and returns predictions.
    """
    # Separate features and target
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' not found in dataset")
        
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Encode categorical columns using pandas get_dummies
    X_encoded = pd.get_dummies(X)
    
    # Initialize and train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_encoded, y)
    
    # Get predictions
    predictions = model.predict(X_encoded)
    
    return predictions
