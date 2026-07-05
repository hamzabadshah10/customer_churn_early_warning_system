import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def train_and_predict(df):
    """
    Trains a RandomForest model on the dataset, evaluates it on a 20% test set,
    prints the metrics, and returns predictions for the entire dataset for UI display.
    """
    # Separate features and target
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' not found in dataset")
        
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Encode categorical columns using pandas get_dummies
    X_encoded = pd.get_dummies(X)
    
    # 1. Split data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # 2. Initialize and train model on the TRAINING set
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Evaluate model on the TESTING set
    y_test_pred = model.predict(X_test)
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    # Print Metrics to Console
    print("\n" + "="*45)
    print("   MODEL PERFORMANCE ON 20% TEST SET")
    print("="*45)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}% (When model predicts churn, it is correct this often)")
    print(f"Recall:    {recall * 100:.2f}% (Out of all true churners, model caught this many)")
    print("\nConfusion Matrix:")
    print("                  Predicted Stay(0) | Predicted Churn(1)")
    print("-" * 58)
    print(f"Actual Stay(0) :       {conf_matrix[0][0]:<12} | {conf_matrix[0][1]}")
    print(f"Actual Churn(1):       {conf_matrix[1][0]:<12} | {conf_matrix[1][1]}")
    print("="*45 + "\n")
    
    # 4. Get predictions for the ENTIRE dataset so the GUI table can be fully populated
    predictions = model.predict(X_encoded)
    
    # 5. Extract top features driving churn
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top_features = feature_importances.sort_values(ascending=False).head(3).to_dict()
    
    return predictions, top_features
