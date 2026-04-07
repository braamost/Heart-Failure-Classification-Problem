import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def _load_data(data_path):
    """
    Load the heart failure dataset.
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    return pd.read_csv(data_path)

def _encode_categorical(df):
    """
    Encode categorical features using one-hot encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with categorical features encoded
    """
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    return df_encoded

def _split_data(df):
    """
    Split the dataset into train, validation, and test sets with stratified sampling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    # First split: 70% train, 30% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Second split: 10% val, 20% test 
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def _standardize_features(X_train, X_val, X_test):
    """
    Standardize numerical features using StandardScaler.
    
    Args:
        X_train, X_val, X_test (pd.DataFrame): Feature matrices
        
    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val_scaled[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def prepare_data(data_path):
    """
    Prepare the heart failure dataset for modeling.
    
    This function loads the data, encodes categorical features, splits into train/val/test,
    and standardizes numerical features.
    
    Args:
        data_path (str): Path to the heart.csv file
        
    Returns:
        dict: Dictionary containing processed data splits and scaler
            {
                'X_train': pd.DataFrame,
                'X_val': pd.DataFrame, 
                'X_test': pd.DataFrame,
                'y_train': pd.Series,
                'y_val': pd.Series,
                'y_test': pd.Series,
                'scaler': StandardScaler
            }
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    df = _load_data(data_path)
    
    # Encode categorical features
    df_encoded = _encode_categorical(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(df_encoded)
    
    # Standardize features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = _standardize_features(
        X_train, X_val, X_test
    )
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler
    }
