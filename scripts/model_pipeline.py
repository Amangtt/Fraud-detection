import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import tensorflow as tf
import numpy as np
import os

# Ensure model directory exists
os.makedirs("./model", exist_ok=True)

def load_data(input_path):
    df = pd.read_csv(input_path)

    # Ensure the target column exists
    target_column = 'Class' if 'Class' in df.columns else 'class'
    
    # Separate target from features
    y = df[target_column].astype(np.float32)  # Convert target to float32
    x = df.drop(columns=[target_column])

    # Convert boolean columns to float32
    bool_cols = x.select_dtypes(include=['bool']).columns
    x[bool_cols] = x[bool_cols].astype(np.float32)

    # Ensure all features are float32
    x = x.astype(np.float32)

    return x.to_numpy(), y.to_numpy()
def train_test(x, y, test_size=0.2, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(x_train, y_train)
    joblib.dump(model, './model/fraud_rf_model.pkl')
    return model

def train_logistic_reg(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    joblib.dump(model, './model/fraud_lr_model.pkl')
    return model

def train_decision_tree(x_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    joblib.dump(model, './model/fraud_dt_model.pkl')
    return model

def train_rnn(x_train, y_train, x_test, y_test):
    time_steps = 1  # Each row is treated as a single sequence
    feature_size = x_train.shape[1]

    x_train_rnn = x_train.reshape((x_train.shape[0], time_steps, feature_size))
    x_test_rnn = x_test.reshape((x_test.shape[0], time_steps, feature_size))

    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(50, activation='relu', input_shape=(time_steps, feature_size)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    model.fit(x_train_rnn, y_train, epochs=10, batch_size=32, validation_data=(x_test_rnn, y_test), verbose=1)
    joblib.dump(model, "./model/fraud_rnn_model.pkl")
  
    return model

def train_lstm(x_train, y_train, x_test, y_test):
    time_steps = 1  # Each row is treated as a single sequence
    feature_size = x_train.shape[1]
    x_train_lstm = x_train.reshape((x_train.shape[0], time_steps, feature_size)) # Reshape for LSTM
    x_test_lstm = x_test.reshape((x_test.shape[0], time_steps, feature_size))

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(1, x_train.shape[1])),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    model.fit(x_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(x_test_lstm, y_test), verbose=1)
    joblib.dump(model,"./model/fraud_lstm_model.pkl")
    
    return model

def evaluate(model, x_test, y_test, is_deep_learning=False):
    if is_deep_learning:
        x_test = np.expand_dims(x_test, axis=1)  # Reshape for RNN/LSTM
    
    y_pred = model.predict(x_test)

    if is_deep_learning:
        y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary class labels

    test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Testing Accuracy: {test_acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return test_acc, precision, recall, f1

def main():
    input_path = './Data/preprocessed/final_fraud.csv'
    x, y = load_data(input_path)
  
    X_train, X_test, y_train, y_test = train_test(x, y)

    mlflow.set_experiment('fraud detection')

    models = {
        "RandomForest": (train_random_forest, False),
        "LogisticRegression": (train_logistic_reg, False),
        "DecisionTree": (train_decision_tree, False),
        "RNN": (train_rnn, True),
        "LSTM": (train_lstm, True)
    }

    for model_name, (train_function, is_deep_learning) in models.items():
        with mlflow.start_run():
            print(f"Training {model_name}...")
            model = train_function(X_train, y_train, X_test, y_test) if is_deep_learning else train_function(X_train, y_train)
            test_acc, precision, recall, f1 = evaluate(model, X_test, y_test, is_deep_learning)

            mlflow.log_param("model", model_name)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            if is_deep_learning:
                mlflow.tensorflow.log_model(model, f"{model_name}_model")
            else:
                mlflow.sklearn.log_model(model, f"{model_name}_model")

if __name__ == '__main__':
    main()
