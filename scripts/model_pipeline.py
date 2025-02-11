import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import argparse
import joblib
import mlflow
#import tensorflow as tf
#from xgboost import XGBClassifier


def load_data(input_path):
    df = pd.read_csv(input_path)
    target_column = 'Class' if 'Class' in df.columns else 'class'
    x = df.drop(columns=target_column)
    y = df[target_column]
    return x,y
def train_test(x,y, test_size=0.2, random_state=42):
    return train_test_split(x,y,test_size=test_size,random_state=random_state)

def train_random_forest(x_train,y_train):
    model=RandomForestClassifier(n_estimators=100, max_depth=5,random_state=42)
    model.fit(x_train,y_train)
    joblib.dump(model, './model/fraud_rf_model.pkl')
    return model

def train_logistic_reg(x_train,y_train):
    model=LogisticRegression()
    model.fit(x_train,y_train)
    joblib.dump(model, './model/fraud_lr_model.pkl')
    return model

def train_Decison_tree(x_train,y_train):
    model=DecisionTreeClassifier()
    model.fit(x_train,y_train)
    joblib.dump(model, './model/fraud_dt_model.pkl')
    return model

def evaluate(model,x_test,y_test):
    y_pred=model.predict(x_test)
    test_acc=accuracy_score(y_pred,y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Testing Accuracy: {test_acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    return test_acc, precision, recall, f1

def main():
    input_path='./Data/preprocessed/final_fraud.csv'
    x,y= load_data(input_path)
    X_train,X_test,y_train,y_test=train_test(x,y)
    mlflow.set_experiment('fraud detection')
    models = {
        "RandomForest": train_random_forest,
        "LogisticRegression": train_logistic_reg,
        "DecisionTree": train_Decison_tree,
        
    }
    
    for model_name, train_function in models.items():
        with mlflow.start_run():
            print(f"Training {model_name}...")
            model = train_function(X_train, y_train)
            test_acc, precision, recall, f1 = evaluate(model, X_test, y_test)
            mlflow.log_param("model", model_name)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, f"{model_name}_model")
if __name__=='__main__':
    main()