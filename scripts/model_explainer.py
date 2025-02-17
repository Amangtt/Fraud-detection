import shap
import lime
from lime import lime_tabular
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ModelExplainer:
    def __init__(self, model, X_test):

        self.model = model
        self.X_test = preprocess_data(X_test)

    def explain_with_shap(self, instance_idx=0):
      
        print("Generating SHAP explanations...")

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)

        # SHAP Summary Plot
        plt.figure(figsize=(15, 4))
        shap.summary_plot(shap_values, self.X_test)
        plt.title('SHAP Summary Plot')

        # Handle base value correctly for binary/multi-class classification
        if isinstance(explainer.expected_value, list):
            base_value = explainer.expected_value[1]  # Choose the correct class index
            instance_shap_values = shap_values[1][instance_idx]  # Use the same class index
        else:
            base_value = explainer.expected_value
            instance_shap_values = shap_values[instance_idx]

        # Ensure the correct shape for force plot
        shap.plots.force(base_value, instance_shap_values, features=self.X_test.iloc[instance_idx], feature_names=self.X_test.columns)

        # SHAP Dependence Plot (for the first feature)
        shap.dependence_plot(self.X_test.columns[0], shap_values[1] if isinstance(shap_values, list) else shap_values, self.X_test)

    def explain_with_lime(self, instance_idx=0):
      
        print("Generating LIME explanations...")

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.X_test.values,
            feature_names=self.X_test.columns.tolist(),
            mode='classification'
        )

        # Select a single instance
        instance = self.X_test.iloc[instance_idx].values.flatten()

        explanation = explainer.explain_instance(instance, self.model.predict_proba)

        # Display the LIME explanation
        explanation.show_in_notebook()
        explanation.as_pyplot_figure()
        plt.title(f'LIME Feature Importance for Instance {instance_idx}')
        plt.show()


def preprocess_data(X):
 
    X = X.copy()
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)  # Convert boolean to integer
    X = X.select_dtypes(include=['number'])  # Keep only numeric columns
    X = X.fillna(X.median())  # Fill missing values with median
    return X


def main():
    input_path = './Data/preprocessed/final_fraud.csv'
    model_path = './model/fraud_rf_model.pkl'

    print("Loading data...")
    df = pd.read_csv(input_path)
    target_column = 'Class' if 'Class' in df.columns else 'class'
    X = df.drop(columns=target_column)
    y = df[target_column]
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    print("Loading model...")
    model = joblib.load(model_path)

    explainer = ModelExplainer(model, X_test)

    print("Explaining model with SHAP...")
    explainer.explain_with_shap(instance_idx=0)

    print("Explaining model with LIME...")
    explainer.explain_with_lime(instance_idx=0)


if __name__ == "__main__":
    main()
