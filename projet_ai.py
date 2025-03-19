import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import os

class AttritionModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.current_working_directory = os.path.dirname(__file__)
        self.numerical_columns = []
        self.categorical_columns = []
        self.full_pipeline = None
        self.models = {}
    
    def load_data(self):
        df = pd.read_excel(self.data_path)
        self.X = df.drop(columns=['EmployeeNumber', 'Attrition', 'Over18', 'EmployeeCount'], axis=1)
        self.y = df['Attrition'].map({'Yes': 1, 'No': 0})
        self.numerical_columns = self.X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.categorical_columns = self.X.select_dtypes(include=['object']).columns.tolist()
        
    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
    
    def build_pipeline(self):
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.numerical_columns = self.X.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = self.X.select_dtypes(include=['object']).columns.tolist()

        self.full_pipeline = ColumnTransformer([
            ('num', num_pipeline, self.numerical_columns),
            ('cat', cat_pipeline, self.categorical_columns)
        ])

    def transform_data(self):
        self.full_pipeline.fit(self.X_train)
        pipeline_dir = os.path.join(self.current_working_directory, "pipeline")
        os.makedirs(pipeline_dir, exist_ok=True)

        joblib.dump(self.full_pipeline, os.path.join(pipeline_dir, "full_pipeline.pkl"))
        joblib.dump(self.full_pipeline, os.path.join(self.current_working_directory, 'full_pipeline.pkl'))
        self.out_train = self.full_pipeline.transform(self.X_train)
        self.out_test = self.full_pipeline.transform(self.X_test)
    
    def train_decision_tree(self):
        dt_model = DecisionTreeClassifier()
        dt_model.fit(self.out_train, self.y_train)
        model_dir = os.path.join(self.current_working_directory, "model")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(dt_model, os.path.join(model_dir, "DecisionTree.model"))
        self.models["DecisionTree"] = dt_model
        print(f"Modèle DecisionTree sauvegardé dans {model_dir}/DecisionTree.model")

    def train_random_forest(self):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.out_train, self.y_train)
        model_dir = os.path.join(self.current_working_directory, "model")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(rf_model, os.path.join(model_dir, "RandomForest.model"))
        self.models["RandomForest"] = rf_model
        print(f"Modèle RandomForest sauvegardé dans {model_dir}/RandomForest.model")

    def train_perceptron(self):
        perceptron_model = Perceptron(eta0=0.001, max_iter=10000, penalty='l2', alpha=0.0001)
        perceptron_model.fit(self.out_train, self.y_train)
        model_dir = os.path.join(self.current_working_directory, "model")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(perceptron_model, os.path.join(model_dir, "Perceptron.model"))
        self.models["Perceptron"] = perceptron_model
        print(f"Modèle Perceptron sauvegardé dans {model_dir}/Perceptron.model")

    def train_models(self):
        self.train_decision_tree()
        self.train_random_forest()
        self.train_perceptron()

    def evaluate_models(self):
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.out_test)
            results[name] = {
                'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1 Score': f1_score(self.y_test, y_pred)
            }
        results_df = pd.DataFrame(results).T
        print(results_df)
        return results_df
        
    def predict_new_data(self, new_data_path):
        df = pd.read_excel(new_data_path)

        pipeline_path = os.path.join(self.current_working_directory, "pipeline", "full_pipeline.pkl")
        self.full_pipeline = joblib.load(pipeline_path)

        df = df[self.numerical_columns + self.categorical_columns]

        mapping = {
            "Low": 1,
            "Medium": 2,
            "High": 3,
            "Very High": 4
        }
        if "JobSatisfaction" in df.columns:
            df["JobSatisfaction"] = df["JobSatisfaction"].map(mapping).fillna(0).astype(int)

        new_data_prepared = self.full_pipeline.transform(df)
        model_dir = os.path.join(self.current_working_directory, "model")
        models_to_load = ["DecisionTree", "RandomForest", "Perceptron"]

        self.models = {name: joblib.load(os.path.join(model_dir, f"{name}.model")) for name in models_to_load}

        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(new_data_prepared).tolist()
            print(f'Prédictions avec {name}.model:', predictions[name])

        return predictions

    def load_and_predict(self, model_name, new_data_path):

        model_dir = os.path.join(self.current_working_directory, "model")
        model_path = os.path.join(model_dir, f"{model_name}.model")

        if not os.path.exists(model_path):
            print(f"Le modèle {model_name} n'existe pas dans le répertoire des modèles.")
            return None

        model = joblib.load(model_path)
        print(f"Modèle {model_name} chargé à partir de {model_path}")

        df = pd.read_excel(new_data_path)
        df = df[self.numerical_columns + self.categorical_columns]

        mapping = {
            "Low": 1,
            "Medium": 2,
            "High": 3,
            "Very High": 4
        }
        if "JobSatisfaction" in df.columns:
            df["JobSatisfaction"] = df["JobSatisfaction"].map(mapping).fillna(0).astype(int)

        new_data_prepared = self.full_pipeline.transform(df)

        predictions = model.predict(new_data_prepared).tolist()

        return predictions

if __name__ == "__main__":
    data_path = "data/data_HR.xlsx"
    model = AttritionModel(data_path)
    model.load_data()
    model.split_data()
    model.build_pipeline()
    model.transform_data()

    new_data_path = "data/add_data.xlsx"
    model_name = "RandomForest"
    predictions = model.load_and_predict(model_name, new_data_path)
    
    if predictions:
        print(f"Prédictions du modèle {model_name}: {predictions}")
