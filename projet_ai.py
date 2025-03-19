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

        pipeline_dir = os.path.join(self.current_working_directory, "pipeline")
        os.makedirs(pipeline_dir, exist_ok=True)

        joblib.dump(self.full_pipeline, os.path.join(pipeline_dir, "full_pipeline.pkl"))
        print(f"Pipeline enregistré dans {pipeline_dir}/full_pipeline.pkl")

    def transform_data(self):
        self.full_pipeline.fit(self.X_train)
        joblib.dump(self.full_pipeline, os.path.join(self.current_working_directory, 'full_pipeline.pkl'))
        self.out_train = self.full_pipeline.transform(self.X_train)
        self.out_test = self.full_pipeline.transform(self.X_test)
    
    def train_models(self):
        self.models['tree'] = DecisionTreeClassifier()
        self.models['forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['perceptron'] = Perceptron(eta0=0.001, max_iter=10000, penalty='l2', alpha=0.0001)
        
        for name, model in self.models.items():
            model.fit(self.out_train, self.y_train)
            joblib.dump(model, os.path.join(self.current_working_directory, f'{name}.model'))
    
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

        # Vérifier que les colonnes attendues sont bien présentes
        df = df[self.numerical_columns + self.categorical_columns]

        # Appliquer le mapping pour JobSatisfaction
        mapping = {
            "Low": 1,
            "Medium": 2,
            "High": 3,
            "Very High": 4
        }
        if "JobSatisfaction" in df.columns:
            df["JobSatisfaction"] = df["JobSatisfaction"].map(mapping).fillna(0).astype(int)

        # Transformation des nouvelles données
        new_data_prepared = self.full_pipeline.transform(df)

        # Prédictions avec les modèles chargés
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(new_data_prepared).tolist()
            print(f'Prédictions avec {name}.model:', predictions[name])

        return predictions


if __name__ == "__main__":
    data_path = "C:/Users/hugol/Downloads/data_HR.xlsx"
    model = AttritionModel(data_path)
    model.load_data()
    model.split_data()
    model.build_pipeline()
    model.transform_data()
    model.train_models()
    model.evaluate_models()
    
    new_data_path = "C:/Users/hugol/Downloads/add_data.xlsx"
    model.predict_new_data(new_data_path)