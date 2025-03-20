import os
os.system("cls")
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc

class AttritionModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.current_working_directory = os.path.dirname(__file__)
        self.numerical_columns = []
        self.categorical_columns = []
        self.full_pipeline = None
        self.models = {}
        self.tab_mean = []
    
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
        self.out_train = self.full_pipeline.transform(self.X_train)
        self.out_test = self.full_pipeline.transform(self.X_test)
    
    def train_decision_tree(self):
        dt_model = DecisionTreeClassifier()
        dt_model.fit(self.out_train, self.y_train)
        mean_score = self.cross_validate_model(dt_model)
        self.tab_mean.append(mean_score)
        model_dir = os.path.join(self.current_working_directory, "model")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(dt_model, os.path.join(model_dir, "DecisionTree.model"))
        self.models["DecisionTree"] = dt_model
        #print(f"Modèle DecisionTree sauvegardé dans {model_dir}/DecisionTree.model avec un score moyen de {mean_score}")

    def train_random_forest(self):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.out_train, self.y_train)
        mean_score = self.cross_validate_model(rf_model)
        self.tab_mean.append(mean_score)
        model_dir = os.path.join(self.current_working_directory, "model")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(rf_model, os.path.join(model_dir, "RandomForest.model"))
        self.models["RandomForest"] = rf_model
        #print(f"Modèle random forest sauvegardé dans {model_dir}/RandomForest.model avec un score moyen de {mean_score}")

    def train_perceptron(self):
        perceptron_model = Perceptron(eta0=0.001, max_iter=10000, penalty='l2', alpha=0.0001)
        perceptron_model.fit(self.out_train, self.y_train)
        mean_score = self.cross_validate_model(perceptron_model)
        self.tab_mean.append(mean_score)
        model_dir = os.path.join(self.current_working_directory, "model")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(perceptron_model, os.path.join(model_dir, "Perceptron.model"))
        self.models["Perceptron"] = perceptron_model
        #print(f"Modèle Perceptron sauvegardé dans {model_dir}/Perceptron.model avec un score moyen de {mean_score}")

    def re_train_perceptron(self):
        model_dir = os.path.join(self.current_working_directory, "model")
        if os.path.isdir(os.path.join(model_dir, "Perceptron_retrain.model")):
            pathto_perceptron_model = os.path.join(model_dir, "Perceptron_retrain.model")
        else:
            pathto_perceptron_model = os.path.join(model_dir, "Perceptron.model")
        perceptron_model = joblib.load( pathto_perceptron_model)
        perceptron_model.partial_fit(self.out_train, self.y_train)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(perceptron_model, os.path.join(model_dir, "Perceptron_retrain.model"))
        self.models["Perceptron"] = perceptron_model
        #print(f"Modèle Perceptron sauvegardé dans {model_dir}/Perceptron_retrain.model")

    def train_models(self):
        self.train_decision_tree()
        self.train_random_forest()
        self.train_perceptron()

    def cross_validate_model(self, model, cv=5):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.out_train, self.y_train, cv=kf, scoring='accuracy')
        #print(f"Scores K-Fold Cross-Validation : {scores}")
        #print(f"Score moyen : {np.mean(scores)}")
        return np.mean(scores)

    def evaluate_models(self):
        results = {}
        i = 0
        for name, model in self.models.items():
            y_pred = model.predict(self.out_test)

            if name == 'Perceptron':
                self.y_probas = model.decision_function(self.out_test)
            else:
                self.y_probas = model.predict_proba(self.out_test)
                self.y_probas = self.y_probas[:, 1]

            self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_probas)
            results[name] = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'AUC':auc(self.fpr, self.tpr),
                'F1 Score': f1_score(self.y_test, y_pred),
                'Conf matrix': confusion_matrix(self.y_test, y_pred),
                'mean_cross_validation':  self.tab_mean[i]
            }
            i=+1

        results_df = pd.DataFrame(results).T
        return results_df

    def load_and_predict(self, model_name, new_data_path):

        model_dir = os.path.join(self.current_working_directory, "model")
        model_path = os.path.join(model_dir, f"{model_name}.model")

        if not os.path.exists(model_path):
            print(f"Le modèle {model_name} n'existe pas dans le répertoire des modèles.")
            return None

        model = joblib.load(model_path)
       # print(f"Modèle {model_name} chargé à partir de {model_path}")

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
    model.train_models()
    print("")
    tab_evaluation = model.evaluate_models()
    print(tab_evaluation)
    print("")
    new_data_path = "data/add_data.xlsx"
    models_name = ["RandomForest", "Perceptron", "DecisionTree"]
    
    for model_name in models_name:
        predictions = model.load_and_predict(model_name, new_data_path)
        print(f"Prédictions du modèle {model_name}: {predictions}")

def launch_test():
    new_data_path = "data/add_data.xlsx"
    models_name = ["Perceptron"] # ["RandomForest", "Perceptron", "DecisionTree"]
    
    for model_name in models_name:
        predictions = model.load_and_predict(model_name, new_data_path)
        print(f"Prédictions du modèle {model_name}: {predictions}")

    model.re_train_perceptron()
    models_name = ["Perceptron"] # ["RandomForest", "Perceptron", "DecisionTree"]
    
    for model_name in models_name:
        predictions = model.load_and_predict(model_name, new_data_path)
        print(f"Prédictions du modèle {model_name}: {predictions}")
