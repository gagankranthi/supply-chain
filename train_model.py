import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

EXCEL_PATH = "Supply chain logistics problem.xlsx"
MODEL_PATH = "model.joblib"


def simple_preprocess(df):
    # Basic preprocessing: pick numeric columns and dropna
    num = df.select_dtypes(include=['number']).copy()
    if num.shape[1] == 0:
        # try to convert any columns that look numeric
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        num = df.select_dtypes(include=['number']).copy()
    num = num.dropna()
    return num


if __name__ == '__main__':
    df = pd.read_excel(EXCEL_PATH)
    data = simple_preprocess(df)
    if data.shape[1] < 2:
        print('Not enough numeric features to train a model. Columns found:', data.columns)
    else:
        # Use last column as target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        # Choose regressor or classifier
        if y.nunique() > 20 and y.dtype.kind in 'fc':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            print('Model type: Classifier')
            print('Accuracy:', accuracy_score(y_test, preds))
        else:
            print('Model type: Regressor')
            rmse = mean_squared_error(y_test, preds)
            import math
            print('RMSE:', math.sqrt(rmse))
        joblib.dump(model, MODEL_PATH)
        print('Saved model to', MODEL_PATH)
