import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import xgboost as xgb
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')
import sklearn
from packaging import version
from sklearn.preprocessing import OrdinalEncoder

# 1. Učitavanje podataka
def load_data(path):
    return pd.read_csv(path)

# 2. Podela na train/val/test
def split_data(df, target, test_size=0.15, val_size=0.15, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size+val_size), random_state=random_state, stratify=y)
    rel_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=rel_val_size, random_state=random_state, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Dalji koraci: pretprocesiranje, treniranje modela, evaluacija, cuvanje modela

def preprocess_pipeline(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    # Posebno tretiraj 'Location' (ordinal encoding)
    location_cols = [col for col in cat_cols if col == 'Location']
    other_cat_cols = [col for col in cat_cols if col != 'Location']
    
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    if version.parse(sklearn.__version__) >= version.parse('1.2'):
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', encoder)
    ])
    location_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    transformers = [
        ('num', num_transformer, num_cols)
    ]
    if other_cat_cols:
        transformers.append(('cat', cat_transformer, other_cat_cols))
    if location_cols:
        transformers.append(('loc', location_transformer, location_cols))
    preprocessor = ColumnTransformer(transformers)
    return preprocessor

# Autoregresivne kolone za XGBoost (lag features)
def add_lag_features(df, target, lags=[1,2,3]):
    df = df.copy()
    for lag in lags:
        df[f'{target}_lag{lag}'] = df[target].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

# Evaluacija modela
def evaluate_classification(y_true, y_pred, y_proba=None):
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('F1 score:', f1_score(y_true, y_pred, pos_label=1))
    if y_proba is not None:
        print('ROC-AUC:', roc_auc_score(y_true, y_proba[:,1]))
    print('\nClassification report:')
    print(classification_report(y_true, y_pred))

def train_xgboost(X_train, y_train, X_val, y_val, preprocessor):
    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('xgb', xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            learning_rate=0.05,
            max_depth=3,
            n_estimators=50
        ))
    ])
    pipe.fit(X_train, y_train)
    print('Model treniran sa najboljim parametrima: learning_rate=0.05, max_depth=3, n_estimators=50')
    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)
    evaluate_classification(y_val, y_pred, y_proba)
    return pipe

# Prophet model (time series, predviđa RainTomorrow po datumu)
def train_prophet(df, date_col='Date', target='RainTomorrow'):
    prophet_df = df[[date_col, target]].copy()
    prophet_df = prophet_df.rename(columns={date_col: 'ds', target: 'y'})
    # Pretvaranje cilja u binarni (1=Yes, 0=No)
    prophet_df['y'] = (prophet_df['y'] == 'Yes').astype(int)
    model = Prophet()
    model.fit(prophet_df)
    return model

def create_rainfall_prediction_plot(df, prophet_model, years_ahead=5):
    """
    Kreira grafik koji prikazuje istorijske podatke o padavinama i predikcije za buduće godine
    """
    # Priprema istorijskih podataka
    historical_data = df[['Date', 'RainTomorrow']].copy()
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    historical_data['RainTomorrow'] = (historical_data['RainTomorrow'] == 'Yes').astype(int)
    
    # Kreiranje budućih datuma (5 godina unapred)
    future_days = years_ahead * 365
    future = prophet_model.make_future_dataframe(periods=future_days)
    forecast = prophet_model.predict(future)
    
    # Kreiranje grafikona
    plt.figure(figsize=(15, 8))
    
    # Istorijski podaci - mesečni prosek
    historical_monthly = historical_data.groupby(historical_data['Date'].dt.to_period('M'))['RainTomorrow'].mean()
    historical_monthly.index = historical_monthly.index.to_timestamp()
    
    # Buduće predikcije - mesečni prosek
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    forecast_monthly = forecast.groupby(forecast['ds'].dt.to_period('M'))['yhat'].mean()
    forecast_monthly.index = forecast_monthly.index.to_timestamp()
    
    # Crtanje istorijskih podataka
    plt.plot(historical_monthly.index, historical_monthly.values, 
             label='Istorijski podaci (mesečni prosek)', color='blue', linewidth=2)
    
    # Crtanje predikcija
    plt.plot(forecast_monthly.index, forecast_monthly.values, 
             label=f'Predikcija za narednih {years_ahead} godina', color='red', linewidth=2, linestyle='--')
    
    # Oznaka trenutnog trenutka
    current_date = historical_data['Date'].max()
    plt.axvline(x=current_date, color='green', linestyle=':', linewidth=2, 
                label=f'Trenutni trenutak ({current_date.strftime("%Y-%m-%d")})')
    
    plt.title(f'Predikcija padavina za narednih {years_ahead} godina\n'
              f'(Verovatnoća kiše po mesecu)', fontsize=16, fontweight='bold')
    plt.xlabel('Datum', fontsize=12)
    plt.ylabel('Verovatnoća kiše (0-1)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Dodaj statistike
    historical_avg = historical_monthly.mean()
    future_avg = forecast_monthly[forecast_monthly.index > current_date].mean()
    
    stats_text = f'Istorijski prosek: {historical_avg:.3f}\n'
    stats_text += f'Predviđeni prosek: {future_avg:.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.show()
    
    return forecast

def main():
    # Učitavanje podataka
    df = load_data('weatherAUS_rainfall_prediction_dataset_cleaned_balanced.csv')
    target = 'RainTomorrow'
    
    # Kreiranje kopije za XGBoost (bez Date kolone)
    df_xgb = df.drop(columns=['Date']) if 'Date' in df.columns else df
    
    # Podela na train/val/test za XGBoost
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_xgb, target)
    # Pretprocesiranje
    preprocessor = preprocess_pipeline(X_train)
    # XGBoost sa autoregresijom
    X_train_lag = add_lag_features(pd.concat([X_train, y_train], axis=1), target)
    y_train_lag = (X_train_lag[target] == 'Yes').astype(int)  # KONVERZIJA U 0/1
    X_train_lag = X_train_lag.drop(columns=[target])
    X_val_lag = add_lag_features(pd.concat([X_val, y_val], axis=1), target)
    y_val_lag = (X_val_lag[target] == 'Yes').astype(int)      # KONVERZIJA U 0/1
    X_val_lag = X_val_lag.drop(columns=[target])
    print('--- XGBoost ---')
    best_xgb = train_xgboost(X_train_lag, y_train_lag, X_val_lag, y_val_lag, preprocessor)
    # Evaluacija na test skupu
    X_test_lag = add_lag_features(pd.concat([X_test, y_test], axis=1), target)
    y_test_lag = (X_test_lag[target] == 'Yes').astype(int)    # KONVERZIJA U 0/1
    X_test_lag = X_test_lag.drop(columns=[target])
    y_pred_test = best_xgb.predict(X_test_lag)
    y_proba_test = best_xgb.predict_proba(X_test_lag)
    print('--- XGBoost Test ---')
    evaluate_classification(y_test_lag, y_pred_test, y_proba_test)
    joblib.dump(best_xgb, 'best_xgboost_pipeline.pkl')
    
    # Prophet - koristi originalni df sa Date kolonom
    print('--- Prophet - Predikcija padavina za buduće godine ---')
    prophet_model = train_prophet(df, date_col='Date', target=target)
    
    # Kreiranje predikcije za 5 godina unapred
    forecast = create_rainfall_prediction_plot(df, prophet_model, years_ahead=5)
    
    print('Čuvanje Prophet modela...')
    joblib.dump(prophet_model, 'best_prophet_model.pkl')
    print('Prophet deo završen!')
    
    # Prikaz dodatnih statistika
    print('\n--- Statistike predikcije ---')
    future_data = forecast[forecast['ds'] > df['Date'].max()]
    print(f'Broj predviđenih dana: {len(future_data)}')
    print(f'Prosečna verovatnoća kiše u budućnosti: {future_data["yhat"].mean():.3f}')
    print(f'Maksimalna verovatnoća kiše: {future_data["yhat"].max():.3f}')
    print(f'Minimalna verovatnoća kiše: {future_data["yhat"].min():.3f}')

if __name__ == '__main__':
    main()
