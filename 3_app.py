import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title='Will it rain tomorrow in Australia?', layout='centered')
st.title('🌧️ Will it rain tomorrow in Australia?')

# ----------------------------
#  UČITAVANJE MODELA
# ----------------------------
@st.cache_resource
def load_xgb_model():
    if os.path.exists('best_xgboost_pipeline.pkl'):
        return joblib.load('best_xgboost_pipeline.pkl')
    return None

xgb_model = load_xgb_model()

prophet_loaded = False
prophet_model = None
try:
    prophet_model = joblib.load('best_prophet_model.pkl')
    prophet_loaded = True
except Exception as e:
    st.warning(f'Greška pri učitavanju Prophet modela: {e}')

# ----------------------------
#  UNOS PODATAKA ZA XGBOOST
# ----------------------------
input_features = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
    'Temp9am', 'Temp3pm', 'Cloud9am', 'Cloud3pm',
    'RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm',
    'Location', 'Month', 'Day', 'Year'
]

st.header('Unesite podatke za XGBoost predikciju:')
user_input = {}
for feat in input_features:
    if feat in ['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Location']:
        # Kategorijalni unos
        if feat == 'RainToday':
            options = ['No', 'Yes']
        elif feat == 'Location':
            options = ['Sydney', 'Melbourne', 'Brisbane', 'Perth',
                       'Adelaide', 'Hobart', 'Darwin', 'Canberra', 'Other']
        else:
            options = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW',
                       'ENE', 'ESE', 'NNE', 'NNW', 'SSE', 'SSW', 'WNW', 'WSW']
        user_input[feat] = st.selectbox(feat, options)
    elif feat == 'Month':
        user_input[feat] = st.number_input('Month', min_value=1, max_value=12,
                                           value=datetime.now().month)
    elif feat == 'Day':
        user_input[feat] = st.number_input('Day', min_value=1, max_value=31,
                                           value=datetime.now().day)
    elif feat == 'Year':
        user_input[feat] = st.number_input('Year', min_value=2000, max_value=2030,
                                           value=datetime.now().year)
    else:
        user_input[feat] = st.number_input(feat, value=0.0)

input_df = pd.DataFrame([user_input])

# ----------------------------
#  PREDIKCIJA XGBOOST
# ----------------------------
if xgb_model is not None:
    try:
        pred = xgb_model.predict(input_df)[0]
        proba = xgb_model.predict_proba(input_df)[0][1]
        st.success(f"Predikcija: {'DA' if pred == 1 else 'NE'} "
                   f"(verovatnoća: {proba:.2f})")
    except Exception as e:
        st.error(f"Greška u predikciji: {e}")

# ----------------------------
#  PROPHET PREDIKCIJA
# ----------------------------
st.markdown('---')
st.header('📈 Prophet prognoza padavina za buduće godine')

if prophet_loaded:
    # Opcija za unos broja godina
    years_ahead = st.slider('Broj godina za predikciju:', min_value=1, max_value=10, value=5)
    
    # Učitaj originalne podatke za istorijski deo
    try:
        df = pd.read_csv('weatherAUS_rainfall_prediction_dataset_cleaned.csv')
        
        # Kreiranje budućih datuma
        future_days = years_ahead * 365
        future = prophet_model.make_future_dataframe(periods=future_days)
        forecast = prophet_model.predict(future)
        
        # Priprema istorijskih podataka
        historical_data = df[['Date', 'RainTomorrow']].copy()
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        historical_data['RainTomorrow'] = (historical_data['RainTomorrow'] == 'Yes').astype(int)
        
        # Kreiranje grafikona
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Istorijski podaci - mesečni prosek
        historical_monthly = historical_data.groupby(historical_data['Date'].dt.to_period('M'))['RainTomorrow'].mean()
        historical_monthly.index = historical_monthly.index.to_timestamp()
        
        # Buduće predikcije - mesečni prosek
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        forecast_monthly = forecast.groupby(forecast['ds'].dt.to_period('M'))['yhat'].mean()
        forecast_monthly.index = forecast_monthly.index.to_timestamp()
        
        # Crtanje istorijskih podataka
        ax.plot(historical_monthly.index, historical_monthly.values, 
                label='Istorijski podaci (mesečni prosek)', color='blue', linewidth=2)
        
        # Crtanje predikcija
        ax.plot(forecast_monthly.index, forecast_monthly.values, 
                label=f'Predikcija za narednih {years_ahead} godina', color='red', linewidth=2, linestyle='--')
        
        # Oznaka trenutnog trenutka
        current_date = historical_data['Date'].max()
        ax.axvline(x=current_date, color='green', linestyle=':', linewidth=2, 
                   label=f'Trenutni trenutak ({current_date.strftime("%Y-%m-%d")})')
        
        ax.set_title(f'Predikcija padavina za narednih {years_ahead} godina\n'
                    f'(Verovatnoća kiše po mesecu)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Datum', fontsize=12)
        ax.set_ylabel('Verovatnoća kiše (0-1)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Dodaj statistike
        historical_avg = historical_monthly.mean()
        future_avg = forecast_monthly[forecast_monthly.index > current_date].mean()
        
        stats_text = f'Istorijski prosek: {historical_avg:.3f}\n'
        stats_text += f'Predviđeni prosek: {future_avg:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        st.pyplot(fig)
        
        # Prikaz dodatnih statistika
        st.subheader('📊 Statistike predikcije')
        future_data = forecast[forecast['ds'] > current_date]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Broj predviđenih dana", len(future_data))
        with col2:
            st.metric("Prosečna verovatnoća kiše", f"{future_data['yhat'].mean():.3f}")
        with col3:
            st.metric("Maksimalna verovatnoća", f"{future_data['yhat'].max():.3f}")
            
    except Exception as e:
        st.error(f'Greška pri učitavanju podataka: {e}')
        st.info('Potrebno je da pokrenete 2_train_pipeline.py da generišete Prophet model.')

else:
    st.info('Prophet model nije dostupan. Pokreni pipeline za treniranje Prophet modela.')
