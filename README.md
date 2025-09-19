# Will it rain tomorrow in Australia?

Ovaj projekat koristi mašinsko učenje za predikciju da li će padati kiša sutra u Australiji, koristeći javno dostupni skup podataka o vremenskim prilikama.

---

## 1. Analiza podataka

- Učitavanje i opis skupa podataka (broj instanci, atributa, tipovi, ciljna promenljiva `RainTomorrow`)
- Analiza i vizualizacija nedostajućih vrednosti (tabela, heatmap)
- Statistički pregled numeričkih i kategorijalnih promenljivih (srednja vrednost, std, raspodela, broj jedinstvenih vrednosti)
- Vizualizacije: histogrami, boxplot, korelaciona matrica, parne relacije
- Detekcija i prikaz balansa klasa (za klasifikaciju)
- Obrazloženje izbora najrelevantnijih karakteristika

> **Pokretanje analize:**
> 
> Pokrenuti Jupyter Notebook `1_data_analysis.ipynb`:
> ```bash
> jupyter notebook 1_data_analysis.ipynb
> ```

---

## 2. Pipeline za treniranje, validaciju i testiranje modela

- Podela podataka na trening, validacioni i test skup (70/15/15)
- Pretprocesiranje: imputacija, standardizacija, enkodovanje, autoregresivne kolone za XGBoost
- Modeli:
    - **Facebook Prophet** (time series analiza)
    - **XGBoost** (sa autoregresijom)
- GridSearchCV za tuning hiperparametara (XGBoost)
- Evaluacija: accuracy, f1-score, ROC-AUC, classification_report, vizualizacija Prophet predikcije
- Čuvanje najboljeg modela i pipeline-a (`.pkl` za XGBoost, Prophet format za Prophet)

> **Pokretanje pipeline-a:**
> 
> ```bash
> python 2_train_pipeline.py
> ```

---

## 3. Streamlit web aplikacija

- Web aplikacija za unos podataka i dobijanje predikcije (XGBoost pipeline)
- Prikaz rezultata: DA/NE i verovatnoća padavina za sutra
- Prikaz Prophet predikcije za narednih 7 dana (grafikon)
- Validacija unosa i prikaz greške ako model nije dostupan

> **Pokretanje aplikacije:**
> 
> ```bash
> streamlit run 3_app.py
> ```

---

## Primer ulaza/izlaza

- **Ulaz:** Forma sa podacima o vremenskim prilikama (temperatura, vlažnost, pritisak, vetar, lokacija, mesec, dan...)
- **Izlaz:** Predikcija (DA/NE) i verovatnoća padavina za sutrašnji dan, plus grafikon Prophet predikcije

---

## Instalacija zavisnosti

Pre pokretanja instalirati potrebne biblioteke:

```bash
pip install -r requirements.txt
```

---

## Tehnička napomena

- Projekat je modularan: svaki deo (analiza, treniranje, aplikacija) je u posebnom fajlu
- Modeli i pipeline se mogu ponovo učitati i koristiti za nove predikcije
- Za pokretanje je potrebna Python 3.8+ i instalirane zavisnosti
- Potrebno je imati fajlove: `weatherAUS_rainfall_prediction_dataset_cleaned.csv`, `best_xgboost_pipeline.pkl`, `best_prophet_model`

---

## Kontakt

Za pitanja i konsultacije, obratiti se asistentu ili autoru projekta.
