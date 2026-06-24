# Laptop Price Predictor

A Streamlit dashboard that predicts laptop prices from specs (brand, RAM, CPU/GPU, storage, display) using a Random Forest regression pipeline, alongside market-insight charts built from the underlying dataset.

>Link for the Deployed Project : https://machinelearningprojects-laptoppricepredictor786.streamlit.app/

## Features

**Predict tab**
- Configure a laptop (brand, type, RAM, weight, CPU, GPU, OS, HDD, SSD, screen size, resolution, touchscreen, IPS) and get an instant price estimate with a ±10% likely range.
- Shows the closest real laptops from the dataset for that brand/type as a sanity check.

**Market Insights tab**
- Average price by brand, type, and CPU
- Price distribution histogram
- Average price vs RAM trend
- Touchscreen / IPS price premium comparison
- Predicted-vs-actual scatter plot with R² and MAE

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Charts | Plotly |
| Model | scikit-learn (`Pipeline` + `ColumnTransformer` + `RandomForestRegressor`) |
| Data wrangling | pandas / numpy |

## Dataset

`data.pkl` — 1,302 laptops, 14 columns:

| Column | Description |
|---|---|
| `Company` | Brand (Apple, HP, Dell, ...) |
| `TypeName` | Notebook, Ultrabook, Gaming, etc. |
| `Inches` | Screen size |
| `Ram` | RAM in GB |
| `Weight` | Weight in kg |
| `Price` | Target variable (₹) |
| `Touchscreen` / `IPS` | Binary flags |
| `ppi` | Pixels per inch, derived from resolution + screen size |
| `Cpu brand` | CPU family |
| `HDD` / `SSD` | Storage in GB |
| `Gpu brand` | GPU family |
| `os` | Operating system |

## Model

`pipe.pkl` is a fitted `sklearn.pipeline.Pipeline`:

1. **`ColumnTransformer`** — One-Hot Encodes `Company`, `TypeName`, `Cpu brand`, `Gpu brand`, `os`; passes numeric columns through unchanged.
2. **`RandomForestRegressor`** (`max_depth=15`, `max_features=0.75`, `max_samples=0.5`) — trained on `log(Price)`. Predictions are exponentiated back (`np.exp`) at inference time.

**Fit quality** (computed on the full dataset, *not* a held-out test split — treat as an optimistic upper bound):
- R² ≈ 0.871
- MAE ≈ ₹7,485



## Setup & Usage

```bash
# 1. Clone / download the project files
# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`. `pipe.pkl` and `data.pkl` must sit in the same folder as `app.py`.

## Limitations

- Trained on a static dataset — prices reflect that snapshot in time, not current market rates.
- R²/MAE reported above are training-data fit, not validated on unseen laptops.
- "Closest matches" lookup is a simple brand + type filter, not a true nearest-neighbor search.

## Possible Improvements

- Hold out a proper test split and report honest generalization metrics
- Add feature importance / SHAP explanations to the Predict tab
- Periodically refresh the dataset with current listings
- Deploy live on Streamlit Community Cloud

## Author

**Arzaan**
[GitHub](https://github.com/arzaanxeng) 
