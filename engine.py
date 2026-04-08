import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
def load_data(path):
    df = pd.read_csv(path)

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])

    df['lead_time'] = (df['Ship Date'] - df['Order Date']).dt.days

    df = df.dropna()

    return df

def preprocess(df):
    le_product = LabelEncoder()
    le_region = LabelEncoder()
    le_ship = LabelEncoder()

    df['product_enc'] = le_product.fit_transform(df['Product Name'])
    df['region_enc'] = le_region.fit_transform(df['Region'])
    df['ship_enc'] = le_ship.fit_transform(df['Ship Mode'])

    return df, le_product, le_region, le_ship

def train_model(df):
    X = df[['product_enc', 'region_enc', 'ship_enc']]
    y = df['lead_time']

    model = RandomForestRegressor()
    model.fit(X, y)

    return model

def predict(model, product, region, ship):
    return model.predict([[product, region, ship]])[0]

def simulate(model, product, region, ship, factories):
    results = []

    for factory in factories:
        pred = predict(model, product, region, ship)

        results.append({
            "factory": factory,
            "lead_time": pred
        })

    results = sorted(results, key=lambda x: x['lead_time'])

    return results

def build_engine(data_path):
    df = load_data(data_path)
    df, le_p, le_r, le_s = preprocess(df)
    model = train_model(df)

    return model, le_p, le_r, le_s

def visualize_results(results):
    df_results = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='factory', y='lead_time', data=df_results)
    plt.title('Predicted Lead Time by Factory')
    plt.xlabel('Factory')
    plt.ylabel('Predicted Lead Time (days)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
