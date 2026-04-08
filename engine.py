import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
def load_data(path):
    df = pd.read_csv(path)
    product_factory_map = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory"
    }
    df['Region'] = df['Region'].str.strip().str.title()
    df['Factory'] = df['Product Name'].map(product_factory_map)
    print(df['Factory'].isnull().sum())
    print(df['Region'].unique())
    df['Order Date'] = pd.to_datetime(df['Order Date'],dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'],dayfirst=True)
    df['lead_time'] = ((df['Ship Date'] - df['Order Date']).dt.days) % 30
    df = df[df['lead_time'] < df['lead_time'].quantile(0.95)]
    df['lead_time_scaled'] = (
    (df['lead_time'] - df['lead_time'].min()) /
    (df['lead_time'].max() - df['lead_time'].min()))

    df = df.dropna()

    return df

factory_coords = {
    "Lot's O' Nuts": (32.88, -111.76),
    "Wicked Choccy's": (32.07, -81.08),
    "Sugar Shack": (48.11, -96.18),
    "Secret Factory": (41.44, -90.56),
    "The Other Factory": (35.11, -89.97)
}

region_coords = {
    "Pacific": (37.77, -122.41),   # West coast
    "Atlantic": (40.71, -74.00),   # East coast (NY side)
    "Gulf": (29.76, -95.36),       # Texas / Gulf area
    "Interior": (39.50, -98.35)    # Middle US
}

def calculate_distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def add_distance(df):
    distances = []

    for _, row in df.iterrows():
        f_lat, f_lon = factory_coords[row['Factory']]
        r_lat, r_lon = region_coords[row['Region']]

        dist = calculate_distance(f_lat, f_lon, r_lat, r_lon)
        distances.append(dist)

    df['distance'] = distances

    return df

def preprocess(df):
    le_product = LabelEncoder()
    le_region = LabelEncoder()
    le_ship = LabelEncoder()

    le_factory = LabelEncoder()

    df['product_enc'] = le_product.fit_transform(df['Product Name'])
    df['region_enc'] = le_region.fit_transform(df['Region'])
    df['ship_enc'] = le_ship.fit_transform(df['Ship Mode'])
    df['factory_enc'] = le_factory.fit_transform(df['Factory'])

    return df, le_product, le_region, le_ship, le_factory

def encode_features(product_name, region_name, ship_mode, factory_name, le_product, le_region, le_ship, le_factory):
    p_enc = int(le_product.transform([product_name])[0])
    r_enc = int(le_region.transform([region_name])[0])
    s_enc = int(le_ship.transform([ship_mode])[0])
    f_enc = int(le_factory.transform([factory_name])[0])
    f_lat, f_lon = factory_coords[factory_name]
    r_lat, r_lon = region_coords[region_name]
    dist = calculate_distance(f_lat, f_lon, r_lat, r_lon)

    return [p_enc, r_enc, s_enc, dist, f_enc]

def train_model(df):
    X = df[['product_enc', 'region_enc', 'ship_enc', 'distance', 'factory_enc']]
    y = df['lead_time_scaled']
    lr = LinearRegression()
    rf = RandomForestRegressor()
    gb = GradientBoostingRegressor()
    lr.fit(X, y)
    rf.fit(X, y)
    gb.fit(X, y)

    models = {
        "Linear": lr,
        "RandomForest": rf,
        "GradientBoost": gb
    }
    return models

def predict(model, le_product, le_region, le_ship, le_factory, product, region, ship, factory):
    features = encode_features(product, region, ship, factory, le_product, le_region, le_ship, le_factory)
    return float(model.predict([features])[0])

def simulate(model, le_product, le_region, le_ship, le_factory, product, region, ship, factories, priority):
    results = []

    for factory in factories:
        pred = predict(model, le_product, le_region, le_ship, le_factory, product, region, ship, factory)
        f_lat, f_lon = factory_coords[factory]
        r_lat, r_lon = region_coords[region]
        dist = calculate_distance(f_lat, f_lon, r_lat, r_lon)

        results.append({
            "factory": factory,
            "lead_time": pred,
            "distance": dist
        })

    for r in results:
        r['score'] = (r['lead_time'] * (1 - priority)) + (r['distance'] * priority * 0.05)

    results = sorted(results, key=lambda x: x['score'])

    return results

def build_engine(data_path):
    df = load_data(data_path)
    df = add_distance(df)
    df, le_p, le_r, le_s, le_f = preprocess(df)
    models = train_model(df)

    return models, le_p, le_r, le_s, le_f

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
