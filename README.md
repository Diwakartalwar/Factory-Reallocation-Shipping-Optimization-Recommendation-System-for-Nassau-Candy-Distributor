# Factory Reallocation & Shipping Optimization Recommendation System for Nassau Candy Distributor


A machine learning–powered decision support system to optimize factory allocation and improve shipping efficiency.

---

## 🧠 Overview

This project predicts shipping lead time and recommends the most efficient factory for product delivery based on multiple factors like region, shipping mode, and distance.

It combines:
- Machine Learning models
- Simulation engine
- Interactive dashboard (Streamlit)

---

## 🎯 Objectives

- Predict shipping lead time
- Optimize factory assignment
- Reduce delivery delays
- Provide data-driven logistics decisions

---

## ⚙️ Features

### 🏭 Factory Optimization Simulator
- Select product, region, and shipping mode
- View predicted performance across factories
- Get best factory recommendation

### 🔄 What-If Analysis
- Compare current vs optimal factory
- Measure efficiency improvement

### 📊 Recommendation Dashboard
- Ranked factory suggestions
- Top-N optimal choices

### ⚠️ Risk & Impact Panel
- Distance-based risk alerts
- Profit impact estimation

---

## 🧠 Machine Learning Models

- Linear Regression (baseline)
- Random Forest Regressor (best performing)
- Gradient Boosting Regressor

---

## 📊 Model Inputs

- Product
- Factory
- Region
- Ship Mode
- Distance (engineered feature)

---

## 📈 Evaluation Metrics

- MAE (Mean Absolute Error)
- R² Score

Random Forest was selected for final predictions due to better performance.

---

## 🔁 Simulation Logic

For each product:
- Assign to all factories
- Predict lead time
- Rank based on:
  - Speed (lead time)
  - Distance (cost proxy)
  - User-defined priority

---

## 🎚️ Optimization Strategy

A weighted scoring system is used:
score = (lead_time * (1 - priority)) + (distance * priority)


- Priority = 0 → speed focus  
- Priority = 1 → cost/distance focus  

---

## 📊 Dataset

Contains:
- Order & Ship Dates
- Product details
- Region & shipping mode
- Sales and cost data

### Data Processing:
- Date conversion
- Lead time calculation
- Outlier removal
- Encoding categorical variables
- Distance feature engineering

---

## 🖥️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- Streamlit

---

## 🚀 Live Demo

👉 https://unified-mentor1.streamlit.app

---

## 📂 Project Structure
├── app.py
├── engine.py
├── data.csv
├── requirements.txt
└── README.md