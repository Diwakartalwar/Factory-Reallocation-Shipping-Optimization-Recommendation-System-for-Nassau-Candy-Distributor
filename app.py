import streamlit as st
from engine import build_engine, simulate ,predict, visualize_results

model, le_p, le_r, le_s = build_engine("data.csv")

st.title("Factory Optimization System")

product = st.selectbox("Select Product", le_p.classes_)
region = st.selectbox("Select Region", le_r.classes_)
ship = st.selectbox("Select Ship Mode", le_s.classes_)

product_enc = le_p.transform([product])[0]
region_enc = le_r.transform([region])[0]
ship_enc = le_s.transform([ship])[0]

factories = [
    "Lot's O' Nuts",
    "Wicked Choccy's",
    "Sugar Shack",
    "Secret Factory",
    "The Other Factory"
]

if st.button("Run Simulation"):
    results = simulate(model, product_enc, region_enc, ship_enc, factories)

    st.write(results)
    st.write("Optimal Factory: ", results[0]['factory'])
    st.write("Predicted Lead Time: ", results[0]['lead_time'], " days")
    visualize_results(results)