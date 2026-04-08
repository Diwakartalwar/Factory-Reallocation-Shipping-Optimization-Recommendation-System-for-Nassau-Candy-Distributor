import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from engine import build_engine, simulate, predict, visualize_results


st.set_page_config(page_title="Factory Optimizer", layout="wide")


def get_factories():
    return [
        "Lot's O' Nuts",
        "Wicked Choccy's",
        "Sugar Shack",
        "Secret Factory",
        "The Other Factory",
    ]


def show_model_info(model_name: str) -> None:
    if model_name == "Linear":
        st.info("Linear Regression: Fast but may underfit complex patterns.")
    elif model_name == "RandomForest":
        st.info("Random Forest: Good accuracy, handles non-linearity, but can be slower.")
    elif model_name == "GradientBoost":
        st.info("Gradient Boosting: High accuracy, captures complex patterns, but can be slow and prone to overfitting.")


def run_tab_simulator(model, le_p, le_r, le_s, le_f, factories, priority, product, region, ship, factory):
    st.markdown(f"""
        **Selected:**  
        Product: `{product}` | Region: `{region}` | Ship: `{ship}` | Priority: `{round(priority,2)}`
    """)

    if st.button("Run Simulation"):
        results = simulate(
            model,
            le_p, le_r, le_s, le_f,
            product,
            region,
            ship,
            factories,
            priority,
        )
        df_results = pd.DataFrame(results)

        st.subheader("Simulation Results")
        st.dataframe(df_results)

        best = df_results.iloc[0]
        st.bar_chart(df_results.set_index('factory')['lead_time'])
        st.success(f"Best Factory: {best['factory']}")
        st.write(f"Estimated Score: {round(best['lead_time'], 3)}")
        st.write("Optimal Factory: ", results[0]['factory'])
        st.write("Predicted Lead Time: ", results[0]['lead_time'], " days")
        visualize_results(results)


def run_tab_whatif(model, le_p, le_r, le_s, le_f, product, region, ship, factories, priority):
    st.header("What-If Scenario Analysis")
    current_factory = st.selectbox("Current Factory", factories)

    if st.button("Compare Scenario"):
        results = simulate(
            model, le_p, le_r, le_s, le_f,
            product, region, ship, factories, priority
        )

        df_results = pd.DataFrame(results)
        best = df_results.iloc[0]
        current = df_results[df_results['factory'] == current_factory].iloc[0]
        improvement = current['lead_time'] - best['lead_time']

        st.write("### Comparison")
        st.write(f"Current Factory: {current_factory}")
        st.write(f"Best Factory: {best['factory']}")
        st.metric("Improvement", round(improvement, 3))


def run_tab_recommendations(model, le_p, le_r, le_s, le_f, product, region, ship, factories, priority):
    st.header("Recommendation Dashboard")
    results = simulate(
        model, le_p, le_r, le_s, le_f,
        product, region, ship, factories, priority
    )
    df_results = pd.DataFrame(results)

    st.dataframe(df_results)
    st.write("### Top 3 Recommendations")

    for i in range(3):
        row = df_results.iloc[i]
        st.success(f"{i+1}. {row['factory']} (Score: {round(row['lead_time'],3)})")

    best = df_results.iloc[0]
    current = df_results.iloc[-1]
    reduction = (current['lead_time'] - best['lead_time']) / current['lead_time']

    st.metric("Lead Time Reduction (%)", round(reduction * 100, 2))
    st.metric("Recommendation Coverage", len(df_results))
    st.metric("Confidence Score", round(1 - best['lead_time'], 2))


def run_tab_risk(model, le_p, le_r, le_s, le_f, product, region, ship, factories, priority):
    st.header("Risk & Impact Panel")
    results = simulate(
        model, le_p, le_r, le_s, le_f,
        product, region, ship, factories, priority
    )
    df_results = pd.DataFrame(results)
    best = df_results.iloc[0]

    if best['distance'] > 15:
        st.warning("⚠️ High distance → potential delay risk")
    else:
        st.success("Low risk route")

    profit_impact = (1 - best['lead_time']) * 100
    st.metric("Estimated Profit Impact (%)", round(profit_impact, 2))


def main():
    col1, col2 = st.columns(2)
    models, le_p, le_r, le_s, le_f = build_engine("data.csv")
    st.title("Factory Optimization System")
    st.markdown("""

    ### Smart Logistics Decision System
    Optimize your factory allocation and shipping routes with our AI-powered recommendation engine.""")

    model_name = st.selectbox("Select Model", list(models.keys()))
    if model_name not in models:
        st.warning("Please select a model to proceed.")
        st.stop()

    show_model_info(model_name)
    model = models[model_name]

    priority = st.slider(
        "Optimization Priority (Speed vs Profit)",
        0.0, 1.0, 0.5
    )

    factories = get_factories()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏭 Simulator",
        "🔄 What-If",
        "📊 Recommendations",
        "⚠️ Risk & Impact",
    ])

    with tab1:
        # create inputs and run simulation UI
        col1, col2, col3,col4 = st.columns(4)

    with col1:
        product = st.selectbox("Product", le_p.classes_)

    with col2:
        region = st.selectbox("Region", le_r.classes_)

    with col3:
        ship = st.selectbox("Ship Mode", le_s.classes_)
    with col4:
        factory = st.selectbox("Factory", le_f.classes_)
    run_tab_simulator(model, le_p, le_r, le_s, le_f, factories, priority, product, region, ship, factory)

    with tab2:
        # rely on the same widgets created above for product/region/ship
        run_tab_whatif(model, le_p, le_r, le_s, le_f, product, region, ship, factories, priority)

    with tab3:
        run_tab_recommendations(model, le_p, le_r, le_s, le_f, product, region, ship, factories, priority)

    with tab4:
        run_tab_risk(model, le_p, le_r, le_s, le_f, product, region, ship, factories, priority)


if __name__ == "__main__":
    main()