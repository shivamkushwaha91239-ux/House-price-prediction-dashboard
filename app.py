# ==========================================
# Import Necessary Libraries
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

from utils.preprocess import clean_data
from utils.prediction import predict_house_price

from utils.visualization import (
    plot_histogram,
    plot_boxplot,
    plot_scatter,
    plot_heatmap,
    plot_target_distribution,
    plot_income_vs_price,
    plot_ocean_proximity
)
# ==========================================
# Configure Streamlit Page
# ==========================================

st.set_page_config(
    page_title="🏠 House Price Prediction Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>

/* Main Background */
.stApp{
    background-color:#0E1117;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background-color:#161A23;
}

/* Metric Cards */
div[data-testid="metric-container"]{
    background:#1E2430;
    border:1px solid #2F3645;
    padding:18px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.4);
}

/* Buttons */
.stButton>button{
    width:100%;
    border-radius:12px;
    height:3em;
    background:#1f77b4;
    color:white;
    font-size:17px;
    border:none;
}

.stButton>button:hover{
    background:#0d6efd;
}

/* Download Button */
.stDownloadButton>button{
    width:100%;
    border-radius:12px;
    background:#198754;
    color:white;
}

/* Headers */
h1{
    color:#4CAF50;
}

h2{
    color:#61dafb;
}

</style>
""", unsafe_allow_html=True)
# ==========================================
# Load Dataset
# ==========================================

@st.cache_data
def load_data():
    """
    Load housing dataset.
    Streamlit cache is used so the dataset is loaded only once.
    """
    df = pd.read_csv("housing.csv")
    return df
df = load_data()
# ==========================================
# Title
# ==========================================

st.title("🏠 House Price Prediction Dashboard")

st.markdown("""
Welcome to the **House Price Prediction Dashboard**.

This project demonstrates a complete end-to-end Machine Learning workflow:

- 📂 Data Loading
- 🧹 Data Cleaning
- 📊 Exploratory Data Analysis
- 📈 Business Insights
- 🤖 Linear Regression Model
- 📉 Model Evaluation
- 💰 House Price Prediction
""")

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📂 Dataset",
        "🧹 Data Cleaning",
        "📊 EDA",
        "📈 Insights",
        "🤖 Model Training",
        "📉 Evaluation",
        "💰 Prediction",
        "ℹ️ About"
    ]
)

# ==========================================
# Home Page
# ==========================================

if page == "🏠 Home":

    st.header("🏠 Dashboard Overview")

    st.write(
        "Welcome to the **House Price Prediction Dashboard**."
    )

    st.divider()

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📄 Rows",
        f"{df.shape[0]}"
    )

    col2.metric(
        "📋 Columns",
        f"{df.shape[1]}"
    )

    col3.metric(
        "❌ Missing",
        f"{df.isnull().sum().sum()}"
    )

    col4.metric(
        "📑 Duplicates",
        f"{df.duplicated().sum()}"
    )

    st.divider()

    st.subheader("📌 Project Workflow")

    st.markdown("""
1. 📂 Load Dataset

2. 🧹 Data Cleaning

3. 📊 Exploratory Data Analysis

4. 📈 Business Insights

5. 🤖 Train Linear Regression Model

6. 📉 Evaluate Model

7. 💰 Predict House Price
""")

    st.divider()

    st.subheader("🛠 Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.success("Python")
        st.success("Pandas")
        st.success("NumPy")

    with tech2:
        st.success("Plotly")
        st.success("Scikit-Learn")
        st.success("Streamlit")

    with tech3:
        st.success("Machine Learning")
        st.success("Linear Regression")
        st.success("Data Visualization")

    st.divider()

    st.info("🚀 End-to-End Machine Learning Project")
# ==========================================
# Dataset Page
# ==========================================

elif page == "📂 Dataset":

    st.header("📂 Dataset")

    st.success("✅ Dataset Loaded Successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicate Rows", df.duplicated().sum())

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str))

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
# ==========================================
# Data Cleaning
# ==========================================

elif page == "🧹 Data Cleaning":

    st.header("🧹 Data Cleaning")

    st.subheader("Before Cleaning")

    col1, col2 = st.columns(2)

    col1.metric("Missing Values", df.isnull().sum().sum())
    col2.metric("Duplicate Rows", df.duplicated().sum())

    st.dataframe(df.head())

    clean_df = clean_data(df)

    st.divider()

    st.subheader("After Cleaning")

    col1, col2 = st.columns(2)

    col1.metric("Missing Values", clean_df.isnull().sum().sum())
    col2.metric("Duplicate Rows", clean_df.duplicated().sum())

    st.dataframe(clean_df.head())

    st.success("✅ Dataset Cleaned Successfully")

    csv = clean_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Cleaned Dataset",
        data=csv,
        file_name="cleaned_housing.csv",
        mime="text/csv"
    )

# ==========================================
# EDA
# ==========================================

elif page == "📊 EDA":

    st.header("📊 Exploratory Data Analysis")

    clean_df = clean_data(df)

    numeric_columns = clean_df.select_dtypes(include=np.number).columns.tolist()

    # ----------------------------
    # Histogram
    # ----------------------------

    st.subheader("📈 Histogram")

    column = st.selectbox(
        "Select Column",
        numeric_columns
    )

    fig = plot_histogram(clean_df, column)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----------------------------
    # Box Plot
    # ----------------------------

    st.subheader("📦 Box Plot")

    fig = plot_boxplot(clean_df, column)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----------------------------
    # Scatter Plot
    # ----------------------------

    st.subheader("📍 Scatter Plot")

    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox(
            "Select X-axis",
            numeric_columns,
            index=0
        )

    with col2:
        y_axis = st.selectbox(
            "Select Y-axis",
            numeric_columns,
            index=1
        )

    fig = plot_scatter(clean_df, x_axis, y_axis)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----------------------------
    # Heatmap
    # ----------------------------

    st.subheader("🔥 Correlation Heatmap")

    fig = plot_heatmap(clean_df)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----------------------------
    # Target Distribution
    # ----------------------------

    st.subheader("🏠 House Price Distribution")

    fig = plot_target_distribution(clean_df)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----------------------------
    # Income vs Price
    # ----------------------------

    st.subheader("💰 Income vs House Price")

    fig = plot_income_vs_price(clean_df)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----------------------------
    # Ocean Proximity
    # ----------------------------

    st.subheader("🌊 Ocean Proximity")

    fig = plot_ocean_proximity(clean_df)

    st.plotly_chart(fig, use_container_width=True)
# ==========================================
# Insights
# ==========================================


elif page == "📈 Insights":

    st.header("📈 Business Insights")

    clean_df = clean_data(df)

    avg_price = clean_df["median_house_value"].mean()

    max_price = clean_df["median_house_value"].max()

    min_price = clean_df["median_house_value"].min()

    avg_income = clean_df["median_income"].mean()

    col1, col2 = st.columns(2)

    col1.metric(
        "Average House Price",
        f"${avg_price:,.0f}"
    )

    col2.metric(
        "Average Median Income",
        f"{avg_income:.2f}"
    )

    st.divider()

    col1, col2 = st.columns(2)

    col1.metric(
        "Maximum House Price",
        f"${max_price:,.0f}"
    )

    col2.metric(
        "Minimum House Price",
        f"${min_price:,.0f}"
    )

    st.divider()

    st.subheader("Ocean Proximity Analysis")

    ocean = (
        clean_df
        .groupby("ocean_proximity")["median_house_value"]
        .mean()
        .sort_values(ascending=False)
    )

    st.dataframe(ocean)

    st.divider()

    st.subheader("Top 5 Most Expensive Houses")

    top5 = clean_df.nlargest(
        5,
        "median_house_value"
    )

    st.dataframe(top5)

    st.divider()

    st.success("Business Insights Generated Successfully")
# ==========================================
# Model Training
# ==========================================

elif page == "🤖 Model Training":

    st.header("🤖 Machine Learning Model")

    st.success("Model Trained Successfully ✅")

    st.subheader("Model Information")

    st.write("""
### Algorithm Used

- Linear Regression

### Train-Test Split

- Training Data : 80%
- Testing Data : 20%

### Features Used

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

### Target Variable

- median_house_value
""")

    st.info("The trained model has been saved as model.pkl")
# ==========================================
# Evaluation
# ==========================================

elif page == "📉 Evaluation":

    import json

    st.header("📉 Model Evaluation")

    with open("metrics.json", "r") as file:
        metrics = json.load(file)

    col1, col2 = st.columns(2)

    col1.metric(
        "R² Score",
        f"{metrics['r2_score']:.4f}"
    )

    col2.metric(
        "Adjusted R²",
        f"{metrics['adjusted_r2']:.4f}"
    )

    st.divider()

    col3, col4, col5 = st.columns(3)

    col3.metric(
        "MAE",
        f"{metrics['mae']:.2f}"
    )

    col4.metric(
        "MSE",
        f"{metrics['mse']:.2f}"
    )

    col5.metric(
        "RMSE",
        f"{metrics['rmse']:.2f}"
    )

    st.success("Model Evaluation Loaded Successfully")
# ==========================================
# Prediction
# ==========================================
elif page == "💰 Prediction":

    st.header("💰 House Price Prediction")

    st.write("Enter the property details below.")

    col1, col2 = st.columns(2)

    with col1:

        longitude = st.number_input(
            "Longitude",
            value=-122.23
        )

        latitude = st.number_input(
            "Latitude",
            value=37.88
        )

        housing_median_age = st.number_input(
            "Housing Median Age",
            value=41
        )

        total_rooms = st.number_input(
            "Total Rooms",
            value=880
        )

        total_bedrooms = st.number_input(
            "Total Bedrooms",
            value=129
        )

    with col2:

        population = st.number_input(
            "Population",
            value=322
        )

        households = st.number_input(
            "Households",
            value=126
        )

        median_income = st.number_input(
            "Median Income",
            value=8.32
        )

        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            [
                "<1H OCEAN",
                "INLAND",
                "ISLAND",
                "NEAR BAY",
                "NEAR OCEAN"
            ]
        )
    
    if st.button("🔮 Predict House Price"):

        try:

            prediction = predict_house_price(
                longitude,
                latitude,
                housing_median_age,
                total_rooms,
                total_bedrooms,
                population,
                households,
                median_income,
                ocean_proximity
            )

            st.success("✅ Prediction Completed Successfully!")

            st.metric(
                label="🏠 Estimated House Price",
                value=f"${prediction:,.2f}"
            )

            st.balloons()

            st.info("🤖 Model Used: Linear Regression")

            st.caption(
                "This prediction is generated using the trained Linear Regression model."
            )

            prediction_df = pd.DataFrame({
                "Longitude": [longitude],
                "Latitude": [latitude],
                "Median Income": [median_income],
                "Predicted Price": [prediction]
            })

            csv = prediction_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Prediction",
                data=csv,
                file_name="house_price_prediction.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Prediction Failed: {e}")

# ==========================================
# About
# ==========================================


elif page == "ℹ️ About":

    st.header("ℹ️ About Project")

    st.markdown("""
# House Price Prediction Dashboard

This is an end-to-end Machine Learning project built using:

- Python
- Pandas
- NumPy
- Plotly
- Scikit-Learn
- Streamlit

## Machine Learning Algorithm

Linear Regression

## Features

- Dataset Analysis
- Data Cleaning
- Exploratory Data Analysis
- Business Insights
- Model Evaluation
- House Price Prediction

---

### Developer

**Shivam Kushwaha**

B.Tech CSE (IoT)

Babu Banarasi Das University
""")