
# ============================================================
# SMART WATER MANAGEMENT SYSTEM
# Water Demand Forecasting & Potential Leakage Detection
# ============================================================

# 1. IMPORT LIBRARIES

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# 2. PAGE CONFIGURATION

st.set_page_config(
    page_title="Smart Water Management System",
    page_icon="💧",
    layout="wide"
)


# 3. CUSTOM CSS

st.markdown("""
<style>
.main {
    background-color: #f4f8fc;
}
h1 {
    color: #075985;
    font-weight: bold;
}
h2, h3 {
    color: #0369a1;
}
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #dbeafe;
}
</style>
""", unsafe_allow_html=True)


# 4. DASHBOARD HEADER

st.title("💧 Smart Water Management System")

st.markdown(
    "### Water Demand Forecasting & Potential Leakage Detection"
)

st.caption(
    "An AI-powered system for monitoring water consumption, "
    "forecasting demand and identifying unusual consumption patterns."
)

st.divider()

st.success("Welcome to the Smart Water Management Dashboard!")


# 5. LOAD AND PREPROCESS DATASET

@st.cache_data
def load_data():

    df = pd.read_csv(
        "waterconsumption_noNaN_fixedTimechange.csv"
    )

    # Handle missing values
    df['Rain'] = df['Rain'].fillna(
        df['Rain'].median()
    )

    # Create Datetime
    df['Hour'] = df['Time'].str.split('-').str[0]

    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Hour'],
        format='%m/%d/%Y %H:%M'
    )

    # Sort data chronologically
    df = df.sort_values(
        by='Datetime'
    ).reset_index(drop=True)

    # Time-based features
    df['Year'] = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    df['Day'] = df['Datetime'].dt.day
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Hour_Number'] = df['Datetime'].dt.hour

    df['IsWeekend'] = df['DayOfWeek'].apply(
        lambda x: 1 if x >= 5 else 0
    )

    # Lag features
    df['Consumption_Lag_1'] = (
        df['Consumption'].shift(1)
    )

    df['Consumption_Lag_24'] = (
        df['Consumption'].shift(24)
    )

    df['Rolling_Mean_24'] = (
        df['Consumption']
        .shift(1)
        .rolling(window=24)
        .mean()
    )

    # Remove missing lag rows
    df = df.dropna().reset_index(drop=True)

    return df


# 6. LOAD TRAINED ML MODELS

@st.cache_resource
def load_models():

    lr_model = joblib.load(
        'linear_regression_model.pkl'
    )

    rf_model = joblib.load(
        'random_forest_model.pkl'
    )

    iso_model = joblib.load(
        'isolation_forest_model.pkl'
    )

    return lr_model, rf_model, iso_model


df = load_data()

lr_model, rf_model, iso_model = load_models()

st.success(
    "Dataset and all 3 ML models loaded successfully!"
)

st.write("Dataset Shape:", df.shape)

st.write(
    "Dataset Date Range:",
    df['Datetime'].min(),
    "to",
    df['Datetime'].max()
)


# 7. DEFINE FEATURES GLOBALLY
# This fixes the NameError in Model Performance.

features = [
    'Temperature',
    'Rain',
    'Sun',
    'Year',
    'Month',
    'Day',
    'DayOfWeek',
    'Hour_Number',
    'IsWeekend',
    'Consumption_Lag_1',
    'Consumption_Lag_24',
    'Rolling_Mean_24'
]

anomaly_features = [
    'Consumption',
    'Temperature',
    'Rain',
    'Sun',
    'Hour_Number',
    'DayOfWeek',
    'IsWeekend'
]

split_index = int(len(df) * 0.80)


# 8. SIDEBAR NAVIGATION

st.sidebar.title("💧 Navigation")

page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Overview",
        "Water Demand Forecasting",
        "Anomaly Detection",
        "Model Performance"
    ]
)


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    st.header("📊 Water Consumption Overview")

    total_records = len(df)

    avg_consumption = df['Consumption'].mean()

    max_consumption = df['Consumption'].max()

    min_consumption = df['Consumption'].min()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{total_records:,}"
    )

    col2.metric(
        "Average Consumption",
        f"{avg_consumption:.2f}"
    )

    col3.metric(
        "Maximum Consumption",
        f"{max_consumption:,}"
    )

    col4.metric(
        "Minimum Consumption",
        f"{min_consumption:,}"
    )

    st.divider()

    # Daily consumption trend

    st.subheader("Daily Water Consumption Trend")

    daily_consumption = (
        df.set_index('Datetime')['Consumption']
        .resample('D')
        .mean()
    )

    st.line_chart(daily_consumption)

    # Hourly consumption

    st.subheader("Average Consumption by Hour")

    hourly_consumption = (
        df.groupby('Hour_Number')['Consumption']
        .mean()
    )

    st.bar_chart(hourly_consumption)

    st.divider()

    st.info(
        "This dashboard provides an overview of "
        "historical water consumption patterns."
    )


# ============================================================
# PAGE 2: WATER DEMAND FORECASTING
# ============================================================

elif page == "Water Demand Forecasting":

    st.header("📈 Water Demand Forecasting")

    st.write(
        "Select a date and hour to compare Random Forest "
        "predictions with actual water consumption."
    )

    test_start = df['Datetime'].iloc[split_index]

    selected_date = st.date_input(
        "Select Date",
        value=test_start.date(),
        min_value=test_start.date(),
        max_value=df['Datetime'].max().date()
    )

    day_data = df[
        df['Datetime'].dt.date == selected_date
    ].copy()

    if not day_data.empty:

        selected_hour = st.selectbox(
            "Select Hour",
            day_data['Hour_Number'].tolist(),
            format_func=lambda x: f"{x:02d}:00"
        )

        selected_row = day_data[
            day_data['Hour_Number'] == selected_hour
        ].iloc[0]

        input_data = selected_row[
            features
        ].to_frame().T

        predicted_consumption = rf_model.predict(
            input_data
        )[0]

        actual_consumption = selected_row[
            'Consumption'
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Consumption",
            f"{predicted_consumption:.2f}"
        )

        col2.metric(
            "Actual Consumption",
            f"{actual_consumption:.2f}"
        )

        col3.metric(
            "Absolute Prediction Error",
            f"{abs(actual_consumption - predicted_consumption):.2f}"
        )

        st.divider()

        # Predict consumption for selected day

        day_data['Predicted_Consumption'] = (
            rf_model.predict(day_data[features])
        )

        chart_data = day_data[
            [
                'Hour_Number',
                'Consumption',
                'Predicted_Consumption'
            ]
        ].set_index('Hour_Number')

        chart_data.columns = [
            'Actual Consumption',
            'Predicted Consumption'
        ]

        st.subheader(
            "Actual vs Predicted Water Consumption"
        )

        st.line_chart(chart_data)

        st.success(
            f"Prediction generated for {selected_date} "
            f"at {selected_hour:02d}:00."
        )

        st.caption(
            "This is a historical one-step-ahead prediction "
            "using previous actual consumption and recorded "
            "weather data. It is not a future multi-hour forecast."
        )

    else:

        st.warning(
            "No consumption records found for this date."
        )


# ============================================================
# PAGE 3: ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.header("🚨 Potential Water Leakage Detection")

    st.write(
        "Isolation Forest identifies unusual water consumption "
        "patterns that may require further investigation."
    )

    # Select test dataset

    test_data = df.iloc[split_index:].copy()

    # Detect anomalies

    test_data['Anomaly'] = iso_model.predict(
        test_data[anomaly_features]
    )

    normal_count = (
        test_data['Anomaly'] == 1
    ).sum()

    anomaly_count = (
        test_data['Anomaly'] == -1
    ).sum()

    total_count = len(test_data)

    anomaly_percentage = (
        anomaly_count / total_count
    ) * 100

    # Display metrics

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Observations",
        f"{total_count:,}"
    )

    col2.metric(
        "Normal Observations",
        f"{normal_count:,}"
    )

    col3.metric(
        "Potential Anomalies",
        f"{anomaly_count:,}"
    )

    st.metric(
        "Anomaly Percentage",
        f"{anomaly_percentage:.2f}%"
    )

    st.divider()

    # Anomaly graph

    st.subheader("Water Consumption Anomaly Graph")

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        test_data['Datetime'],
        test_data['Consumption'],
        color='blue',
        alpha=0.5,
        label='Water Consumption'
    )

    anomalies = test_data[
        test_data['Anomaly'] == -1
    ]

    ax.scatter(
        anomalies['Datetime'],
        anomalies['Consumption'],
        color='red',
        s=20,
        label='Potential Anomalies'
    )

    ax.set_xlabel("Date")

    ax.set_ylabel("Water Consumption")

    ax.set_title(
        "Potential Water Consumption Anomalies"
    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # Anomaly records

    st.subheader("Detected Anomaly Records")

    display_columns = [
        'Datetime',
        'Consumption',
        'Temperature',
        'Rain',
        'Hour_Number'
    ]

    anomaly_table = anomalies[
        display_columns
    ].sort_values(
        by='Consumption',
        ascending=False
    )

    st.dataframe(
        anomaly_table,
        use_container_width=True
    )

    st.warning(
        "These are unusual consumption patterns, not confirmed "
        "water leakages. Further investigation is required."
    )


# ============================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("🏆 Machine Learning Model Performance")

    st.write(
        "Comparison of Linear Regression and Random Forest "
        "for water demand forecasting."
    )

    # Evaluation results from notebook

    performance_data = pd.DataFrame({

        'Metric': [
            'MAE',
            'MSE',
            'RMSE',
            'R2 Score'
        ],

        'Linear Regression': [
            249.58,
            115916.76,
            340.47,
            0.9061
        ],

        'Random Forest': [
            190.71,
            67868.47,
            260.52,
            0.9450
        ]
    })

    st.subheader("Model Comparison")

    st.dataframe(
        performance_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # R2 Score Metrics

    col1, col2 = st.columns(2)

    col1.metric(
        "Linear Regression R2 Score",
        "90.61%"
    )

    col2.metric(
        "Random Forest R2 Score",
        "94.50%",
        delta="3.89 percentage points"
    )

    st.subheader("R2 Score Comparison")

    r2_chart = pd.DataFrame({

        'Model': [
            'Linear Regression',
            'Random Forest'
        ],

        'R2 Score': [
            0.9061,
            0.9450
        ]

    }).set_index('Model')

    st.bar_chart(r2_chart)

    st.divider()

    # Random Forest Feature Importance

    st.subheader(
        "Feature Importance - Random Forest"
    )

    feature_importance = pd.DataFrame({

        'Feature': features,

        'Importance': rf_model.feature_importances_

    }).sort_values(
        by='Importance',
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=feature_importance,
        x='Importance',
        y='Feature',
        ax=ax
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )

    ax.set_xlabel("Importance")

    ax.set_ylabel("Features")

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    st.success(
        "Random Forest is the better-performing forecasting "
        "model based on the evaluated test metrics."
    )

    st.caption(
        "R2 Score measures the proportion of variation "
        "explained by the model. Feature importance indicates "
        "predictive contribution, not causation."
    )


# ============================================================
# END OF APPLICATION
# ============================================================
