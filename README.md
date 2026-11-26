# Smart Water Management System

A machine-learning project for **hourly water demand forecasting** and
**potential water-leak anomaly detection** using historical
water-consumption and weather data.

## Project Overview

Water demand changes over time due to factors such as hour of the day,
day of the week, temperature, rainfall, and previous consumption. This
project analyzes these patterns to forecast hourly water demand. It also
aims to identify unusual consumption patterns that may indicate possible
leaks or other abnormal usage.

> **Important:** The dataset does not contain verified leak labels.
> Anomalies identified by the system should therefore be treated as
> potential leak indicators, not confirmed physical leaks.

## Objectives

-   Forecast hourly water consumption using machine-learning models.
-   Analyze historical consumption and weather conditions.
-   Identify unusual water-consumption patterns using anomaly detection.
-   Support more informed water-resource planning and monitoring.

## Dataset

**Dataset:** Hourly Water Demand of a Mixed District\
**Source:** [Zenodo -- Hourly Water Demand of a Mixed
District](https://zenodo.org/records/11045013)\
**File used:** `waterconsumption_noNaN_fixedTimechange.csv`

The dataset contains hourly records with the following original columns:

  Column          Description
  --------------- ----------------------------
  `Date`          Date of observation
  `Time`          Hourly time interval
  `Consumption`   Recorded water consumption
  `Temperature`   Temperature
  `Rain`          Rainfall
  `Sun`           Sunshine measurement

## Technologies and Libraries

-   Python
-   Jupyter Notebook (Anaconda)
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn

## Project Workflow

1.  **Data loading and inspection** -- Load the dataset and examine its
    structure, data types, and missing values.
2.  **Data cleaning** -- Handle missing rainfall values and prepare the
    date-time information.
3.  **Feature engineering** -- Extract time-based features such as year,
    month, day, weekday, hour, and weekend status.
4.  **Exploratory data analysis** -- Visualize consumption trends,
    hourly patterns, weekday/weekend differences, and the consumption
    distribution.
5.  **Lag feature creation** -- Create previous-hour consumption,
    previous-24-hour consumption, and a rolling 24-hour average.
6.  **Chronological data split** -- Split the data into training and
    testing sets in time order to avoid randomly mixing past and future
    observations.
7.  **Demand forecasting** -- Train and evaluate Linear Regression and
    Random Forest Regression models.
8.  **Anomaly detection** -- Use Isolation Forest to investigate unusual
    consumption patterns as potential leak indicators.

## Forecasting Results

The following results were obtained on the chronological test set:

  Metric       Linear Regression   Random Forest
  ---------- ------------------- ---------------
  MAE                     249.58      **190.71**
  MSE                 115,916.76   **67,868.47**
  RMSE                    340.47      **260.52**
  R² Score                0.9061      **0.9450**

Random Forest performed better than Linear Regression on this test set,
with lower MAE and RMSE and a higher R² score.

**Metric note:** MAE and RMSE measure prediction error in the dataset's
consumption units. R² measures the proportion of target variation
explained by the model; it is not classification accuracy.

## Repository Structure

``` text
Smart-Water-Management/
│
├── Smart_Water_Management.ipynb
├── waterconsumption_noNaN_fixedTimechange.csv
└── README.md
```

Your repository may contain additional files as the project develops.

## How to Run

1.  Clone or download this repository.
2.  Open Anaconda Navigator and launch Jupyter Notebook.
3.  Open `Smart_Water_Management.ipynb`.
4.  Ensure the dataset CSV is in the same folder as the notebook.
5.  Run the notebook cells in order.

## Current Scope and Future Improvements

-   Complete and evaluate the Isolation Forest anomaly-detection module.
-   Visualize detected anomalies against actual consumption.
-   Tune model parameters and evaluate performance on additional time
    periods.
-   Develop an interactive dashboard for viewing demand forecasts and
    anomaly alerts.

## Disclaimer

This project is an educational machine-learning prototype. Its anomaly
detection identifies unusual patterns in recorded consumption; it does
not independently verify the presence of a physical water leak.

## Trained Machine Learning Model

**Model:** Random Forest Regressor

**File:** `random_forest_water_demand_model.pkl`

**Size:** 317 MB

[Download Trained Model](https://drive.google.com/file/d/13reZiHY0h7xtTB1jjeZutlJJtnBpGG6W/view?usp=sharing)
