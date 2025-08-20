# Trade-Behaviour-Analytics
Analyzing cryptocurrency trading historical data and market sentiment (Fear/Greed Index) to understand trading behavior, predict trade direction, and evaluate account performance using data preprocessing, EDA, and machine learning models (XGBoost, Logistic Regression).
# Crypto Trading Analysis and Prediction

This repository contains a Jupyter notebook (`.ipynb` file) that performs an in-depth analysis of cryptocurrency trading historical data and the Fear and Greed Index. The project aims to understand trading behavior, explore the relationship between market sentiment and trading outcomes, evaluate account performance, and build machine learning models to predict trade direction.

## Project Overview

The analysis pipeline covers:

1.  **Data Loading and Preprocessing:** Loading historical trade data and Fear/Greed Index data, cleaning column names, and handling timestamps and data types.
2.  **Exploratory Data Analysis (EDA):** Visualizing the distribution of Fear/Greed Index classifications and exploring key metrics from the historical trading data.
3.  **Data Aggregation:** Aggregating historical trade data to daily and account levels to derive summary statistics.
4.  **Account-Level Analysis:** Analyzing account performance based on aggregated metrics like total PnL, win rate, and trading volume.
5.  **Trade Direction Prediction:** Building and evaluating binary classification models (Logistic Regression, XGBoost) to predict whether an individual trade will be a 'BUY' or 'SELL' based on pre-trade information. This involves careful feature engineering to avoid data leakage and handling class imbalance using techniques like SMOTE.
6.  **Model Interpretation:** Analyzing feature importances to understand which factors are most influential in predicting trade direction.
7.  **Hyperparameter Tuning:** Improving model performance (specifically recall) through hyperparameter tuning using `RandomizedSearchCV` with cross-validation.

## Data

The analysis uses two primary datasets:

*   `historical_data.csv`: Contains detailed information about individual trades, including execution price, size, side (BUY/SELL), timestamp, PnL, fees, etc.
*   `fear_greed_index.csv`: Contains historical data for the cryptocurrency Fear and Greed Index, including timestamp, value, and classification (Extreme Fear, Fear, Neutral, Greed, Extreme Greed).

*(Note: The original data files are assumed to be available at the specified paths or uploaded to the Colab environment.)*

## Methodology

*   **Data Cleaning:** Standardizing column names, converting timestamps, and coercing data types.
*   **Feature Engineering:** Creating relevant features at the trade level (e.g., time of day, day of week) and potentially lagged features while strictly using only pre-trade information. Aggregated features were also created at daily and account levels.
*   **Handling Imbalance:** SMOTE was applied to the training data to address the class imbalance in the trade direction prediction task.
*   **Modeling:** Logistic Regression and XGBoost classification models were used for trade direction prediction.
*   **Hyperparameter Tuning:** `RandomizedSearchCV` with cross-validation was used to tune the XGBoost model, optimizing for recall.
*   **Evaluation:** Models were evaluated using metrics like Accuracy, Precision, Recall, F1-Score, ROC AUC, and Confusion Matrix on a time-based test set.
*   **Interpretation:** Feature importances were analyzed to understand model drivers.

## Key Findings

*   Exploratory analysis provided insights into the distribution of market sentiment and aggregated trading metrics.
*   Account-level analysis helped characterize the performance of different trading accounts.
*   The trade direction prediction task highlighted the importance of careful feature engineering to avoid data leakage.
*   Handling class imbalance significantly impacted the model's ability to predict the minority class ('BUY').
*   Hyperparameter tuning improved the XGBoost model's performance, particularly its recall.
*   Feature importance analysis revealed that trade size, fees, starting position, and the time of the trade were significant predictors of trade direction.

## How to Use

1.  Clone this repository.
2.  Open the `.ipynb` notebook in Google Colab or a Jupyter environment.
3.  Ensure you have the required libraries installed (pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost, lightgbm - some may require separate installation via `pip`).
4.  Upload the `historical_data.csv` and `fear_greed_index.csv` files if running in an environment like Google Colab where they are not already present.
5.  Run the cells sequentially to execute the analysis pipeline.

## Libraries Used

*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `sklearn`
*   `imblearn` (for handling imbalanced data)
*   `xgboost`
*   `lightgbm` (optional, if installed)

## License

*(Consider adding a license here, e.g., MIT, Apache 2.0)*
