# Regression Analysis on Airline Passenger Satisfaction Dataset

## Overview
This project explores different regression models on the "Airline Passenger Satisfaction" dataset to predict customer satisfaction based on various features. The models include:
- Linear Regression (Multiple Variables)
- Polynomial Regression
- Lasso Regression
- Elastic Net Regression

## Dataset
- **Source**: Kaggle ([budincsevity/szeged-weather](https://www.kaggle.com/datasets/budincsevity/szeged-weather))
- **Description**: This dataset contains historical weather data from Szeged, Hungary, including features such as temperature, humidity, wind speed, precipitation, and more. The dataset provides daily weather information from 2006 to 2016.
- **Target Variable**: `Temperature (C)`

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Kaggle API

## Regression Models Implemented
### 1. Linear Regression (Multiple Variables)
- Uses multiple independent variables to predict satisfaction
- Evaluates model performance using R-squared and Mean Squared Error (MSE)

### 2. Polynomial Regression
- Transforms features to polynomial terms to capture non-linearity
- Compares performance with multiple polynomial degrees

### 3. Lasso Regression
- Applies L1 regularization to shrink less significant feature coefficients to zero
- Helps in feature selection by eliminating irrelevant variables

### 4. Elastic Net Regression
- Combines L1 and L2 regularization (Lasso + Ridge)
- Balances feature selection and coefficient shrinkage

## How to Run the Code
1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn kagglehub
   ```
2. Run each notebook (`.ipynb`) in Jupyter Notebook or Google Colab:
   - `LinearRegMulti.ipynb`
   - `PolyRegMulti.ipynb`
   - `LassoReg.ipynb`
   - `ElasticNetReg.ipynb`

## Results & Insights
- Polynomial Regression outperforms Linear Regression in capturing non-linearity.
- Lasso Regression reduces overfitting by removing irrelevant features.
- Elastic Net Regression balances bias-variance trade-off effectively.

## Future Improvements
- Try other regression techniques like Ridge Regression or Decision Trees.
- Use feature engineering to improve model accuracy.
- Experiment with hyperparameter tuning to optimize models.

