# Car Price Prediction with Machine Learning

A machine learning regression project that predicts the selling price of used cars based on their specifications using the CarDekho dataset.

## Project Overview

The objective of this project is to develop and compare machine learning models that estimate used car prices based on features such as manufacturing year, present showroom price, kilometers driven, fuel type, transmission, seller type, and previous ownership.

This project demonstrates a complete supervised machine learning workflow, from data exploration and preprocessing to model training, evaluation, and saving.

## Dataset

**Source:** [Vehicle Dataset from CarDekho – Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

The dataset contains 301 records and 9 original columns.

| Column | Description |
|---|---|
| Car_Name | Car model or name |
| Year | Manufacturing year |
| Selling_Price | Actual selling price (target) |
| Present_Price | Current showroom price |
| Kms_Driven | Distance driven |
| Fuel_Type | Petrol, Diesel, or CNG |
| Seller_Type | Dealer or Individual |
| Transmission | Manual or Automatic |
| Owner | Number of previous owners |

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Joblib

## Project Workflow

### 1. Data Loading and Exploration
- Loaded the CarDekho dataset using Pandas.
- Examined dataset dimensions, column names, and data types.
- Generated statistical summaries.

### 2. Exploratory Data Analysis
- Checked missing values and duplicate records.
- Analyzed selling price distributions.
- Explored numerical and categorical features.
- Visualized relationships between car specifications and selling prices.

### 3. Data Cleaning
- Removed duplicate records.
- Checked data quality and missing values.
- Verified column data types.

### 4. Feature Engineering
- Extracted car brands from car names.
- Created a `Car_Age` feature using the manufacturing year.
- Removed the original car name and year columns after feature extraction.

### 5. Data Visualization
Created visualizations to investigate:
- Correlation between numerical features.
- Present price versus selling price.
- Car age versus selling price.
- Selling price across fuel types.
- Selling price across transmission types.

### 6. Feature Preparation
- Separated independent variables (X) and the target variable (y).
- Applied One-Hot Encoding to categorical features.
- Used Scikit-learn pipelines to combine preprocessing and model training.

### 7. Model Training

Two regression algorithms were implemented:

**Linear Regression**
- Used as a baseline regression model.
- Learns linear relationships between input features and selling prices.

**Random Forest Regression**
- Uses an ensemble of decision trees.
- Captures more complex relationships between car features and selling prices.

The dataset was divided into:
- 80% training data.
- 20% testing data.

### 8. Model Evaluation

Both models were evaluated using:

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error |
| RMSE | Penalizes larger prediction errors |
| R² Score | Measures the proportion of target variance explained |

The models were compared using their evaluation results, and the model with the lowest RMSE was selected.

### 9. Feature Importance

Random Forest feature importance was used to identify which features contributed most to the model's predictions.

### 10. Model Saving and Prediction

The selected model was saved using Joblib as:

`best_car_price_model.pkl`

A sample prediction was also performed using the saved model.

Example:

| | Value |
|---|---:|
| Predicted selling price | 8.68 Lakhs |
| Actual selling price | 8.99 Lakhs |
| Absolute prediction error | 0.31 Lakhs |

## Project Structure

```text
task-3-car-price-prediction/
│
├── data/
│   └── car_data.csv
│
├── notebooks/
│   └── car_price_prediction.ipynb
│
├── models/
│   └── best_car_price_model.pkl
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/BeimnetTadesse/task-3-car-price-prediction.git
cd task-3-car-price-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

`notebooks/car_price_prediction.ipynb`

Run the notebook cells to explore the dataset, train the models, and evaluate their performance.

## Limitations

- The dataset contains only 301 original records, which limits the amount of information available for training.
- The dataset represents a particular used-car market and may not generalize to other markets.
- Predictions depend on the features available in the dataset.
- Model performance on unseen real-world data may differ from the reported test results.

## Conclusion

This project demonstrates the application of supervised machine learning to a real-world regression problem.

By exploring vehicle data, engineering relevant features, training multiple regression models, and evaluating their performance, the project provides a practical introduction to building and assessing machine learning models.

The final trained model can be reused to generate used-car price predictions.