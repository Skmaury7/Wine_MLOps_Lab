Wine Quality MLOps Lab

This project demonstrates a complete machine learning workflow on Databricks using the wine quality dataset. The lab introduces MLOps (Machine Learning Operations) concepts such as experiment tracking, feature importance analysis, and model deployment using MLflow and Unity Catalog.

Objective

Predict whether a wine is high quality (quality score ≥ 7) based on its chemical properties.

Key Components

Exploratory Data Analysis

Visualized wine quality distribution (most wines rated 5–6)

Identified alcohol, density, and acidity as strong predictors
 Model Training

Trained baseline models (untuned random forest)

Trained optimized models using XGBoost with Hyperopt tuning

 Feature Importance

Top features: alcohol, density, volatile_acidity

 Tracking with MLflow

Logged model runs with parameters and performance metrics

Compared results to identify best-performing model

 Model Registry

Saved multiple versions of models in Unity Catalog

Promoted the best-performing model to “Production”

Screenshots Included

Feature importance plot

MLflow tracking (experiment runs)

Model registry view with version history

Key data visualizations (EDA)

Reflection Summary

Best model: XGBoost performed best due to its boosting strategy and regularization

Most important feature: Alcohol content had the strongest predictive power

Real-world use: This model could help wine producers predict quality based on lab tests before bottling or distributing

Tools Used

Databricks

Python (Pandas, Seaborn, Scikit-learn, XGBoost, MLflow)

Unity Catalog

 Structure

MLOps_on_Databricks.ipynb: Main lab notebook

Feature_Importance.png, MLflow_Tracking.png, etc.: Screenshots for assignment submission

Author

Sami Maury (Wine Quality Lab - MLOps Practice)

