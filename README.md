# 🎮 Video Game Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5.0-green)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3.0-lightgreen)
![SHAP](https://img.shields.io/badge/SHAP-0.40.0-purple)

This project predicts video game prices based on various features such as reviews, release dates, system requirements, and more. It uses machine learning models like Random Forest, XGBoost, and LightGBM to achieve accurate predictions.

---

## 📋 Table of Contents
- [📊 Project Overview](#-project-overview)
- [📂 Dataset](#-dataset)
- [⚙️ Features](#️-features)
- [🚀 Models and Performance](#-models-and-performance)
- [📈 SHAP Analysis](#-shap-analysis)
- [📦 Installation](#-installation)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 📊 Project Overview

This project aims to predict the **original price** of video games using a variety of features, including:
- **Game reviews** (e.g., Recent Reviews Summary, Review Scores)
- **Release details** (e.g., Release Year, Game Age)
- **System requirements** (e.g., RAM, CPU, GPU)
- **Game content** (e.g., Description, Tags, Features)

### Key Highlights:
- **Data Cleaning**: Handled missing values, outliers, and inconsistent data.
- **Feature Engineering**: Created over 100 features, including temporal, review-based, and technical features.
- **Model Training**: Compared multiple regression models and fine-tuned the best-performing one.
- **Explainability**: Used SHAP to interpret feature importance and model predictions.

---

## 📂 Dataset

The dataset contains information about video games, including:
- **Target Variable**: `Original Price`
- **Features**: Reviews, release dates, system requirements, tags, and more.
- **Size**: Thousands of records (exact size depends on the dataset used).

### Data Cleaning:
- Removed columns with high missing values (e.g., `All Reviews Summary`, `All Reviews Number`).
- Converted price columns to numeric and replaced "Free" with `0`.
- Removed outliers using fixed thresholds (e.g., prices below $0 or above $100).

---

## ⚙️ Features

### Key Features:
1. **Price-Related**:
   - `Original Price` (target variable)
   - `Discounted Price`
   - `Discount Percentage`

2. **Temporal**:
   - `Release Year`, `Release Month`, `Game Age`
   - `Is Holiday Release`

3. **Review-Based**:
   - `Recent Reviews Summary` (mapped to numeric scores)
   - `Recent Reviews Number` (log-transformed)

4. **System Requirements**:
   - `Min_RAM_GB`, `CPU_Class`, `Requires_GPU`

5. **Content Features**:
   - `Description_Length` (from game descriptions)
   - `Num_Tags`, `Num_Game_Features`

6. **Developer/Publisher**:
   - Frequency encoding and average price by entity.

---

## 🚀 Models and Performance

### Models Trained:
- **Linear Regression**
- **Random Forest**
- **XGBoost**
- **LightGBM**

### Best Model: **Random Forest**
- **Test RMSE**: `X.XX`
- **Test R²**: `X.XX`

### Hyperparameter Tuning:
- Used `RandomizedSearchCV` to optimize parameters like `n_estimators`, `max_depth`, and `min_samples_split`.

### Overfitting/Underfitting Analysis:
- Compared training and testing metrics to ensure a good fit.
- Visualized residuals and performance gaps.

---

## 📈 SHAP Analysis

SHAP (SHapley Additive exPlanations) was used to interpret the model's predictions.

### Key Insights:
1. **Top Features**:
   - `Release Year`, `Recent Reviews Score`, `Min_RAM_GB`, `Discount Percentage`
2. **Business Insights**:
   - **Timing Matters**: Games released during holidays tend to have higher prices.
   - **Reviews Drive Value**: Positive reviews significantly impact pricing.
   - **Tech Specs Matter**: Higher system requirements correlate with higher prices.

### Visualizations:
- **SHAP Summary Plot**: Shows feature importance and impact.
- **SHAP Dependence Plot**: Reveals non-linear relationships.
- **SHAP Waterfall Plot**: Explains individual predictions.

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/videogame-price-prediction.git
   cd videogame-price-prediction
