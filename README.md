# 🏠 California Housing Price Prediction

A complete Machine Learning Regression Analysis project that predicts California housing prices using multiple supervised learning algorithms.

This project includes:

- Data Loading
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Model Training
- Model Evaluation
- Visualization Generation
- Feature Importance Analysis

The system compares multiple regression models and automatically selects the best-performing model using evaluation metrics.

---

# 📌 Project Objective

The objective of this project is to predict median house values in California districts using housing and census-based features such as:

- Median Income
- Population
- House Age
- Number of Rooms
- Number of Bedrooms
- Latitude & Longitude
- Household Density

This project demonstrates a complete end-to-end Machine Learning workflow using Python and Scikit-learn.

---

# 🧠 Machine Learning Models Used

The following regression algorithms are implemented and compared:

| Model | Description |
|---|---|
| Linear Regression | Baseline linear model |
| Random Forest Regressor | Ensemble tree-based model |
| Gradient Boosting Regressor | Sequential boosting algorithm |

---

# 📊 Evaluation Metrics

The models are evaluated using the following metrics:

| Metric | Meaning |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² Score | Variance explained by the model |

---

# 📂 Dataset Information

Dataset Used:
**California Housing Prices Dataset**

| Property | Value |
|---|---|
| Total Samples | 20,640 |
| Total Features | 10 |
| Task Type | Regression |
| Target Variable | Median House Value |

The project automatically:

- Loads `housing.csv` from the local `data/` folder
- Falls back to `sklearn.datasets.fetch_california_housing()` if the CSV file is not found

---

# ⚙️ Features

✅ Automatic dataset loading  
✅ Missing value handling  
✅ Categorical feature encoding  
✅ Outlier removal  
✅ Feature scaling  
✅ Correlation analysis  
✅ Visualization generation  
✅ Multiple model training  
✅ Automatic model evaluation  
✅ Best model selection  
✅ Residual analysis  
✅ Feature importance visualization  

---

# 🛠 Technologies Used

## Programming Language
- Python 3

## Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

# 📁 Project Structure

```text
california-housing-price-prediction/
│
├── data/
│   └── housing.csv
│
├── target_distribution.png
├── correlation_heatmap.png
├── feature_distributions.png
├── target_correlations.png
├── model_comparison.png
├── predicted_vs_actual.png
├── residual_analysis.png
├── feature_importance.png
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🔄 Complete Workflow Pipeline

## 1️⃣ Data Loading
The dataset is loaded either from:
- Local CSV file
- Scikit-learn built-in dataset

---

## 2️⃣ Exploratory Data Analysis (EDA)

The project performs:
- Dataset overview
- Missing value inspection
- Duplicate row detection
- Statistical summary
- Correlation analysis
- Distribution analysis

Generated plots:
- Histogram
- Boxplot
- Heatmap
- Correlation charts

---

## 3️⃣ Data Preprocessing

The preprocessing pipeline includes:

### Missing Value Handling
- Numeric columns → Median filling
- Categorical columns → Mode filling

### Categorical Encoding
- Label Encoding using `LabelEncoder`

### Outlier Removal
- Removes extreme values using percentile clipping

### Feature Scaling
- StandardScaler normalization

### Train/Test Split
- 80% Training
- 20% Testing

---

## 4️⃣ Model Training

Three regression models are trained:

### Linear Regression
Basic regression baseline model.

### Random Forest Regressor
Ensemble-based decision tree model.

### Gradient Boosting Regressor
Boosting algorithm optimized for regression tasks.

---

## 5️⃣ Model Evaluation

Each model is evaluated using:
- MAE
- RMSE
- R² Score

The project automatically identifies the best-performing model.

---

## 6️⃣ Visualization Generation

The project automatically saves the following visualizations:

| File Name | Description |
|---|---|
| target_distribution.png | Distribution of target variable |
| correlation_heatmap.png | Feature correlation matrix |
| feature_distributions.png | Histograms of features |
| target_correlations.png | Feature-target correlation |
| model_comparison.png | Comparison of all models |
| predicted_vs_actual.png | Actual vs predicted scatter plot |
| residual_analysis.png | Residual error analysis |
| feature_importance.png | Feature importance ranking |

---

# 📥 Installation

## Step 1: Clone Repository

```bash
git clone https://github.com/your-username/california-housing-price-prediction.git
```

---

## Step 2: Move into Project Directory

```bash
cd california-housing-price-prediction
```

---

## Step 3: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Or install using requirements file:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python main.py
```

---

# 🧪 Example Console Output

```text
============================================================
CALIFORNIA HOUSING PRICE PREDICTION
Regression Analysis Project
============================================================

[INFO] Dataset loaded: 20640 rows, 10 columns

[INFO] Starting data preprocessing...

[INFO] Training models...

MODEL EVALUATION RESULTS

Linear Regression
MAE: 0.4213
RMSE: 0.6124
R2: 0.71

Random Forest
MAE: 0.2871
RMSE: 0.4418
R2: 0.84

Gradient Boosting
MAE: 0.3012
RMSE: 0.4563
R2: 0.82
```

---

# 📈 Best Model Selection

The project automatically selects the best model based on the highest R² score.

Example:

```text
Winner: Random Forest
R2 Score: 0.84
```

---

# 📷 Visual Outputs

The generated visualizations help analyze:
- Feature distributions
- Correlations
- Prediction performance
- Residual patterns
- Feature importance

These plots are automatically saved in the project directory.

---

# 🚀 Future Improvements

Possible future enhancements:

- Hyperparameter tuning
- XGBoost integration
- LightGBM support
- CatBoost integration
- Cross-validation optimization
- Model deployment using Flask/FastAPI
- Streamlit dashboard
- Docker containerization
- CI/CD integration
- Cloud deployment

---

# 💡 Learning Outcomes

This project demonstrates practical implementation of:

- Supervised Machine Learning
- Regression Analysis
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Model Evaluation
- Data Visualization
- End-to-End ML Pipelines

---

# 👨‍💻 Author

Developed as a Machine Learning Regression Project using Python and Scikit-learn.

---

# 📜 License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this project for educational and personal purposes.

---

# ⭐ Support

If you found this project useful:

- Star the repository
- Fork the project
- Share it with others

---
