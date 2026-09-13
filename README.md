# Diabetes Prediction using Classical Machine Learning

## Overview
This project builds and compares five classical machine learning algorithms to predict the onset of diabetes based on diagnostic health measurements. It uses the widely known **Pima Indians Diabetes Dataset** and covers the full pipeline — from data preprocessing to model evaluation and hyperparameter tuning — with an emphasis on transparent, honest reporting of results rather than only showcasing the best numbers.

## Dataset
The dataset consists of diagnostic measurements for female patients of Pima Indian heritage, aged 21 and above. Each record includes the following features:

- **Pregnancies** — Number of times pregnant
- **Glucose** — Plasma glucose concentration
- **BloodPressure** — Diastolic blood pressure (mm Hg)
- **SkinThickness** — Triceps skinfold thickness (mm)
- **Insulin** — 2-Hour serum insulin (mu U/ml)
- **BMI** — Body mass index
- **DiabetesPedigreeFunction** — A function scoring likelihood of diabetes based on family history
- **Age** — Age in years
- **Outcome** — Target variable (1 = diabetic, 0 = non-diabetic)

## Objective
To build classification models that can accurately predict whether a patient has diabetes based on these diagnostic features, and to compare how different algorithms — and hyperparameter tuning — affect performance.

## Models Used
Five classical machine learning algorithms were implemented and compared:

1. **Logistic Regression** — a baseline linear classifier for binary outcomes
2. **Decision Tree Classifier** — a non-linear model that splits data based on feature thresholds
3. **Random Forest Classifier** — an ensemble of decision trees to reduce overfitting and improve generalization
4. **K-Nearest Neighbors (KNN)** — a distance-based classifier
5. **Support Vector Machine (SVM)** — a margin-based classifier effective in higher-dimensional spaces

## Methodology

### 1. Data Preprocessing
- Handled missing/zero values in biologically implausible fields (e.g. Glucose, BloodPressure, BMI recorded as 0)
- Performed exploratory data analysis (EDA) to understand feature distributions and correlations
- Applied feature scaling (StandardScaler) where required, particularly for KNN and SVM which are sensitive to feature magnitude
- Split the dataset into training and testing sets

### 2. Model Training
Each of the five models was trained on the processed training data using default hyperparameters as a baseline.

### 3. Hyperparameter Tuning
`GridSearchCV` was used to search for optimal hyperparameters for applicable models. Notably, tuning did **not** universally improve results — for **Random Forest** and **KNN**, tuned models showed a *reduction* in test accuracy compared to their default configurations. This finding is reported as-is, rather than omitted or adjusted to favor a cleaner narrative, since understanding when tuning helps (and when it doesn't) is itself a valuable takeaway.

### 4. Evaluation
Models were evaluated using standard classification metrics:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

## Key Findings
- Ensemble and margin-based methods generally performed competitively against simpler baselines.
- GridSearchCV-based tuning improved some models but reduced performance for Random Forest and KNN — a reminder that exhaustive tuning can overfit to validation folds rather than genuinely improving generalization.
- Feature scaling had a visible impact on distance-based and margin-based models (KNN, SVM), reinforcing the importance of preprocessing choices tailored to each algorithm.

## Tech Stack
- **Language:** Python
- **Libraries:** scikit-learn, pandas, NumPy, matplotlib/seaborn
- **Environment:** Google Colab

## How to Run
1. Clone this repository or open the notebook directly in Google Colab.
2. Ensure the dataset (Pima Indians Diabetes Dataset) is accessible in the notebook's working directory or loaded via a public URL.
3. Run the notebook cells sequentially:
   - Data loading and preprocessing
   - Exploratory data analysis
   - Model training (all five algorithms)
   - Hyperparameter tuning via GridSearchCV
   - Model evaluation and comparison

## Project Structure
```
diabetes-prediction-ml/
├── diabetes_prediction.ipynb   # Main notebook with full pipeline
└── README.md                   # Project documentation
```

## Future Improvements
- Experiment with additional models (e.g. Gradient Boosting, XGBoost)
- Address class imbalance if present, using techniques like SMOTE
- Deploy the best-performing model as a simple web app for interactive predictions

## Acknowledgements
Dataset originally from the National Institute of Diabetes and Digestive and Kidney Diseases, made publicly available via the UCI Machine Learning Repository / Kaggle.
