# Credit Card Fraud Detection

This project focuses on building a machine learning model to detect fraudulent credit card transactions using an imbalanced dataset. The dataset contains transactions made by European cardholders over two days in September 2013, with only 0.172% of transactions marked as fraudulent.

## **Features**
- **Time**: Seconds elapsed between each transaction and the first transaction.
- **Amount**: Transaction amount, which can be used for cost-sensitive learning.
- **V1-V28**: Principal components obtained via PCA.
- **Class**: Response variable; `1` indicates fraud, `0` indicates legitimate.

## **Project Workflow**

### **1. Data Preprocessing**
- Handled missing values and duplicates.
- Scaled numerical features using `StandardScaler`.
- Split the dataset into training and testing sets.

### **2. Feature Engineering**
- Applied log transformations to `Time` and `Amount` features.
- Created interaction terms for the top 3 important features using polynomial features.
- Added statistical features (mean and standard deviation) for selected features.

### **3. Model Training**
- Trained models using:
  - **XGBoost**
  - **LightGBM**
- Used `ADASYN` for oversampling to balance the dataset.
- Hyperparameter tuning via `GridSearchCV`.

### **4. Model Evaluation**
- Evaluated using:
  - Confusion matrix
  - Classification report
  - Precision-Recall curve
- Metrics focused on **recall** and **Area Under Precision-Recall Curve (AUPRC)**.

### **5. Results**
- Achieved significant improvements in detecting fraudulent transactions after addressing class imbalance and refining features.

## **Directory Structure**
```
credit_card_fraud_detection/
│
├── data_preprocessing.py     # Data loading, cleaning, and splitting
├── feature_engineering.py    # Feature transformations and creation
├── model_training.py         # Model training and hyperparameter tuning
├── model_evaluation.py       # Model evaluation metrics and visualizations
├── utils.py                  # Helper functions
├── main.py                   # Orchestrates the workflow
├── requirements.txt          # List of required libraries
├── README.md                 # Project documentation
└── data/                     # Dataset directory
    ├── creditcard.csv
    └── processed/
```

## **How to Run**
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

## **Dependencies**
- Python 3.8+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- lightgbm
- matplotlib
- joblib

## **Dataset**
- The dataset is available at [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## **Contact**
For inquiries or suggestions, please reach out to **[Your Name]** at **[Your Email]**.

