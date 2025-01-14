from data_preprocessing import load_data, clean_data, scale_features, split_data
from feature_engineering import log_transform, add_statistical_features, add_interaction_terms
from model_training import train_model, tune_hyperparameters, save_model
from model_evaluation import evaluate_model, plot_precision_recall_curve
from utils import calculate_scale_pos_weight

# Step 1: Load and preprocess data
data = load_data('data/creditcard.csv')
data = clean_data(data)
X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
X_scaled, scaler = scale_features(X)
X_train, X_test, y_train, y_test = split_data(X_scaled, y)

# Step 2: Feature engineering
X_train = log_transform(X_train, [-2, -1])
X_test = log_transform(X_test, [-2, -1])
X_train = add_statistical_features(X_train, [0, 1, 2])
X_test = add_statistical_features(X_test, [0, 1, 2])

# Step 3: Train model
scale_pos_weight = calculate_scale_pos_weight(y_train)
params = {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.2, 'scale_pos_weight': scale_pos_weight}
model = train_model(X_train, y_train, params)

# Step 4: Evaluate model
precision, recall, auprc = evaluate_model(model, X_test, y_test)
plot_precision_recall_curve(precision, recall, auprc)

# Step 5: Save model
save_model(model, 'xgb_model.pkl')
