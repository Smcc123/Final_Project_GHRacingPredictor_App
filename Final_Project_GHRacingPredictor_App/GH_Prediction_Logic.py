import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib 


#Import the data into the model 
data = pd.read_csv("C:\Users\angie\OneDrive\Desktop\Final Project\Final_Project_GHRacingPredictor_App\data\data_final.csv")
data.head()

# Feature 1: Probability according to odds
data['implied_probability'] = 1 / data['Odds']

# Feature 2: Win Percentage
data['win_percentage'] = data['Finish_All'] / data['Races_All']

# Feature 3: correlation between odds and public opinion
data['BSP_Odds_PublicEstimate'] = data['BSP'] + data['Odds'] / data['Public_Estimate']


# Define features (X) and target variable (y)
X = data.drop('Winner', axis=1)  # Features
y = data['Winner']  # Target variable


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE to the training data
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# CatBoost Model with GridSearchCV
catboost_param_grid = {
    'iterations': [200],
    'depth': [10],
    'learning_rate': [0.1],
    'scale_pos_weight': [5]
}

catboost = CatBoostClassifier(random_state=42, verbose=0)
catboost_grid = GridSearchCV(catboost, param_grid=catboost_param_grid, cv=5, scoring='accuracy')
catboost_grid.fit(X_train_resampled, y_train_resampled)
y_pred = catboost_grid.predict(X_test)


# SVM Model with GridSearchCV
svm_param_grid = {
    'C': [10],
    'gamma': [0.1],
    'kernel': ['linear'],
    'class_weight': ['balanced']
}

svm = SVC(probability=True, random_state=42)
svm_grid = GridSearchCV(svm, param_grid=svm_param_grid, cv=5, scoring='accuracy')
svm_grid.fit(X_train_resampled, y_train_resampled)
y_pred = svm_grid.predict(X_test)


# XGBoost Model with GridSearchCV
xgb_param_grid = {
    'n_estimators': [200],
    'learning_rate': [0.1],
    'max_depth': [7],
    'scale_pos_weight': [5]
}

xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_grid = GridSearchCV(xgb, param_grid=xgb_param_grid, cv=5, scoring='accuracy')
xgb_grid.fit(X_train_resampled, y_train_resampled)
y_pred = xgb_grid.predict(X_test)

# Ensemble Stacking Meta Model
estimators = [
    ('catboost', catboost_grid.best_estimator_),
    ('svm', svm_grid.best_estimator_),
    ('xgb', xgb_grid.best_estimator_)
]

stacking = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(max_iter=5000, class_weight='balanced')
)
stacking.fit(X_train_resampled, y_train_resampled)
y_pred = stacking.predict(X_test)

# Save the stacking model (function, save name)
joblib.dump(stacking, 'models/stacking_model.pkl')
#

# Save the scaler
joblib.dump(scaler, 'models/scaler.pkl')

# Save the SMOTE object
joblib.dump(smote, 'models/smote.pkl')
