
#===============================================================================
# %%
#* IMPORT PACKAGES
import os
from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import xgboost as xgb
import sklearn
print(xgb.__version__)
print(sklearn.__version__)

#===============================================================================
# %%
#* SET CWD TO REPO ROOT
print(f"CWD:{os.getcwd()}")
def trim_path_to_substring(path, substring):
    index = path.find(substring)
    if index == -1:
        raise ValueError(f"Substring '{substring}' not found in path: {path}")
    trimmed_path = path[:index + len(substring):]
    trimmed_path = trimmed_path.lstrip(os.sep)
    return trimmed_path

repo_name = 'PAJTK-PAD-proj'
os.chdir(trim_path_to_substring(os.getcwd(),repo_name))
print(f"CWD:{os.getcwd()}")
#===============================================================================
# %%
#* READ DATA
file= os.path.join('data','credit-data.csv')
fraud_data: DataFrame = pd.read_csv(file)
fraud_data: DataFrame = shuffle(fraud_data)
fraud_data_loaded = fraud_data.copy()
#===============================================================================
# %%
#* RELOAD DATA
fraud_data = fraud_data_loaded.copy()
df = fraud_data.copy()
#===============================================================================
# %%
# GLOBAL PARAMS

#===============================================================================
# %%
#* amount bins
def add_amt_bin(df):
    amt_bins=[0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 5000, 10000, 23000]
    bins = amt_bins
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    df['amt_bin'] = pd.cut(df['amt'], bins=bins, labels=bin_labels, right=False)

add_amt_bin(df)

#===============================================================================
# %%
#* time periods
def categorize_time_period(timestamp):
    hour = timestamp.hour
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 16:
        return 'afternoon'  # change from noon to afternoon
    elif 16 <= hour < 20:
        return 'evening'
    else:
        return 'night'
df['trans_date_trans_time_dt'] = pd.to_datetime(arg=df['trans_date_trans_time'], format='%d/%m/%Y %H:%M')
df['time_period'] = df['trans_date_trans_time_dt'].apply(categorize_time_period)
df['day_of_week'] = df['trans_date_trans_time_dt'].dt.day_name()
#===============================================================================
# %%
#* calculate distances
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Calculate differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Calculate distance
    distance = R * c
    return distance
df['cust_merch_distance'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])


#===============================================================================
# %%
#* number of transactions per customer
df['num_of_transactions'] = df.groupby('cc_num').cumcount() + 1

#===============================================================================
# %%
#* client age at transaction time

# Convert 'dob' column to datetime and extract year
dob_year = pd.to_datetime(df['dob'], format='%d/%m/%Y').dt.year

# Convert 'trans_date_trans_time' column to datetime and extract year
trans_year = pd.to_datetime(df['trans_date_trans_time_dt']).dt.year

# Calculate age of the customer
df['age_of_user'] = trans_year - dob_year

#===============================================================================
# %%
#* DROP COLUMNS

# Usuwamy kolumny z unikatowymi identyfikatorami
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('trans_num', axis=1, inplace=True)

# Imie i nazwisko jest nam nie potrzebne bo mamy cc_num
df.drop('first', axis=1, inplace=True)
df.drop('last', axis=1, inplace=True)

# dane juz nie potrzebne, bo przekształcone 
df.drop(columns=['trans_date_trans_time', 'trans_date_trans_time_dt', 'dob', 'cc_num', 'lat', 'long', 'merch_lat', 'merch_long'], inplace=True)

# te dane są kategoryczne o zbyt duze licznie unikalnych wartości aby ich uzyc - moze da się jakoś je przetworzyć?
df.drop(columns=['merchant', 'city', 'job', 'street'], inplace=True)

df.drop(columns=['zip', 'unix_time'], inplace=True)

#===============================================================================
# %%
df.columns
#===============================================================================
# %%
#* CORRELATION MATRIX
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15, 15))
corr = df[lambda df:[c for c in df.columns]].apply(lambda x : pd.factorize(x)[0]).corr()
sns.heatmap(corr.loc[lambda df:[c for c in df.columns],lambda df:[c for c in df.columns]],cmap='coolwarm')

#===============================================================================
# %%
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#===============================================================================
# %%
#* SPLIT FEATURES AND TARGET (independent and dependent variables)
Y = df[['is_fraud']]
X = df.drop(columns=['is_fraud'])

#===============================================================================
# %%
#* cechy o typach nie numerycznych
# cat_columns = ['category', 'gender', 'state', 'time_period', 'day_of_week']
cat_columns = X.select_dtypes(exclude=["bool_","number"]).columns.values.tolist()
print(f"categorical columns:")
print(cat_columns)

#===============================================================================
# %%
#* CATEGORICAL UNIQUE VALUES
# Dictionary to store number of unique values for each column
num_unique_values = {}

# Calculate number of unique values for each column in the DataFrame
for col in df.columns:
    if col in cat_columns:
        num_unique_values[col] = df[col].nunique()

# Display the number of unique values for each categorical column
for col, num_unique in num_unique_values.items():
    print(f"Number of unique values in column '{col}': {num_unique}")

#===============================================================================
# %%
#* REPLACE categorical columns with One Hot Encoded
cat_transformer_X = make_column_transformer(
    (OneHotEncoder(sparse_output=False), cat_columns),
    sparse_threshold= 0,
    verbose_feature_names_out=False,
    remainder='drop')

ohe_df = pd.DataFrame(cat_transformer_X.fit_transform(X), columns=cat_transformer_X.get_feature_names_out())
X = pd.concat([X.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1).drop(cat_columns, axis=1)
print(X.shape)
print(X)


#===============================================================================
# %%
#* FIND NUMERIC CATEGORICAL
# szukamy również danych o charakterze kategorycznym zapisanych w kolumnach numerycznych poprzez wyszukanie kolumn o małych zbiorach wartości unikalnych
print(f"\nunique values in columns:")
for col in X.select_dtypes(include=["number"]).columns:
    c_uniq = len(X[col].unique())
    if  c_uniq > 2:
        print(f'{col}: {c_uniq}')


#===============================================================================
# %%
#* SPLIT DATA INTO train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 5, stratify=Y['is_fraud'])

#===============================================================================
# %%
#* IDENTIFY QUANT COLUMNS
print(f"categorical columns:")
print(cat_columns)

quant_columns = [ c for c in X_train.columns if c not in cat_columns and c not in cat_transformer_X.get_feature_names_out()]

print(f"\nquant_columns:")
print(quant_columns)

#===============================================================================
# %%
#* NORMALIZE/SCALE QUANT COLUMNS
std_scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[quant_columns] = std_scaler.fit_transform(X_train[quant_columns])
X_test_scaled[quant_columns] = std_scaler.transform(X_test[quant_columns])

X_train_scaled.describe()

#===============================================================================
# %%
X_train_scaled

#===============================================================================
# %%
Y_train

#===============================================================================
# %%
#* SAMPLING / REBALANCING
# to address class imbalance -> fraud cases are rare so could ignore or overtrain
# i.e. reduce bias
# SMOTE (Synthetic Minority Over-sampling Technique)
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy = 'auto') # balance to make fraud and not-fraud equal

n = 1000 # rebalance to how many per class
X_train.reset_index(drop=True, inplace=True)
Y_train.reset_index(drop=True, inplace=True)

Y_sampled = Y_train.groupby('is_fraud', group_keys=False).apply(lambda x: x.sample(min(len(x), n), replace=False))
X_sampled = X_train.loc[Y_sampled.index]

X_resampled, Y_resampled = smote.fit_resample(X_sampled, Y_sampled['is_fraud'])

print("\nshape:")
print(X_resampled.shape)

print("\nresampled dataset:")
print(X_resampled)
#===============================================================================
# %%
#* TRAIN MODEL

# Gradient Boosting - handles imbalanced, good for fraud type of data
# Hyperparameter Tuning - optimize and try avoiding over/under fitting
# ROC-AUC - measures how well fraud and non-fraud cases are separated; accuracy can be misleading

# chain steps into a pipeline
pipeline = Pipeline([
    ('clf', GradientBoostingClassifier())
])

# hyperparameter grid
param_grid = {
    'clf__n_estimators': [50, 100, 150, 200],   # number of trees (Boosting rounds)
    'clf__learning_rate': [0.01, 0.1, 0.2],     # step size to control contribution of each tree
    'clf__max_depth': [3, 4, 5, 6]              # maximum depth of each tree (controls model complexity); impacts overfitting
}
# hyperparameter tuning - find best combination
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=10,                  # 10-fold cross-validation (split data into 10 parts, train on 9 and test on 1)
    scoring='roc_auc',      # evaluate based on ROC-AUC (important for fraud detection)
    n_jobs=-1               # use all cpu cores
)

grid_search.fit(X_resampled, Y_resampled)

#===============================================================================
# %%
#* BEST MODEL
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)
model_gradient_boosting = grid_search.best_estimator_
#** CONCLUSION:
# The model flags too many trx as fraud :)
# Could be useful for flagging but not for automated blocking.
#===============================================================================
# %%
#* RERUN BEST MODEL
# best_params = {k.replace('clf__', ''): v for k, v in grid_search.best_params_.items()}

# current_model = GradientBoostingClassifier(**best_params)
# current_model.fit(X_resampled, Y_resampled)

#===============================================================================
# %%
#* TEST MODEL - PREDICT

def test_model(model, name):
    Y_pred = model.predict(X_test)
    Y_pred_proba = model.predict_proba(X_test)  # Returns probability for each class
    model_test_results = {
        'Y_test': Y_test,
        'Y_pred': Y_pred,
        'Y_pred_proba': Y_pred_proba,
        'name': name
    }
    return model_test_results


#===============================================================================
# %%
#* MODEL STATS

def model_stats(Y_test, Y_pred, Y_pred_proba, name):
    
    print(f"\n{'='*20}")
    print(f"\n{'='*2} MODEL NAME: {name}")
    print(f"\n{'='*20}")
    
    
    #* MODEL STATS - confusion matrix
    import matplotlib.pyplot as plt

    # Normalize the confusion matrix
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]  # normalize along the true axis

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (Normalized)')
    plt.show()

    #* MODEL STATS - scalar metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.metrics import roc_auc_score

    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_pred_proba[:, 1])  # Use predicted 

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(
    """
    # Precision: How many of the predicted fraud cases are actually fraud? Low Precision means many false positives.
    # Recall: How many of the actual fraud cases did we detect? High means most frauds were detected(few false negatives).
    # F1-score: A balanced metric that combines Precision and Recall. High means the model can distinguish fraud vs. non-fraud well.
    """   
    )

    #* MODEL STATS - classification report
    from sklearn.metrics import classification_report
    print(classification_report(Y_test, Y_pred))

    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(Y_test, Y_pred_proba[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()

#===============================================================================
# %%
# TEST AND SHOW STATS
model_stats(**test_model(model_gradient_boosting, name = "Gradient Boosting"))

#===============================================================================
# %%
import xgboost as xgb
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)
my_model_xgb = xgb.XGBClassifier(
    n_estimators=100,        # Number of trees
    learning_rate=0.1,       # Step size shrinkage
    max_depth=5,             # Depth of each tree
    scale_pos_weight=10,     # Helps handle class imbalance
    eval_metric="auc"       # Optimize for fraud detection
)
    # use_label_encoder=False  # Avoid warning messages

# Train the model
my_model_xgb.fit(X_resampled, Y_resampled)

#===============================================================================
# %%
# TEST AND SHOW STATS
model_stats(**test_model(my_model_xgb, name = "XGBoost"))



#===============================================================================
# %%



#===============================================================================
# %%



#===============================================================================
# %%



#===============================================================================
# %%



#===============================================================================
# %%





