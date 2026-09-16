import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# a) Data exploration: first few rows, summary statistics, data types
def data_exploration():
    df = pd.read_csv("graduation_dataset.csv")

    # First few rows
    print("First few rows:")
    print(df.head())

    # Summary statistics
    print("\nSummary statistics:")
    print(df.describe()) 

    # Data types
    print("\nData types:")
    print(df.dtypes)

data_exploration()

df = pd.read_csv('graduation_dataset.csv')

# b) Missing values, outliers and unique values
print(df.isnull().sum())

# Outliers with IQR-method
for col in ['Age at enrollment', 'Curricular units 1st sem (grade)', 
            'Curricular units 2nd sem (grade)']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f'{col}: {len(outliers)} outliers')

# Unique values in categorical columns
for col in ['Marital status', 'Gender', 'Course', 'Application mode']:
    print(f'{col}: {df[col].nunique()} unique values')

    cols_to_check = ['Age at enrollment', 
                  'Curricular units 1st sem (grade)', 
                  'Curricular units 2nd sem (grade)']

# Outliers with Z-score
print("\nZ-SCORE METHOD (threshold = 3)")
zscore_results = {}
for col in cols_to_check:
    z_scores = np.abs(stats.zscore(df[col]))
    outliers = df[z_scores > 3]
    zscore_results[col] = len(outliers)
    print(f'{col}: {len(outliers)} outliers')

# Comparison: IQR vs Z-score
iqr_results = {}
for col in cols_to_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    iqr_results[col] = len(outliers)

print("\nCOMPARISON")
print(pd.DataFrame({'IQR': iqr_results, 'Z-score': zscore_results}))

# Boxplots
fig, axes = plt.subplots(1, len(cols_to_check), figsize=(15, 4))
for i, col in enumerate(cols_to_check):
    axes[i].boxplot(df[col])
    axes[i].set_title(col, fontsize=9)
plt.tight_layout()
plt.savefig('outliers_boxplot.png', dpi=150)
plt.show()

# Handle outliers: cap Age at enrollment (IQR-based), keep grades unchanged
Q1 = df['Age at enrollment'].quantile(0.25)
Q3 = df['Age at enrollment'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df['Age at enrollment'] = df['Age at enrollment'].clip(lower=lower, upper=upper)

print(f"\nAge at enrollment after capping: min={df['Age at enrollment'].min()}, "
      f"max={df['Age at enrollment'].max()}")

# 4a) Encoding categorical data

categorical_cols = [
    'Marital status',
    'Application mode',
    'Course',
    'Daytime/evening attendance',
    'Previous qualification',
    'Nacionality',
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder',
    'International'
]

# One-hot encoding
df = pd.get_dummies(
    df,
    columns=categorical_cols,
    dtype=int,
    drop_first=True
)

# Label encoding for Target
label_encoder = LabelEncoder()
df['Target'] = label_encoder.fit_transform(df['Target'])

print("\nData after encoding:")
print(df.head())

print("\nTarget encoding:")
for label, number in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
):
    print(f"{label}: {number}")


# 4b) Feature scaling

numerical_cols = [
    'Application order',
    'Age at enrollment',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

# scaler = StandardScaler()

# df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# print("\nData after feature scaling:")
# print(df[numerical_cols].head())

# 5) Data splitting

# Separate features (X) and target (y)
X = df.drop('Target', axis=1)
y = df['Target']

# Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train[numerical_cols]),
    columns=numerical_cols,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test[numerical_cols]),
    columns=numerical_cols,
    index=X_test.index
)

print("\nTraining set:")
print(X_train.shape)

print("\nTesting set:")
print(X_test.shape)

print("\nTraining target:")
print(y_train.shape)

print("\nTesting target:")
print(y_test.shape)

# 6 PCA

pca_cols = [
    col for col in numerical_cols
    if col != "Application order"
]

pca_full = PCA()

pca_full.fit(X_train_scaled[pca_cols])

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

variance_table = pd.DataFrame({
    "Component": range(1, len(explained_variance) + 1),
    "Explained variance (%)": explained_variance * 100,
    "Cumulative variance (%)": cumulative_variance * 100
})

print("\nPCA explained variance:")
print(variance_table.round(2))


# Final PCA transformation using the selected 10 components
pca = PCA(n_components=10)

X_train_pca_array = pca.fit_transform(
    X_train_scaled[pca_cols]
)

X_test_pca_array = pca.transform(
    X_test_scaled[pca_cols]
)

component_names = [
    f"PC{i}" for i in range(1, 11)
]

X_train_pca = pd.DataFrame(
    X_train_pca_array,
    columns=component_names,
    index=X_train.index
)

X_test_pca = pd.DataFrame(
    X_test_pca_array,
    columns=component_names,
    index=X_test.index
)

print("\nOriginal numerical dimensions:", len(pca_cols))
print("Dimensions after PCA:", pca.n_components_)
print(
    "Variance retained:",
    f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
)

print("Training shape after PCA:", X_train_pca.shape)
print("Testing shape after PCA:", X_test_pca.shape)