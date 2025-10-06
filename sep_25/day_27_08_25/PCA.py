import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder



# Step 1: Load the dataset
df = pd.read_csv('heart.csv')

# Step 2: Separate features and target
X = df.drop(columns=['HeartDisease'])  # Features
y = df['HeartDisease']                 # Target

# Step 3: Identify categorical columns
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Step 4: One-hot encode categorical columns
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Step 5: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Step 6: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(X_pca)

# Step 7: Create a DataFrame with principal components
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['HeartDisease'] = y

print(df_pca.head(10))
