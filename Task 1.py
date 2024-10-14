import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

df_train = pd.read_csv('Titanic-Dataset.csv')
print(df_train.head())

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
df_train['IsAlone'] = (df_train['FamilySize'] == 1).astype(int)
df_train['Title'] = df_train['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

df_train['Title'] = df_train['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                               'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                               'Jonkheer', 'Dona'], 'Rare')
df_train['Title'] = df_train['Title'].replace('Mlle', 'Miss')
df_train['Title'] = df_train['Title'].replace('Ms', 'Miss')
df_train['Title'] = df_train['Title'].replace('Mme', 'Mrs')

print(df_train.head())

print(df_train.describe())

plt.figure(figsize=(6, 4))
df_train['Survived'].value_counts().plot(kind='bar', color=['salmon', 'lightgreen'])
plt.title('Passenger Survival Distribution')
plt.xlabel('Survived (0: No, 1: Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
df_train['Sex'].value_counts().plot(kind='bar', color=['lightblue', 'pink'])
plt.title('Passenger Sex Distribution')
plt.xticks(rotation=0)
plt.xlabel('Gender')
plt.ylabel('Count')

#Text(0, 0.5, 'Count')

plt.subplot(1, 2, 2)
df_train['Embarked'].value_counts().plot(kind='bar', color=['gold', 'blue', 'red'])
plt.title('Passenger Embarkation Distribution (C = Cherbourg; Q = Queenstown; S = Southampton)')
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print(df_train.isnull().sum())

missing_values_df = (df_train.isnull().sum() / len(df_train)) * 100
print(missing_values_df.to_frame('Percentage Missing').sort_values(by='Percentage Missing', ascending=False))

numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])

categorical_features = ['Sex', 'Embarked', 'Title', 'IsAlone']
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features),('cat', categorical_transformer, categorical_features)])

X = df_train[numeric_features + categorical_features]
y = df_train['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train, y_train)

y_pred = best_model.predict(X_val) # type: ignore

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
classification_rep = classification_report(y_val, y_pred)

print("Accuracy:", accuracy," - ",round(accuracy*100,2),"%")
print("Precision:", precision," - ",round(precision*100,2),"%")
print("Recall:", recall," - ",round(recall*100,2),"%")
print("F1 Score:", f1," - ",round(f1*100,2),"%")
print("Classification Report:\n", classification_rep)

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy') # type: ignore
print("Cross-validation Accuracy:", cv_scores.mean())
