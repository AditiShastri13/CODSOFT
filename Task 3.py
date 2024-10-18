import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

file_path = 'IRIS.csv'
data = pd.read_csv(file_path)

data.shape
data.head(10)

data.describe()
data.info()

data.duplicated().sum()
data.drop_duplicates(inplace=True)
data.duplicated().sum()

data.isnull().sum()

for column in data.columns:

    print(f'{column}: {data[column].nunique()}')

data['species'].value_counts()

color = sns.color_palette('RdBu')
plt.pie(data['species'].value_counts(), labels=data['species'].unique(), autopct='%1.f%%',colors=color)
plt.title('Distribution of species')
plt.axis('equal')
plt.show()

le =LabelEncoder()
data['species'] = le.fit_transform(data['species'])
print(le.classes_) #to print clasess in order oo transfor to 0 1 2

data.sample(10)

plt.figure(figsize=(10,5))
sns.heatmap(data.corr(), annot=True,linewidths=2)
plt.title('Correlation Matrix')
plt.show()

data_x = data.drop('species', axis=1)
data_y = data['species']

xtrain, xtest, ytrain, ytest = train_test_split(data_x, data_y, test_size=0.1, random_state=42)

def Regression_model(model):

    model.fit(xtrain, ytrain)

    train_daTa_accuracy = model.score(xtrain, ytrain) #train daTa

    test_daTa_accuracy = model.score(xtest, ytest) #test daTa

    print('model name : ',model)

    print('accuracy of train data = ',train_daTa_accuracy)

    print('accuracy of test data = ',test_daTa_accuracy)

# RandomForestRegressor model
model = RandomForestRegressor()
Regression_model(model)

# DecisionTreeClassifier model
model = DecisionTreeClassifier()
Regression_model(model)
