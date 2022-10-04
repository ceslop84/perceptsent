# import package

import shap
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers

# load data
train_data = pd.read_csv('./titanic_train.csv', index_col=0)
test_data = pd.read_csv('./titanic_test.csv', index_col=0)
train_data.head()

def data_preprocessing(df):
    df = df.drop(columns=['Name', 'Ticket', 'Cabin'])
    
    # fill na
    df[['Age']] = df[['Age']].fillna(value=df[['Age']].mean())
    df[['Embarked']] = df[['Embarked']].fillna(value=df['Embarked'].value_counts().idxmax())
    df[['Fare']] = df[['Fare']].fillna(value=df[['Fare']].mean())
    
    # categorical features into numeric
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
    # one-hot encoding
    embarked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = df.drop('Embarked', axis=1)
    df = df.join(embarked_one_hot)
    
    return df

# train data processing
train_data = data_preprocessing(train_data)
train_data.isnull().sum()
# create data for training
x_train = train_data.drop(['Survived'], axis=1).values
# Check test data
test_data.isnull().sum()
# scale
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
# prepare y_train
y_train = train_data['Survived'].values
test_data = data_preprocessing(test_data)
x_test = test_data.values.astype(float)
# scaling
x_test = scale.transform(x_test)
# Check test data
test_data.isnull().sum()

# build mlp
model = Sequential()
model.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
# compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# fit model
model.fit(x_train, y_train, epochs=10, batch_size=64)

# compute SHAP values
explainer = shap.DeepExplainer(model, x_train)
shap_values = explainer.shap_values(x_test)

shap.summary_plot(shap_values[0], plot_type = 'bar', feature_names = test_data.columns, show=False)
plt.savefig('SHAP_summary_plot.png')

shap.initjs()
shap.force_plot(explainer.expected_value[0].numpy(), shap_values[0][0], features = test_data.columns, show=False)
plt.savefig('SHAP_force_plot.png')

shap.decision_plot(explainer.expected_value[0].numpy(), shap_values[0][0], features = test_data.iloc[0,:], feature_names = test_data.columns.tolist(), show=False)
plt.savefig('SHAP_decision_plot.png')

shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0].numpy(), shap_values[0][0], feature_names = test_data.columns, show=False)
plt.savefig('SHAP_waterfall_plot.png')