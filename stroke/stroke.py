import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Drop the 'id' column
data.drop("id", axis=1, inplace=True)

# Convert 'gender' to numeric (0 for Female, 1 for Male)
def gender2num(gender):
    return 1 if gender == "Male" else 0

data['gender'] = data.gender.apply(gender2num)

# Convert 'ever_married' to numeric (0 for No, 1 for Yes)
def ever_married2num(ever_married):
    return 1 if ever_married == "Yes" else 0

data['ever_married'] = data.ever_married.apply(ever_married2num)

# Convert 'work_type' to numeric (0 for Private, 1 for Self-employed, 2 for Govt_job)
def work_type2num(work_type):
    if work_type == "Private":
        return 0
    elif work_type == "Self-employed":
        return 1
    elif work_type == "Govt_job":
        return 2

data['work_type'] = data.work_type.apply(work_type2num)

# Convert 'Residence_type' to numeric (0 for Urban, 1 for Rural)
def Residence_type2num(Residence_type):
    return 0 if Residence_type == "Urban" else 1

data['Residence_type'] = data.Residence_type.apply(Residence_type2num)

# Convert 'smoking_status' to numeric (0 for formerly smoked, 1 for never smoked, 2 for unknown)
def smoking_status2num(smoking_status):
    if smoking_status == "formerly smoked":
        return 0
    elif smoking_status == "never smoked":
        return 1
    else:
        return 2

data['smoking_status'] = data.smoking_status.apply(smoking_status2num)

# Handle missing values
data['age'].fillna(data['age'].mean(), inplace=True)
data['avg_glucose_level'].fillna(data['avg_glucose_level'].mean(), inplace=True)

data[['hypertension', 'heart_disease', 'stroke']] = data[['hypertension', 'heart_disease', 'stroke']].fillna(0).astype(int)

data['ever_married'].fillna(data['ever_married'].mode()[0], inplace=True)
data['work_type'].fillna(data['work_type'].mode()[0], inplace=True)
data['Residence_type'].fillna(data['Residence_type'].mode()[0], inplace=True)
data['smoking_status'].fillna(data['smoking_status'].mode()[0], inplace=True)

data['bmi'].fillna(data['bmi'].median(), inplace=True)

# # One-hot encode categorical columns
# data = pd.get_dummies(data, columns=['work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# # Ensure 'stroke' is the last column
# stroke_col = data.pop('stroke')
# data['stroke'] = stroke_col

# Convert the DataFrame to NumPy array for model training
data = np.array(data)
x = data[:, :-1]
y = data[:, -1]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, stratify=y)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Save the trained model to a file using pickle
with open('stroke.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# The model is now saved as 'stroke.pkl'
# You can use the following code to load the model later:
# with open('stroke.pkl', 'rb') as model_file:
#     loaded_model = pickle.load(model_file)
# predictions = loaded_model.predict(new_data)