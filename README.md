# Weather Suitability Prediction Using Machine Learning

## 1. Project Overview

This project focuses on developing a machine learning system to predict weather suitability using historical weather data.

The project uses the Szeged Weather History Dataset, which contains approximately 10 years of hourly weather observations. The data includes information such as temperature, humidity, wind speed, wind direction, visibility, atmospheric pressure, and precipitation type.

The main objective is to build and compare supervised machine learning models that can predict whether a weather condition is Clear or Not Clear.

The project covers the complete machine learning workflow:

- Dataset loading
- Data understanding and inspection
- Data preprocessing
- Missing-value handling
- Duplicate removal
- Feature engineering
- Target-variable creation
- Categorical encoding
- Exploratory Data Analysis (EDA)
- Train/test splitting
- Feature scaling
- Class-imbalance handling
- Machine learning model development
- Hyperparameter tuning
- Model evaluation
- Prediction on new weather observations


# 2. Project Objective

The main objective is to develop a classification model that predicts:

0 = Not Clear
1 = Clear

The prediction is made using measurable weather characteristics rather than directly providing the weather description to the model.

The main input variables are:

- Temperature
- Apparent_Temperature
- Humidity
- Wind_Speed
- Wind_Bearing
- Visibility
- Pressure
- Precip_Rain
- Precip_Snow

The final feature matrix contains 9 input features.


# 3. Project Structure

The project is organized into a shared group pipeline and individual student directories.

project/
|
|-- group_pipeline/
|   |
|   |-- README.md
|   |
|   |-- data/
|   |   |-- raw/
|   |   |   |-- weatherHistory.csv
|   |   |
|   |   |-- processed/
|   |       |-- weatherHistoryProcessed.csv
|   |
|   |-- notebooks/
|   |   |-- weather_suitability.ipynb
|   |
|   |-- output/
|       |-- eda/
|       |-- models/
|       |-- predictions/
|
|-- T25101195/
|-- T25102052/
|-- T25102202/
|-- T25103007/
|-- T25300054/
|-- T25300254/


Directory Description:

group_pipeline/
    Main shared machine learning pipeline.

data/raw/
    Contains the original weatherHistory.csv dataset.

data/processed/
    Contains cleaned and processed datasets.

notebooks/
    Contains Jupyter notebooks used for preprocessing, EDA, model development and evaluation.

output/eda/
    Contains EDA visualizations.

output/models/
    Contains model results and model-related outputs.

output/predictions/
    Contains prediction results.

T25.../
    Individual group-member directories.


# 4. Dataset

Dataset Name:

weatherHistory.csv

The dataset contains hourly weather observations collected over approximately 10 years.

The original dataset contains approximately:

Rows: 96,453
Columns: 12

The raw dataset is stored at:

group_pipeline/data/raw/weatherHistory.csv

The raw dataset should remain unchanged.


# 5. Original Dataset Columns

The original dataset contains the following columns:

Formatted Date
    Date and time of the weather observation.

Summary
    General weather condition.

Precip Type
    Type of precipitation.

Temperature (C)
    Temperature in Celsius.

Apparent Temperature (C)
    Feels-like temperature.

Humidity
    Relative humidity.

Wind Speed (km/h)
    Wind speed.

Wind Bearing (degrees)
    Wind direction.

Visibility (km)
    Visibility distance.

Loud Cover
    Loud cover measurement.

Pressure (millibars)
    Atmospheric pressure.

Daily Summary
    Daily weather summary.


# 6. Technologies Used

Python
    Main programming language.

Pandas
    Data manipulation and preprocessing.

NumPy
    Numerical operations.

Matplotlib
    Data visualization.

Seaborn
    Statistical visualization.

Scikit-learn
    Machine learning algorithms and evaluation.

Imbalanced-learn
    SMOTE and class-imbalance handling.

TensorFlow / Keras
    Artificial Neural Network development.

Jupyter Notebook
    Interactive development and experimentation.

VS Code
    Development environment.


# 7. Environment Setup

## 7.1 Recommended Python Version

Python 3.11 is recommended, especially when TensorFlow is used for the Artificial Neural Network.

Check the installed Python version:

python --version

Example:

Python 3.11.x


# 8. Open the Project

Open the project directory in VS Code.

Navigate to the main group pipeline:

cd project/group_pipeline


# 9. Create a Virtual Environment

Create a virtual environment:

python -m venv .venv

Activate the environment on Windows:

.venv\Scripts\activate

After activation, the terminal should show:

(.venv)


# 10. Install Required Libraries

Install the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow jupyter openpyxl

Alternatively, create a requirements.txt file containing:

pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
tensorflow
jupyter
openpyxl

Then install the dependencies:

pip install -r requirements.txt


# 11. Open the Jupyter Notebook

The main notebook is located at:

group_pipeline/notebooks/weather_suitability.ipynb

Start Jupyter Notebook:

jupyter notebook

Alternatively, open the notebook directly using VS Code.

Select the project's .venv as the Python kernel.


# 12. Dataset Loading

The notebook is located inside:

group_pipeline/notebooks/

The raw dataset is located inside:

group_pipeline/data/raw/

Therefore, the dataset can be loaded using:

import pandas as pd

df = pd.read_csv("../data/raw/weatherHistory.csv")

print(df.shape)
print(df.head())

Expected original dimensions:

(96453, 12)


# 13. Initial Data Inspection

The dataset is inspected before preprocessing.

Display the first records:

df.head()

Display dataset information:

df.info()

Display statistical information:

df.describe()

Check missing values:

df.isnull().sum()

Check duplicate records:

df.duplicated().sum()

Check weather summary categories:

df["Summary"].value_counts()


The initial inspection helps identify:

- Data types
- Missing values
- Duplicate records
- Numerical variables
- Categorical variables
- Weather-condition distribution
- Preprocessing requirements


# 14. Data Preprocessing

The preprocessing workflow is:

Raw weatherHistory.csv
        |
        v
Load Dataset
        |
        v
Initial Inspection
        |
        v
Remove Loud Cover
        |
        v
Rename Columns
        |
        v
Convert Date
        |
        v
Handle Missing Values
        |
        v
Remove Duplicates
        |
        v
Create Suitability Target
        |
        v
Prevent Target Leakage
        |
        v
Encode Precipitation Type
        |
        v
Select Features
        |
        v
Train/Test Split
        |
        v
Feature Scaling
        |
        v
Class Imbalance Handling
        |
        v
Machine Learning Models
        |
        v
Evaluation


# 15. Remove Unnecessary Column

The Loud Cover column is not required for the prediction task.

It is removed using:

df = df.drop(columns=["Loud Cover"])


# 16. Rename Columns

The original column names contain spaces and measurement units.

They are renamed to simpler names:

df = df.rename(columns={
    "Formatted Date": "Date",
    "Precip Type": "Precip_Type",
    "Temperature (C)": "Temperature",
    "Apparent Temperature (C)": "Apparent_Temperature",
    "Wind Speed (km/h)": "Wind_Speed",
    "Wind Bearing (degrees)": "Wind_Bearing",
    "Visibility (km)": "Visibility",
    "Pressure (millibars)": "Pressure",
    "Daily Summary": "Daily_Summary"
})


The new names are easier to use in Python and machine learning algorithms.


# 17. Convert Date Column

The Date column is converted to a Pandas datetime object:

df["Date"] = pd.to_datetime(df["Date"], utc=True)

This allows temporal information to be extracted if required.

For example:

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Hour"] = df["Date"].dt.hour
df["Day_of_Week"] = df["Date"].dt.dayofweek

The final machine learning feature set does not directly use the original Date column.


# 18. Handle Missing Values

The dataset contains missing values, particularly in Precip_Type.

Forward filling is used:

df["Precip_Type"] = df["Precip_Type"].ffill()

If missing values remain at the beginning of the dataset, backward filling can be applied:

df["Precip_Type"] = df["Precip_Type"].bfill()

The result can be checked using:

print(df["Precip_Type"].isnull().sum())


# 19. Remove Duplicate Records

Duplicate records are removed:

df = df.drop_duplicates()

Check the result:

print("Duplicate rows:", df.duplicated().sum())

The expected result is:

Duplicate rows: 0


# 20. Create Target Variable

The target variable is called Suitability.

It is created from the Summary column.

A weather observation is assigned 1 when the Summary is exactly Clear.

df["Suitability"] = (
    df["Summary"] == "Clear"
).astype(int)

The target classes are:

0 = Not Clear
1 = Clear


# 21. Target Leakage Prevention

The target is created using the Summary column:

Suitability = Summary == "Clear"

Therefore, Summary must NOT be used as an input feature.

If Summary is included in the model input, the model can directly learn the information used to generate the target.

This is called target leakage.

Therefore, Summary is excluded from X.

The model must learn the relationship between measurable weather parameters and the target.


# 22. Precipitation Type Encoding

Precip_Type is a categorical variable.

Typical values include:

rain
snow

Machine learning algorithms require numerical input, so one-hot encoding is applied.

df = pd.get_dummies(
    df,
    columns=["Precip_Type"],
    prefix="Precip",
    dtype=int
)

This produces:

Precip_Rain
Precip_Snow

Example:

Precip_Rain = 1
Precip_Snow = 0

means rain.

Precip_Rain = 0
Precip_Snow = 1

means snow.


# 23. Final Feature Selection

The final input variables are:

Temperature
Apparent_Temperature
Humidity
Wind_Speed
Wind_Bearing
Visibility
Pressure
Precip_Rain
Precip_Snow

The feature list is:

features = [
    "Temperature",
    "Apparent_Temperature",
    "Humidity",
    "Wind_Speed",
    "Wind_Bearing",
    "Visibility",
    "Pressure",
    "Precip_Rain",
    "Precip_Snow"
]

Create X and y:

X = df[features]
y = df["Suitability"]

Therefore:

X = 9 input features
y = Suitability target


# 24. Handle Remaining Missing Values

Check the selected features:

print(X.isnull().sum())

If numerical missing values remain:

X = X.fillna(X.median())

Check the target:

print(y.isnull().sum())


# 25. Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the data before model training.

The EDA includes:

- Target distribution
- Temperature distribution
- Apparent temperature distribution
- Humidity distribution
- Wind speed distribution
- Visibility distribution
- Pressure distribution
- Feature relationships
- Correlation analysis
- Class distribution


# 26. Target Distribution

The target distribution is checked using:

print(y.value_counts())

A bar chart can be created:

import matplotlib.pyplot as plt

y.value_counts().plot(kind="bar")

plt.xlabel("Suitability")
plt.ylabel("Count")
plt.title("Weather Suitability Distribution")

plt.show()

The dataset contains significantly more Not Clear observations than Clear observations.

Therefore, the dataset has a class imbalance problem.


# 27. Numerical Feature Distributions

Histograms can be used to understand numerical features.

Example:

plt.hist(df["Temperature"], bins=30)

plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.title("Temperature Distribution")

plt.show()

The same process can be applied to:

- Apparent_Temperature
- Humidity
- Wind_Speed
- Wind_Bearing
- Visibility
- Pressure


# 28. Correlation Analysis

A correlation matrix can be created:

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

sns.heatmap(
    df[features + ["Suitability"]].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Matrix")

plt.show()

EDA visualizations should be saved inside:

group_pipeline/output/eda/


# 29. Train/Test Split

The dataset is divided into training and testing datasets.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Parameters:

test_size = 0.2
    20% is used for testing.

random_state = 42
    Makes the split reproducible.

stratify = y
    Maintains the class distribution.

Approximately:

80% = Training
20% = Testing


# 30. Feature Scaling

The features have different numerical ranges.

For example:

Temperature
Humidity
Wind Speed
Wind Bearing
Visibility
Pressure

have different units and ranges.

StandardScaler is used:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

The scaler is fitted only on the training data.

The test data is transformed using the scaler learned from the training data.

This prevents data leakage.


# 31. Class Imbalance

The dataset contains significantly more Not Clear observations than Clear observations.

Therefore, accuracy alone is not sufficient for evaluating the models.

The following metrics are considered:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Special attention is given to the Clear class.


# 32. SMOTE

SMOTE stands for:

Synthetic Minority Over-sampling Technique

It is used to create synthetic samples for the minority class.

SMOTE is applied only to the training data:

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

The testing data is NOT SMOTE-resampled.

Correct workflow:

Original Dataset
       |
       v
Train/Test Split
       |
       +----------------+
       |                |
       v                v
   Training          Testing
       |
       v
     SMOTE
       |
       v
Balanced Training

The testing data remains unchanged.


# 33. Machine Learning Models

The project can evaluate multiple supervised learning models.

The main models include:

1. Logistic Regression
2. Support Vector Machine
3. Random Forest
4. Gradient Boosting
5. Artificial Neural Network


# 34. Logistic Regression

Logistic Regression is used as a baseline classification algorithm.

It predicts the probability of an observation belonging to a particular class.

It is useful as a simple reference model for comparison with more complex algorithms.


# 35. Support Vector Machine

Support Vector Machine (SVM) attempts to find a decision boundary that separates the two classes.

Possible kernels include:

- Linear
- RBF

Class weighting can also be used to address class imbalance.


# 36. Random Forest

Random Forest is an ensemble machine learning algorithm based on multiple decision trees.

It can capture nonlinear relationships between weather variables.

Random Forest can be trained using:

- Original training data
- Class weighting
- SMOTE-balanced training data

Hyperparameters can be optimized using GridSearchCV.


# 37. Gradient Boosting

Gradient Boosting builds models sequentially.

Each new model attempts to improve the errors made by previous models.

Gradient Boosting is useful for learning complex nonlinear relationships between weather parameters.


# 38. Artificial Neural Network

An Artificial Neural Network is also used to model nonlinear relationships.

A typical architecture used in the project is:

Input Layer
    |
    v
Dense Layer
64 neurons
ReLU
    |
    v
Dropout
    |
    v
Dense Layer
32 neurons
Tanh
    |
    v
Dropout
    |
    v
Dense Layer
8 neurons
ReLU
    |
    v
Output Layer
1 neuron
Sigmoid


# 39. ANN Input Shape

The final dataset contains 9 features.

Therefore, the ANN input shape is obtained dynamically:

input_features = X_train_scaled.shape[1]

print("Number of input features:", input_features)

Expected output:

Number of input features: 9

The input layer can then be created using:

layers.Input(shape=(input_features,))


# 40. ANN Compilation

For binary classification, the ANN can use:

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

Adam is used as the optimizer.

Binary crossentropy is used as the loss function.

Accuracy is used as one evaluation metric.


# 41. Cross-Validation

Stratified K-Fold cross-validation can be used during model selection and hyperparameter tuning.

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

Five folds are used while maintaining the class distribution.


# 42. Hyperparameter Tuning

GridSearchCV can be used to identify suitable model hyperparameters.

Example:

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

F1-score is used as the scoring metric because the dataset is imbalanced and both precision and recall are important.


# 43. Model Evaluation

After training, predictions are generated:

y_pred = model.predict(X_test)

The following metrics are used:

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# 44. Accuracy

Accuracy represents the proportion of correctly classified observations.

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

Because the dataset is imbalanced, accuracy should not be considered by itself.


# 45. Precision

Precision measures how many observations predicted as Clear were actually Clear.

Precision = True Positive / (True Positive + False Positive)

High precision means the model produces fewer false Clear predictions.


# 46. Recall

Recall measures how many actual Clear observations were successfully detected.

Recall = True Positive / (True Positive + False Negative)

Recall is important when the objective is to identify as many Clear weather observations as possible.


# 47. F1-Score

F1-score combines precision and recall.

F1 = 2 x (Precision x Recall) / (Precision + Recall)

F1-score is especially useful when dealing with imbalanced classes.


# 48. Classification Report

The classification report can be generated using:

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Clear", "Clear"]
    )
)

The report contains:

- Precision
- Recall
- F1-score
- Support
- Accuracy
- Macro average
- Weighted average


# 49. Confusion Matrix

The confusion matrix shows the classification results.

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

It can also be visualized:

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Clear", "Clear"]
).plot()

plt.title("Confusion Matrix")

plt.show()


The confusion matrix contains:

Actual Not Clear + Predicted Not Clear
    = True Negative

Actual Not Clear + Predicted Clear
    = False Positive

Actual Clear + Predicted Not Clear
    = False Negative

Actual Clear + Predicted Clear
    = True Positive


# 50. Model Comparison

The models should be compared using a common evaluation table.

Example:

Model                  Accuracy    Precision    Recall    F1-Score

Logistic Regression       -           -           -          -

SVM                       -           -           -          -

Random Forest             -           -           -          -

Gradient Boosting         -           -           -          -

ANN                       -           -           -          -

The final values should be populated using the results from the notebook.


# 51. Prediction on New Weather Data

New weather observations can be passed to the trained model.

The input must contain the same 9 features used during training:

1. Temperature
2. Apparent_Temperature
3. Humidity
4. Wind_Speed
5. Wind_Bearing
6. Visibility
7. Pressure
8. Precip_Rain
9. Precip_Snow


Example:

sample = [[
    20.5,
    20.1,
    0.60,
    10.2,
    180,
    12.5,
    1015.2,
    0,
    0
]]

The sample must be scaled using the same scaler:

sample_scaled = scaler.transform(sample)

The trained model can then generate the prediction.


# 52. Prediction Interpretation

For binary classification:

0 = Not Clear
1 = Clear

For an ANN with a sigmoid output, the model produces a probability between 0 and 1.

Example:

Probability = 0.82

Using a threshold of 0.5:

0.82 >= 0.5

Therefore:

Prediction = Clear

Another example:

Probability = 0.31

Since:

0.31 < 0.5

the prediction is:

Prediction = Not Clear


# 53. Output Directory

Generated files should be organized inside:

group_pipeline/output/

Recommended structure:

output/
|
|-- eda/
|   |-- temperature_distribution.png
|   |-- humidity_distribution.png
|   |-- wind_speed_distribution.png
|   |-- visibility_distribution.png
|   |-- correlation_matrix.png
|
|-- models/
|   |-- model_results.csv
|   |-- confusion_matrix.png
|   |-- classification_reports.txt
|
|-- predictions/
    |-- predictions.csv


# 54. Saving the Processed Dataset

The processed dataset should be saved to:

group_pipeline/data/processed/weatherHistoryProcessed.csv

Example:

df.to_csv(
    "../data/processed/weatherHistoryProcessed.csv",
    index=False
)


# 55. Reproducibility

The project uses:

random_state = 42

where applicable.

This provides reproducibility for:

- Train/test splitting
- SMOTE
- Cross-validation
- Machine learning experiments

Using the same dataset, preprocessing procedure, random seed, and model configuration should produce consistent results.


# 56. Important Data Handling Rules

## Rule 1: Keep the Raw Dataset Unchanged

Do not modify:

data/raw/weatherHistory.csv

All preprocessing should be performed on a DataFrame loaded from the raw dataset.


## Rule 2: Do Not Include Summary in the Features

The target is generated from:

Suitability = Summary == "Clear"

Therefore, Summary must not be included in X.

Otherwise, target leakage will occur.


## Rule 3: Do Not Apply SMOTE to Test Data

Correct:

Training Data -> SMOTE
Testing Data  -> Original

Incorrect:

Training Data -> SMOTE
Testing Data  -> SMOTE


## Rule 4: Fit the Scaler Only on Training Data

Correct:

scaler.fit(X_train)

Then:

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

Do not fit the scaler using the complete dataset before splitting.


## Rule 5: Use the Same Features During Prediction

The prediction input must contain the same features and feature order used during training.

Final feature order:

1. Temperature
2. Apparent_Temperature
3. Humidity
4. Wind_Speed
5. Wind_Bearing
6. Visibility
7. Pressure
8. Precip_Rain
9. Precip_Snow


# 57. Complete Preprocessing Code

import pandas as pd

# Load Dataset
df = pd.read_csv("../data/raw/weatherHistory.csv")

print("Original shape:", df.shape)


# Remove unnecessary column
df = df.drop(columns=["Loud Cover"])


# Rename columns
df = df.rename(columns={
    "Formatted Date": "Date",
    "Precip Type": "Precip_Type",
    "Temperature (C)": "Temperature",
    "Apparent Temperature (C)": "Apparent_Temperature",
    "Wind Speed (km/h)": "Wind_Speed",
    "Wind Bearing (degrees)": "Wind_Bearing",
    "Visibility (km)": "Visibility",
    "Pressure (millibars)": "Pressure",
    "Daily Summary": "Daily_Summary"
})


# Convert Date
df["Date"] = pd.to_datetime(
    df["Date"],
    utc=True
)


# Handle missing precipitation type
df["Precip_Type"] = df["Precip_Type"].ffill()
df["Precip_Type"] = df["Precip_Type"].bfill()


# Remove duplicate rows
df = df.drop_duplicates()


# Create target
df["Suitability"] = (
    df["Summary"] == "Clear"
).astype(int)


# One-hot encode precipitation type
df = pd.get_dummies(
    df,
    columns=["Precip_Type"],
    prefix="Precip",
    dtype=int
)


# Define features
features = [
    "Temperature",
    "Apparent_Temperature",
    "Humidity",
    "Wind_Speed",
    "Wind_Bearing",
    "Visibility",
    "Pressure",
    "Precip_Rain",
    "Precip_Snow"
]


# Create X and y
X = df[features]
y = df["Suitability"]


# Handle remaining numerical missing values
X = X.fillna(X.median())


# Display final information
print("Final X shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# 58. Complete Machine Learning Workflow

The complete workflow is:

weatherHistory.csv
        |
        v
Data Loading
        |
        v
Initial Inspection
        |
        v
Data Cleaning
        |
        v
Missing-Value Handling
        |
        v
Duplicate Removal
        |
        v
Feature Engineering
        |
        v
Target Creation
        |
        v
Target Leakage Prevention
        |
        v
Categorical Encoding
        |
        v
Feature Selection
        |
        v
Train/Test Split
        |
        v
Feature Scaling
        |
        v
Class Imbalance Handling
        |
        +------------------+
        |                  |
        v                  v
Original Training       SMOTE
        |                  |
        +--------+---------+
                 |
                 v
        Machine Learning Models
                 |
                 v
        Hyperparameter Tuning
                 |
                 v
          Model Evaluation
                 |
        +--------+--------+
        |        |        |
        v        v        v
    Accuracy Precision Recall
                 |
                 v
             F1-Score
                 |
                 v
        Confusion Matrix
                 |
                 v
          Model Comparison
                 |
                 v
          Final Prediction


# 59. How to Run the Project

Step 1:
Open the project in VS Code.

Step 2:
Navigate to the group pipeline:

cd group_pipeline

Step 3:
Activate the virtual environment:

.venv\Scripts\activate

Step 4:
Install dependencies:

pip install -r requirements.txt

Step 5:
Make sure the dataset exists:

data/raw/weatherHistory.csv

Step 6:
Open:

notebooks/weather_suitability.ipynb

Step 7:
Select the .venv Python kernel.

Step 8:
Run the notebook from the first cell to the final prediction cell.


# 60. Google Colab Setup

The project can also be executed using Google Colab.

Mount Google Drive:

from google.colab import drive

drive.mount("/content/drive")

If the dataset is stored at:

MyDrive/AIML-Project/G1/weatherHistory.csv

load it using:

import pandas as pd

file_path = (
    "/content/drive/MyDrive/"
    "AIML-Project/G1/weatherHistory.csv"
)

df = pd.read_csv(file_path)

print(df.shape)

If required, install additional packages:

!pip install imbalanced-learn


# 61. Expected Final Deliverables

The group_pipeline directory should contain:

group_pipeline/
|
|-- README.md
|
|-- data/
|   |-- raw/
|   |   |-- weatherHistory.csv
|   |
|   |-- processed/
|       |-- weatherHistoryProcessed.csv
|
|-- notebooks/
|   |-- weather_suitability.ipynb
|
|-- output/
    |-- eda/
    |-- models/
    |-- predictions/


The notebook should demonstrate:

- Data loading
- Data inspection
- Data cleaning
- Missing-value handling
- Duplicate removal
- Feature engineering
- Target creation
- Target leakage prevention
- One-hot encoding
- Exploratory Data Analysis
- Train/test splitting
- Feature scaling
- Class imbalance handling
- Model training
- Hyperparameter tuning
- Model evaluation
- Prediction


# 62. Final Project Goal

The final goal of the project is to develop a reliable binary classification system capable of predicting whether a weather observation is:

Clear

or:

Not Clear

based on measurable weather parameters.

The model does not use Summary as an input feature because Summary is used to create the target variable.

The final model uses:

Temperature
Apparent_Temperature
Humidity
Wind_Speed
Wind_Bearing
Visibility
Pressure
Precip_Rain
Precip_Snow

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Because of the class imbalance, particular attention is given to the Precision, Recall, and F1-score of the Clear class.


# 63. Conclusion

This project demonstrates a complete machine learning pipeline for weather classification, starting from raw historical weather data and progressing through preprocessing, exploratory analysis, feature engineering, class-imbalance handling, model development, hyperparameter optimization, and evaluation.

The group_pipeline directory keeps the shared dataset, notebooks, processed data, and model outputs organized and reproducible.

The individual T25... directories remain separate for the respective group members.