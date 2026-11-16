Weather Suitability Prediction Using Machine Learning
A collaborative machine learning project that analyses historical weather observations and develops classification models to identify whether the observed weather condition is Clear or Not Clear.
Project type: Academic group project
Focus areas: Data preprocessing, exploratory data analysis, feature engineering, supervised machine learning, and artificial neural networks
Primary language: Python

Table of Contents
- Project Overview
- Problem Statement and Objective
- Dataset
- Machine Learning Workflow
- Models Developed
- Repository Structure
- Technologies and Libraries
- Getting Started
- Running the Notebooks
- Group Members and Contributions
- Important Notes
Project Overview
Weather conditions influence many everyday activities and decisions. This project explores how historical weather observations can be used to classify the observed weather condition as Clear or Not Clear.

The project uses the Szeged Weather History Dataset and follows a collaborative workflow. The group pipeline covers the shared data analysis and machine learning process, while individual notebooks document the preprocessing tasks and model-development work completed by group members.

The work includes data inspection, cleaning, exploratory analysis, feature preparation, model training, and model evaluation.
Problem Statement and Objective
The project addresses a binary classification problem.

The target variable is defined from the dataset's Summary column:
Target value	Class
1	Clear
0	Not Clear


An observation is labelled Clear when its Summary value is exactly Clear. Other summary categories are treated as Not Clear.

The objective is to develop and examine machine learning classifiers that learn from weather-related input features and classify the corresponding observation.

Task clarification: Since the label is derived from the weather summary for the same observation, this project is a classification of the observed/current weather condition (a nowcasting-style task), not a prediction of weather at a future time.
Dataset
- Dataset file: weatherHistory.csv
- Dataset name: Szeged Weather History Dataset
- Observations: Approximately 96,453 rows in the original dataset
- Original attributes: 12 columns
The dataset contains hourly weather observations and includes attributes such as:
- Formatted Date
- Summary
- Precip Type
- Temperature (C)
- Apparent Temperature (C)
- Humidity
- Wind Speed (km/h)
- Wind Bearing (degrees)
- Visibility (km)
- Loud Cover
- Pressure (millibars)
- Daily Summary
The original dataset is stored in:
group_pipeline/data/raw/weatherHistory.csv
The Summary field is used to construct the target label. It must not be included as a model input because doing so would reveal the information used to create the target (target leakage).
Machine Learning Workflow
The project covers the following stages:
1. Data loading and inspection – load the weather dataset and examine its structure, data types, summary statistics, and class distribution.
2. Data cleaning – investigate missing values, identify duplicate records, and review columns that are not required.
3. Missing-value handling – apply appropriate handling to missing data.
4. Duplicate detection and removal – identify and remove duplicate observations where appropriate.
5. Categorical encoding – prepare categorical variables for machine learning.
6. Outlier detection and treatment – inspect numerical features and investigate outliers.
7. Feature engineering and selection – prepare and select relevant weather-related features.
8. Exploratory Data Analysis (EDA) – explore feature distributions and relationships using visualisations.
9. Feature scaling and dimensionality reduction – apply scaling and explore Principal Component Analysis (PCA).
10. Model development – train and examine multiple classification algorithms.
11. Model evaluation – assess model performance using suitable classification evaluation methods.
The exact preprocessing and modelling choices may differ between individual experiments; refer to each notebook for its implementation and results.
Models Developed
The repository contains individual model-development notebooks for the following approaches:
- Artificial Neural Network (ANN)
- Support Vector Machine (SVM)
- Logistic Regression
- K-Nearest Neighbours (KNN)
- Decision Tree
- Gradient Boosting
The notebooks provide the implementation and evaluation for each model. No single performance score is stated here because results should be read directly from the corresponding notebook outputs.
Repository Structure
Weather-Suitability-Prediction-ML/
│
├── group_pipeline/
│   ├── data/
│   │   └── raw/
│   │       └── weatherHistory.csv
│   ├── notebooks/
│   │   ├── group_pipeline.ipynb
│   │   └── group-pipeline-ann.ipynb
│   └── results/
│       └── eda_visualizations/
│           └── group_pipeline_correlation.png
│
├── Individual_work/
│   ├── IT25101195_Scaling_and_PCA.ipynb
│   ├── IT25102052_Missing_Data_Handling.ipynb
│   ├── IT25102202_Outlier_Detection_Treatment.ipynb
│   ├── IT25103007_Duplicate_Detection_Removal.ipynb
│   ├── IT25300054_Categorical_Encoding.ipynb
│   └── IT25300254_Feature_Engineering_Selection.ipynb
│
├── Model Development/
│   ├── IT25101195-ANN-Best-Model.ipynb
│   ├── IT25102052_SVM_improved.ipynb
│   ├── IT25102202_Logistic_Regression.ipynb
│   ├── IT25103007_KNN.ipynb
│   ├── IT25300054_decisionTree.ipynb
│   └── IT25300254-GradientBoosting.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
The tree above describes the organised project layout. Keep the actual folder and notebook names in the repository consistent with the files you upload.

Technologies and Libraries
Technology / Library	Purpose
Python	Main programming language
Jupyter Notebook	Interactive development and experiments
Pandas	Data manipulation and analysis
NumPy	Numerical operations
Matplotlib	Data visualisation
Seaborn	Statistical visualisation
Scikit-learn	Machine learning, preprocessing, and evaluation
Imbalanced-learn	Class-imbalance handling techniques
TensorFlow / Keras	Artificial Neural Network development


Getting Started
Prerequisites
- Python installed on your computer
- Git (if cloning the repository)
- Jupyter Notebook or Visual Studio Code with the Jupyter extension
Python 3.11 is a suggested environment for this project. TensorFlow compatibility depends on the operating system and Python version.
1. Clone the repository
Replace <YOUR-GITHUB-USERNAME> with the GitHub account that hosts this repository.
git clone https://github.com/<YOUR-GITHUB-USERNAME>/Weather-Suitability-Prediction-ML.git
2. Open the project directory
cd Weather-Suitability-Prediction-ML
3. Create a virtual environment
python -m venv .venv
4. Activate the virtual environment
Windows:
.venv\Scripts\activate
macOS / Linux:
source .venv/bin/activate
5. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
If TensorFlow installation fails, check the TensorFlow installation requirements for your operating system and Python version.
Running the Notebooks
Start Jupyter Notebook from the repository root:
jupyter notebook
Or open the repository in Visual Studio Code and select the .venv environment as the notebook kernel.

Start with the shared notebooks in:
group_pipeline/notebooks/
The raw dataset is located at:
group_pipeline/data/raw/weatherHistory.csv
Notebook dataset paths may be relative to the notebook's location. If a notebook cannot locate the CSV file, check its working directory and update the relative path accordingly.

Run the individual preprocessing and model-development notebooks from their respective folders to review each group member's work.
Group Members and Contributions
This project was completed collaboratively by the following group members.
Student ID	Contribution
IT25101195	Scaling and PCA; Artificial Neural Network (ANN)
IT25102052	Missing Data Handling; Support Vector Machine (SVM)
IT25102202	Outlier Detection and Treatment; Logistic Regression
IT25103007	Duplicate Detection and Removal; K-Nearest Neighbours (KNN)
IT25300054	Categorical Encoding; Decision Tree
IT25300254	Feature Engineering and Selection; Gradient Boosting


Important Notes
- The notebooks are the primary record of the implementation, experiments, and model results.
- Results can vary depending on preprocessing, data splitting, random states, and model configuration.
- The Summary column is used to create the target label and should be excluded from the input features to avoid target leakage.
- The project classifies the weather condition associated with an observation; it should not be described as forecasting future weather.
- Before publishing the repository publicly, make sure the group has permission to share the work and that the dataset's terms allow redistribution.
- Do not commit passwords, API keys, private credentials, or other sensitive information.
Acknowledgement
This project was completed as a collaborative academic project at the Sri Lanka Institute of Information Technology (SLIIT).
