# Exploratory Data Analysis and application of Regression techniques.
***************************************************************************
Project goal - Prediction of insurance charges for customers
****************************************************************************
Problem type - Regression
****************************************************************************
Project description -
This work predicts insurance charges for customers based on their several physical characteristics and some other features.
Exploratory Data analysis (EDA) is performed for data overview, visualization and study of statistical details.
Linear regression, ADABoost Regressor and Random Forest Regressor are implemented using Python's Scikit-learn package. 
All three algorithms are appied on both train and test data.
Oridnary Least Square technique is applied from Statsmodel for prediction and to generate full statistical summary
including p-values, correlation coefficients and F-statistics for the model.
Two approaches are used for creating train and test data from the dataset.
1. Cross-vaidation - Using entire dataset and perform cross-validation.
2. Train-test splitting - Splitting the dataset, 80% for training and rest for testing.
****************************************************************************
About dataset-
****************************************************************************
Source - Kaggle (Medical Insurance Personal dataset)
Dependent variable - charges 
Independent variables -
Age - Age of customer (numeric)
Sex - Gender of customer (categorical, String type)
Region - Areas they live in (categorical, String type)
Children - Number of children they have (numeric)
BMI - Body-mass index (continous, float)
Smoker - Whether smokes or not (categorical, String type)
****************************************************************************************
To compile and execute - 
python Insurance.py
****************************************************************************************
Output type - Continous, float values
Output parameter - Cost charges projected for each customer by predictive models.
*****************************************************************************************
*Note - Update input file path with local file pathname.


***************************************************** End of file **************************************************



