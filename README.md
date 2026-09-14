# ml-and-omics-practice
Hands-on implementation of machine learning algorithms and omics-related scripts, using Python and R.

## Project 1: Hand-coded Machine Learning Operators (Python)
Built without deep learning frameworks.
- LeakyReLU activation function
- DNA one-hot encoding for biological sequences
- 2D max pooling
- Gradient descent solver for nonlinear sinusoidal regression with MSE loss

### How to run
1. Put `q4data.txt` under `/python/data/`
2. Run `python/ml_basics.py`

## Project 2: Titanic Passenger Survival Prediction (R)
An end-to-end binary classification workflow.
- Data cleaning & feature engineering
- Stratified train/test split
- 5-fold cross validation
- Logistic regression and decision tree models
- Evaluation: confusion matrix, ROC and AUC

### How to run
1. Put `train.csv` under `/r-titanic/data/`
2. Source `r-titanic/titanic_analysis.R`
