# Online-Shopping-
Public shopping intention classification using machine learning algorithms. 

# Online Shoppers Purchase Intention — ML Classification

A notebook that predicts whether a website visitor will make a purchase (`Revenue` = True/False), comparing several classic ML algorithms.


1. Load & preprocess — read `online_shoppers_intention.csv`, drops duplicates, one-hot encodes categorical columns, splits into train/test (stratified), and scales features.
2. Explains K-Fold CV & GridSearchCV — how cross-validation and hyperparameter search work, and why F1 (not accuracy) is used given the class imbalance (~85% non-buyers).
3. Trains & tunes 6 algorithms, each with GridSearchCV where applicable:
   - Logistic Regression (no regularization, L1, L2)
   - SVM (linear & RBF kernels)
   - Decision Tree
   - Random Forest
   - XGBoost
   - K-Nearest Neighbours
4. Compares all models — leaderboard table, bar charts, and ROC curves.
5. PCA analysis — visualizes decision boundaries in 2D, checks how many components explain 95%/99% of variance, and re-tests the top 3 models on PCA-reduced features.
6. Retrains top 3 models on full features to finalize clean, non-mutated model objects.

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## How to run

1. Place `online_shoppers_intention.csv` in the same folder as the notebook.
2. Open and run `Final_File__1_.ipynb` top to bottom in Jupyter.

## Output

- Printed classification reports, confusion matrices, and F1/accuracy scores for each model
- Comparison charts (bar plots, heatmaps, ROC curves)
- PCA variance and feature-loading plots
- A final leaderboard ranking all models by Test F1

## Key takeaway

Ensemble methods (Random Forest, XGBoost) tend to outperform simpler models like Logistic Regression and KNN, suggesting purchase behavior involves non-linear feature interactions best captured by tree-based ensembles.
