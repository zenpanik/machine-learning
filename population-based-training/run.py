from random import random
from sklearn.tree import DecisionTreeClassifier
import random
from functools import partial
import numpy as np
import itertools
from sklearn.metrics import roc_auc_score
from pbt import PopulationBasedTraining

random.seed(10)


parameters_range = {
    'criterion' : ("gini", "entropy", "log_loss"),
    'splitter' : ("best", "random"),
    'max_depth' : list(range(1, 15)) + list(range(15,25,2)) + list(range(25,51,5)),
    'min_samples_split' : list(np.arange(0.01,1,0.05)),
    'min_samples_leaf' : list(np.arange(0.01,1,0.01)),
    'min_weight_fraction_leaf' : list(np.arange(0.01,0.5,0.01)),
    'max_features' : ['sqrt', 'log2', None],
    'max_leaf_nodes' : list(range(20, 500, 10)),
    'min_impurity_decrease' : list(np.arange(0.02,5,0.02)),
    'class_weight' : [{0: 1, 1:w} for w in range(1,51)],
    'ccp_alpha' : list(np.arange(0.05,5,0.05))
}


pbt = PopulationBasedTraining(parameters_range, initial_population_size = 2, model = DecisionTreeClassifier)

from sklearn.model_selection import train_test_split
import pandas as pd

df_ = pd.read_csv('SC_data.csv')

x_columns = [x for x in df_.columns if 'FIRST_MOB' not in x]
y_col = ['default_flag']

X, X_valid, y, y_valid  = train_test_split(df_[x_columns], df_[y_col], test_size=0.33)

pbt.run(
    X = X,
    y = y,
    X_valid = X_valid,
    y_valid = y_valid,
    scoring_fcn = roc_auc_score,
    top_k = 100,
    dominance = 0.5,
    max_epochs = 10,
    mutate_condition = .05
)
