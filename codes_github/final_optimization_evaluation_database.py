# log 29Oct2022:    scoring as 'f1_macro': unweighted average of f1 per class
#                   removed SVC from the distance-based models              

####
# Import libraries and data
####

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split, StratifiedKFold

from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Categorical
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import *
from preprocess import preprocess
from helper_features import *

####
# Experiment all particles
experiment = 'all'
####


# Preprocessing
mf = pd.read_csv('qia_processed.csv', index_col = 0)

df_filtered, mf_processed, X_train, X_test, y_train,  y_test, y_labels, dict_target_inv, no_classes = preprocess(
    mf,
    rescale = 'standard', 
    outlier = 'keep', 
    imbalance = 'oversample_train', 
    filter_d= False
    )

# Initialize the estimators
clf1 = RandomForestClassifier(random_state=42)
clf2 = SVC(probability=True, random_state=42)
clf3 = LogisticRegression(random_state=42)
clf4 = DecisionTreeClassifier(random_state=42)
clf5 = KNeighborsClassifier()
clf6 = MultinomialNB()
clf7 = GradientBoostingClassifier(random_state=42)
clf8 = XGBClassifier(random_state=42)

# Initialize the hyperparameters for each model
# RandomForest
param1 = {}
param1['classifier__n_estimators'] = [10, 25] # 50, 100]
#param1['classifier__max_features'] = ['sqrt'] # default is good.
#param1['classifier__criterion'] = ['gini'] # default is good
param1['classifier__max_depth'] = [3, 5, 7] # 5, 10] low to avoid overfitting
param1['classifier__min_samples_split'] = [20, 30] # high 
#param1['classifier__class_weight'] = Categorical([HashableDict({0:1, 1:5}), HashableDict({0:1, 1: 10}), HashableDict({0:1, 1: 25})]) # not needed as the dataset is balanced
param1['classifier'] = [clf1]

# SVC
param2 = {}
param2['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1]#, 10**2]
#param2['classifier__class_weight'] = Categorical([HashableDict({0:1, 1:5}), HashableDict({0:1, 1: 10}), HashableDict({0:1, 1: 25})])
param2['classifier'] = [clf2]

# DecisionTree
param4 = {}
param4['classifier__max_depth'] = [3,5, 7]
param4['classifier__min_samples_split'] = [20, 30]
#param4['classifier__class_weight'] = Categorical([HashableDict({0:1, 1:5}), HashableDict({0:1, 1: 10}), HashableDict({0:1, 1: 25})])
param4['classifier'] = [clf4]

# KNN
param5 = {}
param5['classifier__n_neighbors'] = [5, 10, 25]#[2,5,10,25]
param5['classifier'] = [clf5]

# GradientBoostingClassifier
param7 = {}
param7['classifier__n_estimators'] = [10, 50]#, 100]#, 250]
param7['classifier__max_depth'] = [5, 10]
param7['classifier__min_samples_split'] = [20, 30] # high
param7['classifier__learning_rate'] = [0.001, 0.01] # default was 0.1
param7['classifier'] = [clf7]

# XGB
param8 = {}
param8['classifier__n_estimators'] = [10, 50]#, 100]#, 250]
param8['classifier__colsample_bytree'] = [0.3, 0.5]# , 0.8 ]
param8['classifier__reg_alpha'] = [1, 5] # 0, 0.5, 
param8["classifier__reg_lambda"] = [1, 5] # 0, 0.5, 
param8['classifier__learning_rate'] = [0.001, 0.01] # default was 0.1
param8['classifier'] = Categorical([clf8])
param8['classifier__max_depth'] = [5, 10]

####
# Prepare cross-validation
####

# pipe for each model
pipeline = Pipeline([('classifier', clf8)]) 
params = [param8, param1, param4, param5, param7]

# train the grid search model
pipe = Pipeline([
    ('classifier', clf8)
])

# initialise bayesian search
custom_f1 = make_scorer(f1_score, greater_is_better=True, average="weighted")
cv = StratifiedKFold(n_splits=10, shuffle=True)
opt = BayesSearchCV(
    pipe,
    # (parameter space, # of evaluations)
    params,
    cv=cv,
    scoring = 'f1_macro',
    n_jobs = -1, # parallelise
    refit = True,
    optimizer_kwargs={
        "base_estimator": "GBRT" # gradient boost as sorrogate
    }
)

####
# Fit to train data
####

opt.fit(X_train, y_train)

# output results for all models
results_df = pd.DataFrame(opt.cv_results_)
results_df.to_csv(f'{experiment}_cvs_all.csv')

####
# Model comparison
####

# compare best models
best_cvs = {}
best_params = {}
best_params_table = {}
for i in results_df['param_classifier'].unique():
    m = results_df[results_df['param_classifier'] == i]
    st = str(i)
    st = st.split('(')[0]
    idx = m['mean_test_score'].idxmax()

    # best cross-validation scores
    best_cv = m.loc[idx, 'split0_test_score':'split9_test_score']
    best_param = m.loc[idx, 'params']

    # best hyperparameters
    best_cvs[st] = best_cv.values
    best_params[st] = best_param

    # save hyperparams in table
    best_param_table = m.loc[idx, 'param_classifier__colsample_bytree':'param_classifier__n_neighbors']
    best_params_table[st] = best_param_table

df_cvs = pd.DataFrame.from_dict(best_cvs, orient = 'columns')
df_cvs.to_csv(f'{experiment}_cvs_best.csv')

df_params = pd.DataFrame.from_dict(best_params_table, orient = 'columns')
df_params.to_csv(f'{experiment}_params_best.csv')

####
# Model evaluation
####

# evaluate best models
model_evals = {}

for i, (k, v) in enumerate(best_params.items()):

    # new model contains the hyperparams updated
    dic = dict(v)
    model = list(dic.items())[0][1]
    params = list(dic.items())[1:]
    dict_params = {}

    if i == 0:
        for x,y in params:
            dict_params[x] = y
        new_model = model.set_params(**dict_params)

    else:
        params2 = [(i[0].split('__')[1],i[1]) for i in params] 
        for x,y in params2:
            dict_params[x] = y
        new_model = model.set_params(**dict_params) 

    new_model.fit(X_train, y_train)
    grid_predict = new_model.predict(X_test)
    scores = score(y_test, grid_predict)
    acc = accuracy_score(y_test, grid_predict)
    scores = [i.mean() for i in scores]
    scores = scores + [acc]
    model_evals[k]=scores

df_eval = pd.DataFrame.from_dict(model_evals, orient='columns')
df_eval.index = ['precision', 'recall', 'f_score', 'support', 'accuracy']
df_eval.to_csv(f'{experiment}_predictions.csv')






















