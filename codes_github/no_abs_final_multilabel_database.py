# Import necessary packages
import pandas as pd
import shap, os

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import *
from sklearn.inspection import permutation_importance

from collections import defaultdict

from preprocess import preprocess
from helper_features import *

from sklearn.base import clone

directory = '/home/damia001/features/MULTILABEL'

# initialize hyperparameters from optimization
params = {}
params['classifier__n_estimators'] = [45]
params['classifier__colsample_bytree'] = [0.472967089]
params['classifier__reg_alpha'] = [1] 
params['classifier__reg_lambda'] = [1] 
params['classifier__learning_rate'] = [0.009519078] 
params['classifier__max_depth'] = [10]
params['classifier__verbosity'] = [0]

# initialize model
classifier = XGBClassifier(random_state=42).set_params(**params)
clf = clone(classifier)

mf = pd.read_csv('qia_processed.csv', index_col = 0)
mf.drop('luminance', axis = 1, inplace=True)
feature_names = mf.columns[:38]

group = 'Database_only'

# create folder with group name
if not os.path.exists(os.path.join(directory, group)):
     os.mkdir(os.path.join(directory, group))

if not os.path.exists(os.path.join(directory,group,'interpretation')):
    os.mkdir(os.path.join(directory,group,'interpretation'))

if not os.path.exists(os.path.join(directory,group,'shap')):
    os.mkdir(os.path.join(directory,group,'shap'))

group_metrics = {}
exp = mf.copy()

df_filtered, mf_processed, X_train, X_test, y_train,  y_test, y_labels, dict_target_inv, no_classes = preprocess(
exp, 
rescale = 'standard', 
outlier = 'keep', 
imbalance = 'oversample_train', 
filter_d = False)

# dataframe for viz
head_df = mf_processed.groupby('Main type').head(20)
mf_viz = pd.DataFrame(head_df.loc[:,'convexity':'value_mean'], index=head_df.index, columns=head_df.loc[:,'convexity':'value_mean'].columns)
mf_viz['Main type'] = head_df['Main type']

# data exploration
colors_dic = {'Altered material':'#F7931E','Free-crystal':'#29ABE2','Juvenile':'#B81A5D','Lithic':'#006837'}
colors_final = {}

for k,v in dict_target_inv.items():
    color = colors_dic[v]
    colors_final[v] = color

# Simple histogram plot
plot_multiple_histograms(mf_viz, mf_viz.loc[:,'convexity':'value_mean'])
plt.savefig(f'{directory}/{group}/all_histograms.svg')
plt.close()

# # corrmat
corrmat = mf_viz.loc[:,'convexity':'value_mean'].corr()
fig, ax = plt.subplots(figsize=(30,30))
sns.heatmap(corrmat, annot = True, square = True, cmap = 'Greens', annot_kws={"size": 8})
plt.savefig(f'{directory}/{group}/corrmat.svg')
plt.close()

# fit
clf.set_params(num_class=no_classes, verbosity = 0)
clf.fit(X_train, y_train)

# permutation
fig, ax = plt.subplots(figsize=(5,9))

perm_importance = permutation_importance(clf, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.savefig(f'{directory}/{group}/permutation_importance_features.svg')
plt.close()

# remove those with permutation importance lower than threshold
threshold = 0.0
permutation_importances_series = pd.Series(perm_importance.importances_mean[sorted_idx], index = feature_names[sorted_idx])
#permutation_importances_series.to_csv(f'{directory}/{group}/permutation_importances_series.csv')
imp = permutation_importances_series[permutation_importances_series<threshold]
keep = permutation_importances_series[permutation_importances_series>=threshold]

# remove low-ranked features and process data again
perm_exp = mf[list(keep.index)[::-1]]

# copy categorical columns
for i in ['Main type', 'Volcano', 'Eruptive style']:
    perm_exp[i] = mf[i]


df_filtered, mf_processed, X_train, X_test, y_train,  y_test, y_labels, dict_target_inv, no_classes = preprocess(
perm_exp, 
    rescale = 'standard', 
    outlier = 'keep', 
    imbalance = 'oversample_train', 
    filter_d = False,
    v_start = keep.index[-1],
    v_end = keep.index[0])


# refit
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)
print(y_prob.shape)
#if no_classes == 2: y_pred = [np.where(i == float(1))[0][0] for i in y_pred]
#print(classification_report(y_test, y_pred, target_names=dict_target_inv.values(), digits = 2))

# metrics and plots
scores = score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
scores = [i.mean() for i in scores]
scores = scores + [acc]
group_metrics[f'{group} (n={no_classes})'] = scores

# confusion matrix
plot_cm(y_test, y_pred, dict_target_inv.values())
plt.savefig(f'{directory}/{group}/cm.svg')
plt.close()

# bar plot from confusion matrix
plot_bar_cm(y_labels, y_pred, dict_target_inv, main_type = True)
plt.savefig(f'{directory}/{group}/bar_cm.svg')
plt.close()

df_classification = pd.DataFrame.from_dict(y_prob, orient='columns')
df_classification.columns = list(dict_target_inv.values())
df_classification.index = y_test.index

# copy OVR classification results
df_clf = df_classification.copy()

# create column with true labels
df_clf['Main type'] = y_labels

# create column with particle type with highest probability
df_clf['Max'] = df_clf.iloc[:,:no_classes].idxmax(axis=1)

# create a factorized column
df_clf['y_pred'] = df_clf.Max
dict_target = dict(zip(dict_target_inv.values(), dict_target_inv.keys()))
df_clf.replace({'y_pred': dict_target}, inplace=True)

# create column with true values
df_clf['y_test'] = y_test.values


#
# Getting the second highest probability
#

# define function to extract 2nd highest prob
def get_second_colname(row):
    s = row.astype(float)
    second_largest = s.nlargest(2).idxmin()
    return second_largest

# create column with second highest probability
df_clf['Max2'] = df_clf.iloc[:,:no_classes].apply(get_second_colname, axis=1)

# get predictions and probs from overall
y_pred = df_clf['y_pred'].values
y_pred_labels = np.vectorize(dict_target_inv.get)(y_pred)
y_test = df_clf['y_test']

# save classification report
report = classification_report(y_test, y_pred,  target_names=list(dict_target.keys()), output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(f'{directory}/{group}/classification_score.csv')

# save probabilities
df_clf.to_csv(f'{directory}/{group}/probabilities_score.csv')

# empty dics that allows to append
f1_di = defaultdict(list)
support_di = defaultdict(list)

# prob threshold iterations
for i in range(310,1000,1):
    i = i/1000 # <- max is i=1

    df_clf2 = df_clf.copy()

    # select rows where any prediction has a prob higher than threshold
    df_clf2 = df_clf2[(df_clf2.iloc[:,:no_classes]>i).any(axis=1)]
    
    # extract columns
    y_pred2 = df_clf2['y_pred'].values
    y_test2 = df_clf2['y_test']

    # create report to track f1-score and support (ie, number of particles)
    preds = list(np.unique(y_pred2))
    tests = list(y_test2.unique())
    l = preds + tests

    filt_dict = {k:v for (k,v) in dict_target_inv.items() if k in l}
    report = classification_report(y_test2, y_pred2,  target_names=list(filt_dict.values()), output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop(['accuracy'])

    # append tracked params to dicts
    for name, row in df_report.iloc[:-1,:].iterrows():
        if name == 'macro avg':
            name = 'Overall'
        f1_di[name].append(row['f1-score'])
        support_di[name].append(row['support'])

colors_dic = {'Altered material':'#F7931E','Free-crystal':'#29ABE2','Juvenile':'#B81A5D','Lithic':'#006837'}
colors_dic = {k:v for (k,v) in colors_dic.items() if k in list(dict_target_inv.values())}
colors_dic['Overall'] = '#000000'

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (6, 8), sharex=True)

for k, v in f1_di.items():
    xs = list(range(310,1000,1))
    xs = [i/10 for i in xs]
    xs = xs[:len(v)]
    ax[0].plot(xs, v, color=colors_dic[k], label = k)
    ax[0].set_ylabel('F1-score (macro)')
    ax[0].spines['bottom'].set_visible(False)
ax[0].grid()
ax[0].legend(ncol = 3)

for k, v in support_di.items():
    xs2 = xs[:len(v)]
    ax[1].plot(xs2, v, color=colors_dic[k], label = k)
    ax[1].set_ylabel('Number of particles (test set)')
    ax[1].set_xlabel('Probability threshold')

#ax[1].legend()
ax[1].grid()
plt.tight_layout(h_pad = -0.3)
plt.savefig(f'{directory}/{group}/f1_score_thr.svg', dpi=300)
plt.close()

#for name, col in df_clf.iloc[:,:4].iteritems():
fig, ax = plt.subplots(nrows = no_classes, ncols = 2, figsize = (10,15))
for ii, (name, gr) in enumerate(df_clf.groupby('Max')):
    ax[ii,0].set_title(f'{name} is most voted')
    for i in range(len(gr)):
        ax[ii,0].scatter(i, gr.loc[gr.index[i], name], color='black' if gr.loc[gr.index[i],'y_pred'] == gr.loc[gr.index[i],'y_test'] else 'red', s = 3, marker = 'o')

for ii, (name, gr) in enumerate(df_clf.groupby('Max2')):
    ax[ii,1].set_title(f'{name} is second most voted')
    for i in range(len(gr)):
        ax[ii,1].scatter(i,
            gr.loc[gr.index[i],
            name], 
            color='black' if gr.loc[gr.index[i],'y_pred'] == gr.loc[gr.index[i],'y_test'] else 'red',
            s = 3,
            marker = 'o')
plt.savefig(f'{directory}/{group}/probs_true_or_false.svg')
plt.close()

# for i in range(len(y_test)):
#     fig, ax1 = plt.subplots()
#     im_name = y_test.index[i]
#     volcano = im_name[:2]
#     particle_type = im_name.split('_')[-1]
#     ylabel = f'{volcano}\nType:{particle_type}'
#     # if no_classes==2:
#     #     g = shap.force_plot(explainer.expected_value, shap_values[i,:].round(decimals=2), X_test.iloc[i,:].round(decimals=2), show=False, matplotlib=True)
#     # else:
#     #     g = shap.force_plot(explainer.expected_value[0], shap_values[0][i,:].round(decimals=2), X_test.iloc[i,:].round(decimals=2), show=False, matplotlib=True)
#     # g.set_figwidth(12)
    
#     # ax1 = g.add_axes([0.2, -0.7, 0.5, 1])
#     im = Image.open(f'/home/damia001/relabeled/{im_name}.png')
#     ax1.imshow(im)
#     ax1.set_axis_off()
#     ax1.set_title(ylabel,color='black' if y_pred[i] == y_test[i] else 'red', size=15)

#     ax2 = ax1.inset_axes([0,-0.1,1,0.1])
#     ax2.set(xticks=[], yticks=[])
#     ax2.axis('off')

#     pred = pd.DataFrame(y_prob[i])
#     pred = pred.T
#     pred.columns = dict_target_inv.values()
#     h = pred.plot(kind='barh',stacked=True,rot=90, ax = ax2,color=colors_final, legend=False)
#     plt.savefig(f'{directory}/{group}/interpretation/{im_name}_im_grid.svg')
#     plt.tight_layout()
#     plt.close()

# SHAP
explainer = shap.TreeExplainer(clf, X_test, model_output = "raw", feature_pertubation = "interventional", output_names = list(dict_target_inv.values()))

# Plot local explanations with SHAP values
# for i, pred_type in enumerate(y_pred):
#     pred_type_name = dict_target_inv[pred_type]
#     shap_local = explainer(X_test.iloc[i,:])
#     row = pd.Series(shap_local.values[:,pred_type], index = X_test.columns)
#     top_cols = abs(row).nlargest(8).index
#     top_means = [np.mean(mf[i]) for i in top_cols]
#     top_means_class = [np.mean(mf[mf['Main type'] == dict_target_inv[pred_type]][i]) for i in top_cols]
#     top = row.filter(items = top_cols)
#     fig, ax, index1 = waterfall_plot(top.index,
#     top.values,sorted_value=True, 
#     net_label="output\nvalue", 
#     blue_color = "#0492c2", 
#     green_color = "#B2D3C2", 
#     red_color = '#bc544b', 
#     figsize = (6,3.5))

#     particle = y_test.index[i]
#     row_vals = mf.loc[particle,:]    
#     top_row_vals = row_vals.filter(items = index1)  

#     ax2 = ax.inset_axes([0,1,1,0.1])
#     ax2.set(xticks=[], yticks=[])
#     ax2.axis('off')

#     for i, bar in enumerate(ax.patches[:8]): 
#         ax2.annotate(f'{round(top_row_vals.values[i],2)}\n{round(top_means[i],1)}\n{round(top_means_class[i],1)}', (i*1.06/10+0.04,0.9), color = '#36454F')# + bar.get_width() / 20,1))        
    
#     ax2.set_title('Value:\nMean:\nMean_class:', x=0., y=-5, ha='right', size = 10, color = '#36454F')
#     plt.title(f'Predicted type: {pred_type_name}')
#     plt.rcParams['axes.titlepad'] = +60
#     fig.tight_layout()
#     fig.savefig(f'{directory}/{group}/shap/{particle}_shapley.svg')
#     plt.close()

shap_values = explainer.shap_values(X_test)
shap_df = pd.DataFrame(columns=X_train.columns.tolist())
shap_df['true'] = np.nan
shap_df['pred'] = np.nan

# compute shapley values for each instance according to y_pred
for i, part_type in enumerate(y_pred):
    
    shap_type = shap_values[part_type][i]
    shap_type = np.append(shap_type,int(y_test[i]))
    shap_type = np.append(shap_type,int(part_type))
    shap_s = pd.Series(shap_type)
    shap_df.loc[i] = shap_type

shap_df.to_csv(f'{directory}/{group}/shap/shapley_all.csv')

#### Not Absolute Shapley #### changed 23/08/2023
not_abs_final_importances_by_class = {k:[] for k in list(dict_target_inv.values())}

for i,part_type in dict_target_inv.items():
    shap_df = pd.DataFrame(shap_values[i])
    shap_df.columns = X_test.columns

    not_absolute_shap_mean = shap_df.mean(axis=0).values # removed .abs()
    shap_mean = shap_df.mean(axis=0).values

    not_abs_final_importances_by_class[part_type] = not_absolute_shap_mean

shap_abs = pd.DataFrame(not_abs_final_importances_by_class)
shap_abs.index = X_test.columns
  
shap_abs['Total'] = shap_abs.sum(axis = 1)
shap_abs = shap_abs.sort_index(key=shap_abs.abs().sum(1).get)
#shap_abs = shap_abs.sort_index(key=shap_abs.sum(1).get)#.head(20) # removed .abs()
colors_dic = {'Altered material':'#F7931E','Free-crystal':'#29ABE2','Juvenile':'#B81A5D','Lithic':'#006837'}
fig, ax = plt.subplots(figsize=(5,5))
shap_abs.iloc[-20:,:4].plot.barh(stacked=True, color = colors_dic, ax = ax)
plt.tight_layout()
plt.savefig(f'{directory}/{group}/shap/global_not_abs_shapley_contribution.svg')
plt.close()
shap_abs.to_csv(f'{directory}/{group}/shap/global_not_abs_shapley_contribution.csv')


#### Exact Shapley ####
# shap_all = pd.DataFrame()
# for k,v in dict_target_inv.items():

#     # use function to extract mean Shapley values
#     abs_shap = get_ABS_SHAP(shap_values[k], X_test)
#     abs_shap['class'] = v
#     shap_all = pd.concat([shap_all,abs_shap])

# for i, feature in enumerate(shap_abs.index):
#     gr = shap_all[shap_all['Variable'] == feature]
#     g = sns.barplot(x = 'SHAP', y = 'Variable', hue = 'class', data = gr.iloc[::-1], palette=colors_dic, capsize=.4)
#     g.legend(title = 'Predicted class')
#     plt.ylabel('')
#     plt.yticks([1], [''])
#     plt.title(feature)
#     plt.xlabel('Shapley value')
#     plt.savefig(f'{directory}/{group}/shap/{feature}_shapley_contribution.svg')
#     plt.close()

# pivot_df = shap_all.pivot(index='Variable', columns='class')['SHAP']
# pivot_df['Total'] = pivot_df.abs().sum(axis=1)
# pivot_df = pivot_df.sort_values(by='Total', ascending = False)
# pivot_df.to_csv(f'{directory}/{group}/shap/shapley_contribution.csv') # the total should be the same as in ABS


# # load JS for plots
# shap.initjs()

# # get colors
# colors_dic = {'Altered material':'#F7931E','Free-crystal':'#29ABE2','Juvenile':'#B81A5D','Lithic':'#006837'}
# colors_final = {}

# for k,v in dict_target_inv.items():
#     color = colors_dic[v]
#     colors_final[v] = color

# for i in list(dict_target_inv.keys()):
#     # dependence plot
#     for variable in keep.index[::-1]:
#         shap_df = pd.DataFrame(shap_values[i], columns=X_train.columns.tolist())
#         shap_var = pd.DataFrame(index = X_test.index)
#         shap_var[f'{variable}_test'] = X_test[variable]
#         shap_var[f'{variable}_shap'] = shap_df[variable]
#         shap_var['Main type'] = list(y_pred_labels)
#         #shap_var = shap_var.groupby(shap_var['Main type']).head(100)

#         fig, ax = plt.subplots(figsize=(7,7))
#         for name, p_type in shap_var.groupby('Main type'):
#             plt.scatter(p_type[f'{variable}_test'], p_type[f'{variable}_shap'], color=colors_final[name], label = name, alpha = 1, edgecolors='black', s = 50)
#         plt.xlabel(f'{variable}')
#         plt.ylabel(f'Shapley value for\n{variable}')
#         plt.legend()
#         plt.savefig(f'{directory}/{group}/shap/{dict_target_inv[i]}_{variable}_shap_perm_predicted.svg', pad_inches=0.0, bbox_inches="tight")
#         plt.close()

# from matplotlib.colors import ListedColormap
# colors = list(colors_dic.values())
# names = list(colors_dic.keys())
# my_cmap = ListedColormap(colors)
# shap.summary_plot(list(shap_values), X_test, feature_names = X_test.columns, class_names=names,color=my_cmap, plot_type='bar', show = False)
# plt.savefig(f'{directory}/{group}/shap/global_shap.svg')
# plt.close()

# if no_classes > 2:

#     foo_all = pd.DataFrame()

#     for k,v in list(enumerate(range(no_classes))):

#         foo = get_ABS_SHAP(shap_values[k], X_test) # <- function in helper_features.py
#         foo['class'] = v
#         foo_all = pd.concat([foo_all,foo])

#     foo_all['class'] = foo_all['class'].map(dict_target_inv)
#     pivot_df = foo_all.pivot(index='Variable', columns='class')['SHAP']

#     colors_dic = {'Altered material':'#F7931E','Free-crystal':'#29ABE2','Juvenile':'#B81A5D','Lithic':'#006837'}
#     pivot_df2 = pivot_df.sort_index(key=pivot_df.abs().sum(1).get)
#     pivot_df2 = pivot_df2.iloc[::-1]
#     pivot_df2.to_csv(f'{directory}/{group}/shap/mean_shapley_values.csv')
#     fig, ax = plt.subplots(figsize = (5,30))
#     sns.barplot(x = 'SHAP', y = 'Variable', hue = 'class', data = foo_all.iloc[::-1], ax = ax, palette=colors_dic)
#     fig.savefig(f'{directory}/{group}/shap/shap_tornado.svg')
#     plt.close()

# else:

#     # get global variable importance plot
#     s = ''
#     for k, v in dict_target_inv.items():
#         s += str(k) + '_'
#         s += str(v) + '_'

#     plt_shap = shap.summary_plot(shap_values, #Use Shap values array
#                                  features=X_test, # Use training set features
#                                  feature_names=X_test.columns, #Use column names
#                                  show=False, #Set to false to output to folder
#                                  plot_size=(20,10),# Change plot size
#                                  class_names=dict_target_inv.values()) 

#     # Save my figure to a directory
#     plt.savefig(f'{directory}/{group}/shap/{s}global_shap.svg', dpi=300)

# # # global explainability summary plot
# # if no_classes==2:
# #     shap.summary_plot(shap_values, X_test, feature_names=X_train.columns, show=False)
# # else:
# #     shap.summary_plot(shap_values[0], X_test, feature_names=X_train.columns, show=False)
# # f = plt.gcf()
# # f.savefig(f'{directory}/{group}/shap/shap_impact_variable.svg')
# # plt.close()

# # # impact per features by class on model
# # shap.summary_plot(shap_values, X_test, class_names=list(dict_target_inv.values()), show=False)
# # f = plt.gcf()
# # f.savefig(f'{directory}/{group}/shap/shap_impact_per_class.svg')
# # plt.close()

# df_groups = pd.DataFrame.from_dict(group_metrics, orient='columns')
# df_groups.index = ['precision', 'recall', 'f_score', 'support', 'accuracy']
# df_groups = df_groups.reindex(['accuracy', 'precision', 'recall', 'f_score', 'support'])
# df_groups.to_csv(f'{directory}/{group}.csv')






















