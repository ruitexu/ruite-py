
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:12:45 2020

@author: rxu
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df_1 = pd.read_csv(r"L:\Analytics Team\1. Reporting & Dashboard\10. Monthly Acquisition Value reports\Sep\Good Cx factors\20200909_factor for new cx.csv")

df_1[[ 'return_Count',
 'total_return',
 'return_salesquantity',
 'return_regularpricequant',
 'return_GMAmount']] = df_1[[ 'return_Count',
 'total_return',
 'return_salesquantity',
 'return_regularpricequant',
 'return_GMAmount']].abs()

df_1['first_tran_GM%'] = df_1['first_tran_GMAmount']/df_1['first_tran_total_sales']

df_1['first_tran_regular%'] = df_1['first_tran_regularpriceamount']/df_1['first_tran_total_sales']

df_1['first_tran_regularquant%'] = df_1['first_tran_regularpricequant']/df_1['first_tran_total_salesquantity']

df_1 = df_1.dropna()

df_class = pd.read_csv(r"L:\Analytics Team\1. Reporting & Dashboard\10. Monthly Acquisition Value reports\Sep\Good Cx factors\new cx class by cx level.csv")

df = pd.merge(df_1, df_class,  how='left', left_on=['F_EmailAddress',
 'F_CountryCode2',
 'F_BannerName',
 'Acq_Channel'], right_on = ['F_EmailAddress',
 'F_CountryCode2',
 'Banner',
 'Channel'])
                             
df = df.dropna()

zero_col = (df != 0).any(axis=0)

null_col = df.isnull().sum()

# df.to_csv(r"L:\Analytics Team\1. Reporting & Dashboard\10. Monthly Acquisition Value reports\Sep\Good Cx factors\processed_factor for new cx.csv")

brand = 'Garage'

df_y = df[df[ 'F_BannerName'] == brand]['is_return_cx']

df_x = df[df[ 'F_BannerName'] == brand][[
 'is_Loyalty',
 # 'total_sales',
 # 'total_salesquantity',
 # 'regularpricequant',
 # 'GMAmount',
 # # 'return_Count',
 # 'total_return',
 # 'return_salesquantity',
 # 'return_regularpricequant',
 # 'return_GMAmount',
 'first_tran_total_sales',
 # 'first_tran_total_salesquantity',
 # 'first_tran_regularquant%',
 'first_tran_GM%',
 # 'first_tran_regular%',
  'contain_Activewear',
 'contain_Bags',
 'contain_Beauty',
 'contain_Belts',
 'contain_Closed Footwear',
 'contain_Cold Weather',
 'contain_Damages',
 'contain_Denim Blouses',
 'contain_Denim Capris',
 'contain_Denim Dresses',
 'contain_Denim Outerwear',
 'contain_Denim Pants',
 'contain_Denim Rompers',
 'contain_Denim Shorts',
 'contain_Denim Skirts',
 'contain_Denim Tops',
 'contain_Fleece Dresses',
 'contain_Fleece Overpiece',
 'contain_Fleece Pants',
 'contain_Fleece Shorts',
 'contain_Fleece Tops',
 'contain_Food',
 'contain_Franchisee Dummy Class',
 'contain_Giftware',
 'contain_Hair ',
 'contain_Housewares',
 'contain_Jewellery',
 'contain_Knit Blouses',
 'contain_Knit Capris',
 'contain_Knit Dresses',
 'contain_Knit Overpiece',
 'contain_Knit Pants',
 'contain_Knit Rompers',
 'contain_Knit Shorts',
 'contain_Knit Skirts',
 'contain_Knit Tops',
 'contain_Miscalleneous',
 'contain_Open Footwear',
 'contain_PU/Leather Outerwear',
 'contain_Scarves',
 'contain_Sleepwear',
 'contain_Socks',
 'contain_Sunglasses',
 'contain_Swimwear',
 'contain_Third Party',
 'contain_Tights',
 'contain_UDW',
 'contain_Unknown',
 'contain_Woven Blazers',
 'contain_Woven Blouses',
 'contain_Woven Dresses',
 'contain_Woven Outerwear',
 'contain_Woven Overpiece',
 'contain_Woven Pants',
 'contain_Woven Rompers',
 'contain_Woven shorts',
 'contain_Woven Skirts',
 'contain_Woven Tops',
 'contain_Yarn Dresses',
 'contain_Yarn Overpiece',
 'contain_Yarn Pants',
 'contain_Yarn Skirts',
 'contain_Yarn Tops'
 ]]


df_2= pd.get_dummies(df[['F_CountryCode2',
 # 'F_BannerName',
 'Acq_Channel']], drop_first=True)

df_x_final =  pd.concat([df_x, df_2.reindex(df_x.index)], axis=1).reindex(df_x.index)

# import matplotlib.pyplot as plt
# import seaborn as sns
# fig, ax = plt.subplots(figsize=(100,100))
# sns.heatmap(df_x_final.corr(), annot = True)
# plt.show()

#undersampling bad cx
from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy=1)
df_x_final_u, df_y_u = undersample.fit_resample(df_x_final, df_y)

#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
 df_x_final_u, df_y_u, test_size=0.33, random_state=42)

####first model feature selection, select 8-9 using RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, 9, step=15)
fit = rfe.fit(df_x_final, df_y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

df_result_rfe_d = pd.DataFrame({'coef':list(fit.support_), 'name':list(df_x_final)})
                           
####using RFE results to train first logistic regression model 
# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()
name_to_keep = list(df_result_rfe_d[df_result_rfe_d['coef'] == True]['name'])
name_to_keep.append(('first_tran_GM%'))

df_final_2 = x_train[name_to_keep]
logisticRegr.fit(df_final_2, y_train)
score = logisticRegr.score(x_test[name_to_keep], y_test)
score_2 = logisticRegr.score(df_final_2, y_train)

########first model accuracy 
from sklearn import metrics
predictions = logisticRegr.predict(x_test[name_to_keep])
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

# group_names = ['True Neg','False Pos','False Neg','True Pos']
# group_counts = ["{0:0.0f}".format(value) for value in
#                 cm.flatten()]
# group_percentages = ["{0:.2%}".format(value) for value in
#                      cm.flatten()/np.sum(cm)]
# labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
#           zip(group_names,group_counts,group_percentages)]
# labels = np.asarray(labels).reshape(2,2)
# sns.heatmap(cm, annot=labels, fmt="", cmap='Blues')

df_result = pd.DataFrame({'coef':list(logisticRegr.coef_[0]), 'name':list(df_final_2)})

####################################################

#second model random forest feature selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(max_depth=100, random_state=40)
model.fit(df_x_final, df_y)
a = pd.Series(model.feature_importances_, index=df_x_final.columns).nlargest(9)
b = a.to_frame()
b['name'] = b.index
b = b.reset_index()
(pd.Series(model.feature_importances_, index=df_x_final.columns)
   .nlargest(9)
   .plot(kind='barh')) 



clf = RandomForestClassifier(max_depth=100, random_state=40)
clf.fit(x_train[list(b['name'])], y_train)
random_prediction = clf.predict(x_test[list(b['name'])])
accuracy_score(y_test, random_prediction)
cm_rf = metrics.confusion_matrix(y_test, random_prediction)
print(cm_rf)



##########################################################
#combine all features test correlation and remove them
df_corr_test = df_x_final[list(set(b['name']).union(set(df_result_rfe_d[df_result_rfe_d['coef'] == True]['name'])))]
corr = df_corr_test.corr()

#test VIF after removing multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

#calculate VIF for each explanatory variable
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(df_corr_test.values, i) for i in range(df_corr_test.shape[1])]
vif['variable'] = df_corr_test.columns


##########final model using logistic regression model with all features
#logistic 2
logisticRegr_2 = LogisticRegression()
name_to_keep_2 = list(set(b['name']).union(set(df_result_rfe_d[df_result_rfe_d['coef'] == True]['name'])))

df_final_3 = x_train[name_to_keep_2]
logisticRegr_2.fit(df_final_3, y_train)
score_3 = logisticRegr_2.score(x_test[name_to_keep_2], y_test)
score_4 = logisticRegr_2.score(df_final_3, y_train)


#draw confusion matrix
from sklearn import metrics
predictions_2 = logisticRegr_2.predict(x_test[name_to_keep_2])
cm_2 = metrics.confusion_matrix(y_test, predictions_2)
print(cm_2)

group_names_2 = ['True Neg','False Pos','False Neg','True Pos']
group_counts_2 = ["{0:0.0f}".format(value) for value in
                cm_2.flatten()]
group_percentages_2 = ["{0:.2%}".format(value) for value in
                     cm_2.flatten()/np.sum(cm_2)]
labels_2 = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names_2,group_counts_2,group_percentages_2)]
labels_2 = np.asarray(labels_2).reshape(2,2)
sns.heatmap(cm_2, annot=labels_2, fmt="", cmap='Blues')


df_result_2 = pd.DataFrame({'coef':list(logisticRegr_2.coef_[0]), 'name':list(df_final_3)})


predictions_test = logisticRegr_2.predict_proba(x_test[name_to_keep_2])






