# Importar as Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dataframe_image as dfi
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour  
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve, roc_auc_score

import warnings
warnings.filterwarnings('ignore')

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# configurações
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# Parametros Gráfico
rc_params = {'axes.spines.top': False,
             'axes.spines.right': False,
             'legend.fontsize': 8,
             'legend.title_fontsize': 8,
             'legend.loc': 'upper right',
             'legend.fancybox': False,
             'axes.titleweight': 'bold',
             'axes.titlesize': 12,
             'axes.titlepad': 12}
sns.set_theme(style='ticks', rc=rc_params)
sns.set_color_codes('muted')


#Base de Dados
df = pd.read_csv('ManutencaoPreditiva/desafio_manutencao_preditiva_treino.csv', sep=',')
print(df)


df = df.drop(['udi', 'product_id'], axis=1)

# Criando coluna Target
df['target'] = df['failure_type'].replace(['Heat Dissipation Failure', 'Power Failure', 'Overstrain Failure', 'Tool Wear Failure', 'Random Failures'], 1).replace('No Failure', 0)

# Verificação do dataset
check_df = pd.DataFrame({
                    'type': df.dtypes,
                    'missing': df.isna().sum(),
                    'size': df.shape[0],
                    'unique': df.nunique()})
check_df['percentual'] = round(check_df['missing'] / check_df['size'], 2)
print(check_df)
dfi.export(check_df, 'check_df.png')

check = df['failure_type'].value_counts()
print(check)

# EDA
# Número de Tipos de Falhas
ax1 = plt.subplots(figsize=(10, 4))
ax1 = sns.countplot(x='failure_type', color = sns.xkcd_rgb['windows blue'], data= df, 
                                order = ['Heat Dissipation Failure', 'Power Failure', 'Overstrain Failure', 'Tool Wear Failure', 'Random Failures'])
for i in ax1.patches:
    h = i.get_height()
    ax1.annotate('{:.0f}'.format(h),
                  (i.get_x() + i.get_width()/2, h),
                  ha='center',
                  va='baseline',
                  fontsize=10,
                  color='black',
                  xytext=(0, 2),
                  textcoords='offset points')

plt.title('Número de Tipos de Falhas', fontsize=12, loc='left', pad=10)
plt.tight_layout()
plt.show()


# Tipo de Produto
ax2 = plt.subplots(figsize=(8, 6))
ax2 = sns.countplot(x ='type', data = df, order = ['L', 'M', 'H'], color = sns.xkcd_rgb['windows blue'])
plt.title('Tipo de Produto', fontsize=14, loc='left', pad=10)

for i in ax2.patches:
    h = i.get_height()
    ax2.annotate('{:.0f}'.format(h),
                  (i.get_x() + i.get_width()/2, h),
                  ha='center',
                  va='baseline',
                  fontsize=10,
                  color='black',
                  xytext=(0, 2),
                  textcoords='offset points')
plt.tight_layout()
plt.show()


# Gráfico Tipo de Falha
ax3 = plt.subplots(figsize=(8, 5))
ax3 = sns.countplot(x=df.target, color = sns.xkcd_rgb['windows blue'])
plt.title('Tipo de Falha', fontsize=14, loc='left', pad=10)
for i in ax3.patches:
    h = i.get_height()
    ax3.annotate('{:.0f}'.format(h),
                  (i.get_x() + i.get_width()/2, h),
                  ha='center',
                  va='baseline',
                  fontsize=10,
                  color='black',
                  xytext=(0, 2),
                  textcoords='offset points')
plt.tight_layout()
plt.show()


# Representação em porcentagem no dataset
print('{:.2f}% de Failure'.format((df[df.target == 1].shape[0] / df.shape[0]) * 100))
print('{:.2f}% de No Failure'.format((df[df.target == 0].shape[0] / df.shape[0]) * 100))


# Estatistica Descritiva
describe = df.describe()
print(describe)
#dfi.export(describe, 'describe.png')

# Skewness Analysis
df_skew =  df.loc[:,['air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 'torque_nm', 'tool_wear_min']]
skew = df_skew.skew()
print(skew)
#dfi.export(skew, 'skew.png')

# Histograma das variáveis
fig, ax4 = plt.subplots(nrows=2, ncols=5, figsize=(20,8), constrained_layout=True)
sns.histplot(data = df, x = 'rotational_speed_rpm', kde=True, ax = ax4[0,0])
sns.histplot(data = df, x = 'torque_nm', kde=True, ax = ax4[0,1])
sns.histplot(data = df, x = 'tool_wear_min', kde=True, ax = ax4[0,2])
sns.histplot(data = df, x = 'air_temperature_k', kde=True, ax = ax4[0,3])
sns.histplot(data = df, x = 'process_temperature_k', kde=True, ax = ax4[0,4])
sns.boxplot(data = df, x = 'rotational_speed_rpm', ax = ax4[1,0], color = sns.xkcd_rgb['windows blue'])
sns.boxplot(data = df, x = 'torque_nm', ax = ax4[1,1], color = sns.xkcd_rgb['windows blue'])
sns.boxplot(data = df, x = 'tool_wear_min', ax = ax4[1,2], color = sns.xkcd_rgb['windows blue'])
sns.boxplot(data = df, x = 'air_temperature_k', ax = ax4[1,3], color = sns.xkcd_rgb['windows blue'])
sns.boxplot(data = df, x = 'process_temperature_k', ax = ax4[1,4], color = sns.xkcd_rgb['windows blue'])
plt.tight_layout()
plt.show()


# Gráfico de Correlação
plt.figure(figsize = (10, 8))
sns.heatmap(df.corr(), cmap='Blues', annot=True, fmt="1.1f")
plt.title('Correlação', fontsize=14, loc='left', pad=10)
plt.tight_layout()
plt.show()


# Relação entre variáveis
plt.figure(figsize=(12,5))
sns.scatterplot(x='air_temperature_k', y='process_temperature_k', hue='target', alpha=0.75, data=df)
plt.xlabel('Air temperature [K]')
plt.ylabel('Process temperature [K]')
plt.title('Air temperature [K] x Process temperature [K]', fontsize=14, loc='left', pad=10)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
sns.scatterplot(x='rotational_speed_rpm', y='torque_nm', hue='target', alpha=0.75, data=df)
plt.xlabel('Rotational Speed [rpm]')
plt.ylabel('Torque [Nm]')
plt.title('Rotational Speed [rpm] x Torque [Nm]', fontsize=14, loc='left', pad=10)
plt.tight_layout()
plt.show()


# Preparação dos Dados
# Separar dados entre Treino e Teste
df1 = df.copy()

label_encoder = LabelEncoder()
label_encoder.fit(df1['type'])
df1['type'] = label_encoder.transform(df1['type'])

df1.info()

X = df1.drop(['target', 'failure_type'], axis=1)
y = df1['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# Algoritmos
rf = RandomForestClassifier()
svc = SVC()
lr = LogisticRegression()
knn = KNeighborsClassifier()
xgb = XGBClassifier()

# Padronização dos dados
Rob_scaler = RobustScaler()
X_train_Rscaled = Rob_scaler.fit_transform(X_train)


# Definindo funçao de validação com dados balanceados
def val_model_balanced(X, y, model, quite=False):
  X = np.array(X)
  y = np.array(y)

  scores = cross_val_score(model, X, y, scoring='recall')

  if quite == False:
    print('Recall: {:.4f} (+/- {:.4f})'.format(scores.mean(), scores.std() * 2))
  return scores.mean()

# Balanceamento dos dados - Oversample
# Smote
smote = SMOTE(random_state=42)
X_smt_Rscaled, y_train_smt_Rscaled = smote.fit_resample(X_train_Rscaled, y_train)

# Checando o balanceamento das classes
sns.countplot(x=y_train_smt_Rscaled, color = sns.xkcd_rgb['windows blue'])
plt.title('Balanceamento SMOTE', fontsize=12, loc='left', pad=10)
plt.tight_layout()
plt.show()

# ADASYN
ada = ADASYN()
X_ada_Rscaled, y_train_ada_Rscaled = ada.fit_resample(X_train_Rscaled, y_train)

'''
# Balanceamento SMOTE e RobustScaler
print('\nCross-Validation com Balanceamento SMOTE e RobustScaler')
print('RF:')
score_teste1 = val_model_balanced(X_smt_Rscaled, y_train_smt_Rscaled, rf)
print('\nSVC:')
score_teste2 = val_model_balanced(X_smt_Rscaled, y_train_smt_Rscaled, svc)
print('\nLR:')
score_teste3 = val_model_balanced(X_smt_Rscaled, y_train_smt_Rscaled, lr)
print('\nKNN:')
score_teste4 = val_model_balanced(X_smt_Rscaled, y_train_smt_Rscaled, knn)
print('\nXGB:')
score_teste5 = val_model_balanced(X_smt_Rscaled, y_train_smt_Rscaled, xgb)
'''

# Slavar os resultados
model = []
recall = []

model.append('Random Forest Classifier')
recall.append(val_model_balanced(X_smt_Rscaled, y_train_smt_Rscaled, rf, quite=True))
model.append('SVC')
recall.append(val_model_balanced(X_smt_Rscaled, y_train_smt_Rscaled, svc, quite=True))
model.append('Logistic Regression')
recall.append(val_model_balanced(X_smt_Rscaled, y_train_smt_Rscaled, lr, quite=True))
model.append('KNeighbors Classifier')
recall.append(val_model_balanced(X_smt_Rscaled, y_train_smt_Rscaled, knn, quite=True))
model.append('XGBClassifier')
recall.append(val_model_balanced(X_smt_Rscaled, y_train_smt_Rscaled, xgb, quite=True))

recall_model = pd.DataFrame(data=recall, index=model, columns=['Recall'])
print(recall_model)
#dfi.export(recall_model, 'recall_model.png')

'''
# Balanceamento ADASYN e RobustScaler
print('\nCross-Validation com Balanceamento ADASYN e RobustScaler')
print('RF:')
score_teste1 = val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, rf)
print('\nSVC:')
score_teste2 = val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, svc)
print('\nLR:')
score_teste3 = val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, lr)
print('\nKNN:')
score_teste4 = val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, knn)
print('\nXGB:')
score_teste5 = val_model_balanced(X_ada_Rscaled, y_train_ada_Rscaled, xgb)

# Balanceamento dos dados - Undersample
# Balanceamento RUS
rus = RandomUnderSampler()
X_rus_Rscaled, y_train_rus_Rscaled = rus.fit_resample(X_train_Rscaled, y_train)

# Condensed nearest neighbour
cnn = CondensedNearestNeighbour()
X_cnn_Rscaled, y_train_cnn_Rscaled = cnn.fit_resample(X_train_Rscaled, y_train)

# Balanceamento RUS e RobustScaler
print('\nCross-Validation com Balanceamento RUS e RobustScaler')
print('RF:')
score_teste1 = val_model_balanced(X_rus_Rscaled, y_train_rus_Rscaled, rf)
print('\nSVC:')
score_teste2 = val_model_balanced(X_rus_Rscaled, y_train_rus_Rscaled, svc)
print('\nLR:')
score_teste3 = val_model_balanced(X_rus_Rscaled, y_train_rus_Rscaled, lr)
print('\nKNN:')
score_teste4 = val_model_balanced(X_rus_Rscaled, y_train_rus_Rscaled, knn)
print('\nXGB:')
score_teste5 = val_model_balanced(X_rus_Rscaled, y_train_rus_Rscaled, xgb)

# Balanceamento CondensedNearestNeighbour e RobustScaler
print('\nCross-Validation com Balanceamento CondensedNearestNeighbour e RobustScaler')
print('RF:')
score_teste1 = val_model_balanced(X_cnn_Rscaled, y_train_cnn_Rscaled, rf)
print('\nSVC:')
score_teste2 = val_model_balanced(X_cnn_Rscaled, y_train_cnn_Rscaled, svc)
print('\nLR:')
score_teste3 = val_model_balanced(X_cnn_Rscaled, y_train_cnn_Rscaled, lr)
print('\nKNN:')
score_teste4 = val_model_balanced(X_cnn_Rscaled, y_train_cnn_Rscaled, knn)
print('\nXGB:')
score_teste5 = val_model_balanced(X_cnn_Rscaled, y_train_cnn_Rscaled, xgb)
'''

'''
# Hiperparâmetros
#RandomForestClassifier
param_grid = {'max_depth': [5, 10, 20, 30, 40, 50, 100],
              'n_estimators': [10, 50, 100, 300, 1000],
              'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': [2, 6, 10],
              'min_samples_leaf': [1, 3, 4]}

kfold = StratifiedKFold(n_splits=10, shuffle=True)
grid_search = GridSearchCV(rf, param_grid=param_grid, scoring='recall', n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_smt_Rscaled, y_train_smt_Rscaled)
print('RandomForestClassifier - ', f'Melhor: {grid_result.best_score_} para {grid_result.best_params_}')

#SVC
kfold = StratifiedKFold(n_splits=10, shuffle=True)
clf_svc = RandomizedSearchCV(svm.SVC(gamma='auto'), {
          'C': [100, 10, 20, 1.0, 0.1, 0.001],
          'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}, n_jobs=-1, scoring='recall', n_iter=15, cv=kfold)

clf_svc.fit(X_smt_Rscaled, y_train_smt_Rscaled)
print('SVC - ', clf_svc.best_params_)

# LogisticRegression
param_grid = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
              'C': np.logspace(-4, 4, 20),
              'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
              'max_iter': [100, 1000, 2500, 5000]}

grid_search = GridSearchCV(lr, param_grid=param_grid, cv=5, return_train_score=False, scoring='recall')
grid_result = grid_search.fit(X_smt_Rscaled, y_train_smt_Rscaled)
print('LogisticRegression - ', f'Melhor: {grid_result.best_score_} para {grid_result.best_params_}')

# KNeighborsClassifier
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
              'metric': ['euclidean', 'manhattan', 'minkowski'],
              'weights': ['uniform', 'distance']}

kfold = StratifiedKFold(n_splits=10, shuffle=True)
grid_search = GridSearchCV(knn, param_grid=param_grid, cv=kfold, n_jobs=-1, scoring='recall')
grid_result = grid_search.fit(X_smt_Rscaled, y_train_smt_Rscaled)
print('KNeighborsClassifierf - ', f'Melhor: {grid_result.best_score_} para {grid_result.best_params_}')


# XGBClassifier
param_grid = {'n_estimators': [5, 50, 250, 500, 1000],
              'learning_rate': [0.05, 0.10, 0.15, 0.20],
              'max_depth': [ 1, 3, 5, 8, 10],
              'min_child_weight': [ 1, 3, 5, 7, 10],
              'gamma': [0.1, 0.5, 1, 1.5, 2]}

kfold = StratifiedKFold(n_splits=10, shuffle=True)
grid_search = GridSearchCV(xgb, param_grid, scoring='recall', n_jobs=-1, cv=kfold, verbose=3)
grid_result = grid_search.fit(X_smt_Rscaled, y_train_smt_Rscaled)
print('XGBClassifier - ', f'Melhor: {grid_result.best_score_} para {grid_result.best_params_}')
'''

# Classificação Binaria - Variavel Target (0, 1)
# Definindo o melhor modelo

#RandomForestClassifier
model_rf = RandomForestClassifier(max_depth=30, max_features='sqrt', n_estimators=300, min_samples_leaf=1, min_samples_split=2)
model_rf.fit(X_smt_Rscaled, y_train_smt_Rscaled)

X_test_Rscaled = Rob_scaler.transform(X_test)
y_pred_rf = model_rf.predict(X_test_Rscaled)
print('Relatório de Clasfficação - Random Forest Classifier: \n\n', classification_report(y_test, y_pred_rf))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_rf)))


# SVC
model_svc = svm.SVC(kernel='rbf', C=20)
model_svc.fit(X_smt_Rscaled, y_train_smt_Rscaled)

y_pred_svc = model_svc.predict(X_test_Rscaled)
print('Relatório de Clasfficação - SVC: \n\n', classification_report(y_test, y_pred_svc))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_svc)))


# Regressão Logistica
model_rl = LogisticRegression(C=0.0006951927961775605, max_iter=2500, penalty='l2', solver='lbfgs')
model_rl.fit(X_smt_Rscaled, y_train_smt_Rscaled)

y_pred_rl = model_rl.predict(X_test_Rscaled)
print('Relatório de Clasfficação - Logistic Regression: \n\n', classification_report(y_test, y_pred_rl))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_rl)))


#KNeighborsClassifier
model_knn = KNeighborsClassifier(metric='euclidean', n_neighbors=13, weights='distance')
model_knn.fit(X_smt_Rscaled, y_train_smt_Rscaled)

y_pred_knn = model_knn.predict(X_test_Rscaled)
print('Relatório de Clasfficação - KNeighbors Classifier: \n\n', classification_report(y_test, y_pred_knn))

# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_knn)))


# XGBClassifier
model_xgb = XGBClassifier(gamma=0.1, max_depth=20, min_child_weight=1, learning_rate=0.2, n_estimators=1000)
model_xgb.fit(X_smt_Rscaled, y_train_smt_Rscaled)

y_pred_xgb = model_xgb.predict(X_test_Rscaled)
print('Relatório de Clasfficação - XGBClassifier: \n\n', classification_report(y_test, y_pred_xgb))

report = classification_report(y_test, y_pred_xgb, output_dict=True)

df_report = pd.DataFrame(report).transpose()
#dfi.export(df_report, 'XGBClassifier.png')

# Gráfico Matriz de Confusão
f, ax5 = plt.subplots()
ax5 = sns.heatmap(confusion_matrix(y_test, y_pred_xgb), cmap='Blues', fmt='g', annot=True)
ax5.set_xlabel('Predicted Values')
ax5.set_ylabel('Actual Values')
ax5.set_title('Matriz de Confusão', fontsize=12, loc='left', pad=10)
ax5.xaxis.set_ticklabels(['No Failure', 'Failure'])
ax5.yaxis.set_ticklabels(['No Failure', 'Failure'])
plt.tight_layout()
plt.show()


# Imprimir a área sob curva
print('AUC: {:.4f}\n'.format(roc_auc_score(y_test, y_pred_xgb)))

# Gráfico ROC-Curve
y_pred_prob_XGB = model_xgb.predict_proba(X_test_Rscaled)[:,1]

fpr_XGB, tpr_XGB, _ = roc_curve(y_test, y_pred_xgb)
roc_auc_XGB_smt = auc(fpr_XGB, tpr_XGB)

fig = plt.figure(figsize=(10, 8))
plt.plot(fpr_XGB, tpr_XGB, label='XGBClassifier = %0.2f' % roc_auc_XGB_smt, color='blue')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC-Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc=4)
plt.show()


# Classificação Multiclasse - No Failure e 'Heat Dissipation Failure', 'Power Failure', 'Overstrain Failure', 'Tool Wear Failure'
df2 = df1.copy()

df2.drop(df2.loc[df2['failure_type']=='Random Failures'].index, inplace=True)

df2['failure_type'] = df2['failure_type'].replace(['No Failure', 'Heat Dissipation Failure', 'Power Failure', 'Overstrain Failure', 'Tool Wear Failure'], [0, 1, 2, 3, 4])


X_multi = df2.drop(['target', 'failure_type'], axis=1)
y_multi = df2['failure_type']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.25, random_state=42)

# Padronização dos dados
Rob_scaler = RobustScaler()
X_multi_Rscaled = Rob_scaler.fit_transform(X_train_m)

# Balanceamento dos dados - Oversample
# Smote
smote = SMOTE(random_state=42)
X_multi_smt_Rscaled, y_multi_smt_Rscaled = smote.fit_resample(X_multi_Rscaled, y_train_m)

# Checando o balanceamento das classes
sns.countplot(x=y_multi_smt_Rscaled, color = sns.xkcd_rgb['windows blue'])
plt.title('Balanceamento SMOTE', fontsize=12, loc='left', pad=10)
plt.tight_layout()
plt.show()


# Hiperparâmetros
'''
#RandomForestClassifier
param_grid = {'max_depth': [5, 10, 20, 30, 40, 50, 100],
              'n_estimators': [10, 50, 100, 300, 1000]}

kfold = StratifiedKFold(n_splits=10, shuffle=True)
grid_search = GridSearchCV(rf, param_grid=param_grid, n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_multi_smt_Rscaled, y_multi_smt_Rscaled)
print('RandomForestClassifier - ', f'Melhor: {grid_result.best_score_} para {grid_result.best_params_}')

#SVC
kfold = StratifiedKFold(n_splits=10, shuffle=True)
clf_svc = RandomizedSearchCV(svm.SVC(gamma='auto'), {
          'C': [100, 10, 20, 1.0, 0.1, 0.001],
          'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}, n_jobs=-1, n_iter=15, cv=kfold)

clf_svc.fit(X_multi_smt_Rscaled, y_multi_smt_Rscaled)
print('SVC - ', clf_svc.best_params_)

# LogisticRegression
param_grid = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
              'C': np.logspace(-4, 4, 20),
              'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
              'max_iter': [100, 1000, 2500, 5000]}

grid_search = GridSearchCV(lr, param_grid=param_grid, cv=5, return_train_score=False)
grid_result = grid_search.fit(X_multi_smt_Rscaled, y_multi_smt_Rscaled)
print('LogisticRegression - ', f'Melhor: {grid_result.best_score_} para {grid_result.best_params_}')

# XGBClassifier
param_grid = {'n_estimators': [5, 50, 250, 500, 1000],
              'learning_rate': [0.05, 0.10, 0.15, 0.20],
              'max_depth': [ 1, 3, 5, 8, 10],
              'min_child_weight': [ 1, 3, 5, 7, 10],
              'gamma': [0.1, 0.5, 1, 1.5, 2],
              'objective' : ['multi:softprob']}

kfold = StratifiedKFold(n_splits=5, shuffle=True)
grid_search = GridSearchCV(xgb, param_grid, scoring='recall', n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_multi_smt_Rscaled, y_multi_smt_Rscaled)
print('XGBClassifier - ', f'Melhor: {grid_result.best_score_} para {grid_result.best_params_}')
'''

# Definindo o melhor modelo
# SVC
svc_multi = svm.SVC(kernel='rbf', C=20)
svc_multi.fit(X_multi_smt_Rscaled, y_multi_smt_Rscaled)

X_test_multi_Rscaled = Rob_scaler.transform(X_test_m)
y_pred_multi_svc = svc_multi.predict(X_test_multi_Rscaled)
print('Relatório de Clasfficação Multiclasse - SVC: \n\n', classification_report(y_test_m, y_pred_multi_svc))


# XGBClassifier
xgb_multi = XGBClassifier(gamma=0.1, max_depth=10, min_child_weight=1, learning_rate=0.2, n_estimators=300, objective='multi:softmax')
xgb_multi.fit(X_multi_smt_Rscaled, y_multi_smt_Rscaled)

y_pred_multi_xgb= xgb_multi.predict(X_test_multi_Rscaled)
print('Relatório de Clasfficação Multiclasse - XGBClassifier: \n\n', classification_report(y_test_m, y_pred_multi_xgb))

report_multi = classification_report(y_test_m, y_pred_multi_xgb, output_dict=True)

df_report_multi = pd.DataFrame(report_multi).transpose()
dfi.export(df_report_multi, 'XGBMulticlasse.png')


# Gráfico Matriz de Confusão
f, ax6 = plt.subplots()
ax6 = sns.heatmap(confusion_matrix(y_test_m, y_pred_multi_xgb), cmap='Blues', fmt='g', annot=True)
ax6.set_xlabel('Predicted Values')
ax6.set_ylabel('Actual Values')
ax6.set_title('Matriz de Confusão', fontsize=12, loc='left', pad=10)
ax6.xaxis.set_ticklabels(['No Fail', 'HDF', 'PWF', 'OSF', 'TWF'])
ax6.yaxis.set_ticklabels(['No Fail', 'HDF', 'PWF', 'OSF', 'TWF'])
plt.tight_layout()
plt.show()


#RandomForestClassifier
rf_multi = RandomForestClassifier(max_depth=30, max_features='sqrt', n_estimators=10, min_samples_leaf=1, min_samples_split=2)
rf_multi.fit(X_multi_smt_Rscaled, y_multi_smt_Rscaled)

y_pred_multi_rf = rf_multi.predict(X_test_multi_Rscaled)
print('Relatório de Clasfficação - Random Forest Classifier: \n\n', classification_report(y_test_m, y_pred_multi_rf))


# Previsão de falhas
df_predict = pd.read_csv('ManutencaoPreditiva/desafio_manutencao_preditiva_teste.csv', sep=',')
df_predict = df_predict.drop(['udi', 'product_id'], axis=1)

label_encoder = LabelEncoder()
label_encoder.fit(df_predict['type'])
df_predict['type'] = label_encoder.transform(df_predict['type'])

xgb_multi.fit(X_train_m, y_train_m)
predict = xgb_multi.predict(df_predict)

results = pd.DataFrame({'rowNumber': df_predict.index,'predictedValues': predict})
results.set_index('rowNumber', inplace=True)

# Salvar resultado em CSV
results.to_csv('predicted.csv')