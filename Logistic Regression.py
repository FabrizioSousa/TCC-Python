
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:47:51 2020

@author: Asus
"""
"""conda install -c glemaitre imbalanced-learn"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE

from numpy.random import randn
from numpy.random import seed

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)



"""sns.countplot(x='FL_INADIMPLENTE',data=data, palette='hls')
plt.show()
plt.savefig('count_plot')
"""
"""count_no_sub = len(data[data['FL_INADIMPLENTE']=='Não'])
count_sub = len(data[data['FL_INADIMPLENTE']=='Sim'])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("% não inadimplentes: ", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("% inadimplentes: ", pct_of_sub*100)
"""

"""data.groupby('Idade').mean()"""

"""%matplotlib inline
pd.crosstab(data.Sexo,data.FL_INADIMPLENTE).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')
"""

"""table=pd.crosstab(data.Sexo,data.FL_INADIMPLENTE)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')
"""

"""['ID', 'Idade', 'Sexo', 'FL_INADIMPLENTE'
 , 'FL_CONTAS_ATRASO', 'RENDA_MENSAL', 'GASTO_FIXO'
 , 'GASTO_VARIAVEL', 'FAIXA_GASTO_FIXO', 'FAIXA_GASTO_VARIAVEL']
"""

data = pd.read_csv(r"C:\Users\Asus\Downloads\Estudo\Programacao\Dados Inadimplentes Tratado - Copia.csv", header=0,encoding='latin-1',delimiter=';')
data = data.dropna()
Porc_Gasto_Fixo = data['GASTO_FIXO']/data['RENDA_MENSAL']*100
Porc_Gasto_Fixo





plt.scatter(Porc_Gasto_Fixo, data.FL_INADIMPLENTE,marker='+',color='red' )
sns.regplot(x=Porc_Gasto_Fixo, y=data.FL_INADIMPLENTE, data=data, logistic=True)

X_train, X_Test, y_train, y_test = train_test_split(data[['Porc_Gasto_Fixo']]
,data.FL_INADIMPLENTE,test_size=0.6)

model = LogisticRegression()
model.fit(X_train, y_train)

X_Test
model.predict(X_Test)

model.score(X_Test,y_test)

model.predict_proba(-6000)

model.predict(-5000)

"""
cat_vars=['FL_INADIMPLENTE'
 , 'FL_CONTAS_ATRASO']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['FL_INADIMPLENTE'
 , 'FL_CONTAS_ATRASO']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]

X = data_final.loc[:, data_final.columns != 'FL_INADIMPLENTE']
y = data_final.loc[:, data_final.columns == 'FL_INADIMPLENTE']

logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
"""
# prepare data
data2 = data.Idade
data1 = data. FL_INADIMPLENTE

np.corrcoef(data1, data2)[0, 1]
