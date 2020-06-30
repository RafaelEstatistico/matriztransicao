# -*- coding: utf-8 -*-
import requests
import pandas as pd
import numpy as np
import random
from geopy.geocoders import Nominatim
import os

# libraries for displaying images
from IPython.display import Image
from IPython.core.display import HTML
from IPython.display import HTML
from pandas.io.json import json_normalize
import warnings

# opções
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)

'---------------------------------------------------------'
'---           pib municipal atualizado                ---'
'---------------------------------------------------------'

pibma = pd.read_excel('PIB dos Municípios 2010-2017.xls', sheet_name='PIB_dos_Municípios')



pibma.columns.values


pibma.describe().transpose()

pibma.groupby('Ano').count()




'---------------------------------------------------------'
'---     CAPAG: Nota de Crédito dos Municipios         ---'
'---------------------------------------------------------'


capag = pd.read_csv('CAPAG-Municipios.csv',delimiter=';')



capag.dtypes

capag.head()

capag[['CodIBGE','UF']].groupby('UF').count()


capag.shape
capag.columns


capag.describe().transpose()

'---------------------------------------------------------'
'---      Cota do Fundo de Participação Municipal      ---'
'---------------------------------------------------------'



cota = pd.read_excel('Cota do Fundo de Participação Municipal.xls')


cota.head()

'---------------------------------------------------------'
'---    Despesa Municipal com Saúde e Saneamento       ---'
'---------------------------------------------------------'

sanea = pd.read_excel('Despesas por Função Saúde Saneamento.xls')


sanea.head()

'---------------------------------------------------------'
'---     Receita Tributária com Impostos Municipal     ---'
'---------------------------------------------------------'

impostos = pd.read_excel('Receita Tributária Impostos Municipal.xls')


impostos.head()

'---------------------------------------------------------'
'---     Transferências de ICMS para os Municípios     ---'
'---------------------------------------------------------'


icms = pd.read_excel('Transferências ICMS Municípios.xls')


icms.head()


'---------------------------------------------------------'
'---           Exportações e Importações               ---'
'---------------------------------------------------------'

importa = pd.read_excel('IMPORTACAO.2014-2019.xls')


importa.head()



#exporta = pd.read_excel('EXPORTACAO.2014-2019.xls')


#exporta.head()

'---------------------------------------------------------'
'---             Area dos Municípios                   ---'
'---------------------------------------------------------'

area = pd.read_excel('AREA.2013-2018.xls')


area.head()




'---------------------------------------------------------'
'---         Exemplo de Multilayer Perceptron          ---'
'---------------------------------------------------------'

import re
r = re.compile(".*pib_")

newlist = list(filter(r.match, list(df_final.columns.values)))
newlist.remove('pib_0')


X = df_final[newlist]
y = df_final['pib_0']

X.shape

from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from neupy import algorithms
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


#x_train, x_test, y_train, y_test = train_test_split(preprocessing.minmax_scale(X),
#preprocessing.minmax_scale(y),test_size=0.3)


x_train, x_test, y_train, y_test = train_test_split(preprocessing.minmax_scale(X),y,test_size=0.3)

nn = MLPRegressor(hidden_layer_sizes=(17,),  activation='relu', solver='adam',
alpha=0.0005,batch_size='auto', learning_rate='constant', learning_rate_init=0.0025,
power_t=0.5, shuffle=True, random_state=0, tol=0.0001, verbose=False,
warm_start=False,momentum=0.9, nesterovs_momentum=True,
validation_fraction=0.15,beta_1=0.9, beta_2=0.999, epsilon=1e-08)


n = nn.fit(x_train, y_train)
y_test_pred = n.predict(x_test)
print(np.corrcoef(y_test_pred, y_test))


y_all = nn.predict(preprocessing.minmax_scale(X))
print(np.corrcoef(y_all, preprocessing.minmax_scale(y)))



'-----------------------------------------------'
'---       Support Vector Regression         ---'
'-----------------------------------------------'
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge

metricas = ['accuracy', 'average_precision', 'f1', 'precision', 'recall', 'roc_auc']


svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})


svr.fit(x_train, y_train)

#x_train, x_test, y_train, y_test
import time
t0 = time.time()
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"  % svr_fit)

t0 = time.time()
kr.fit(x_train, y_train)
kr_fit = time.time() - t0
print("KRR complexity and bandwidth selected and model fitted in %.3f s"  % kr_fit)
