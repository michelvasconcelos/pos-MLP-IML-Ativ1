import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from scipy import stats
from sklearn.pipeline import Pipeline
#import seaborn as sns
#import matplotlib.pyplot as plt

#Leitura de base de dados, exclusão de features não categóricas e conversão de cylinder number em inteiros
df = pd.read_csv(r'C:\Users\miche\OneDrive\Cursos\Pos MLP\IML\IML1\Ativ1\atividade-1-precificacao-veiculos.csv')
dfGraph = df.copy()
#dfname = df[['name']].copy()
df.set_index('name', inplace=True)
dfPrice = df[['price']].copy()
df = df.drop(['ID', 'symboling', 'carheight', 'stroke', 'compressionratio', 'peakrpm', 'fueltypes', 'aspiration', 'doornumbers', 'carbody', 'drivewheels', 'enginelocation', 'enginetype', 'fuelsystem', 'price'], axis=1) # 
df['cylindernumber'] = df['cylindernumber'].replace({'one': 1, 'two': 2, 'three': 3,'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12})

#Após estatística descritiva realizada com ProfileReport
#Remoção de algumas features com base na baixa correlação com preço e entre si, e
#Separação entre treino e teste
X = df[['carwidth', 'curbweight', 'enginesize', 'horsepower', 'highwaympg']] #'cylindernumber','boreratio', 'wheelbase', 'carlength', 'citympg'
y = dfPrice['price']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 10)

#Análise e Remoção de Outliers não realizada


#Realização de normalização
scaler = MinMaxScaler()
arr = scaler.fit_transform(x_train)
x_train = pd.DataFrame(arr, columns= ['carwidth', 'curbweight', 'enginesize', 'horsepower', 'highwaympg'])
arr2 = scaler.fit_transform(x_test)
x_test = pd.DataFrame(arr2, columns= ['carwidth', 'curbweight', 'enginesize', 'horsepower', 'highwaympg'])

#Realização de modelo de regressão e cálculo de indicadores
model = LinearRegression()
model.fit(x_train, y_train)
modelResult = model.score(x_test, y_test)
print('Indicadores de regressão Múltipla.')
print('R2: %.4f' % modelResult)
predictions = model.predict(x_test)
print('MSE: %.2f' % mean_squared_error(predictions, y_test, squared=False))
print('MAE: %.2f' % mean_absolute_error(predictions, y_test))

#Regressão Lasso: Busca de Alpha ótimo e verificação se mais alguma feature foi desconsiderada.
pipeline = Pipeline([('scaler', MinMaxScaler()), ('model', Lasso())])
search = GridSearchCV(pipeline, {'model__alpha':np.arange(0.1, 3.1, 0.1)}, cv = 5, scoring = 'neg_mean_squared_error', verbose=3)
search.fit(x_train, y_train)
print(search.best_params_)
coef = search.best_estimator_[1].coef_
print(coef)
features = np.array(['carwidth', 'curbweight', 'enginesize', 'horsepower', 'highwaympg'])
print(np.array(features)[coef != 0])

#Realização de regressão Lasso (com penalização) e cálculo de indicadores.
LassoModel = Lasso(alpha=1.4).fit(x_train, y_train)
LassoModelResult = LassoModel.score(x_test, y_test)
print('Indicadores de regressão Lasso.')
print('R2: %.4f' % LassoModelResult)
LPredictions = LassoModel.predict(x_test)
print('MSE: %.2f' % mean_squared_error(LPredictions, y_test, squared=False))
print('MAE: %.2f' % mean_absolute_error(LPredictions, y_test))


"""Excluir outliers"""


#Visualização de regressão e dados de teste
#figure, axis = plt.subplots(1, 1, figsize=(5, 5))
#sns.scatterplot(data=dfGraph, x='enginesize', y='price')
#axis.set_title('Engine size x Price');