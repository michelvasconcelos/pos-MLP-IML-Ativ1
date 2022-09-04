import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas_profiling import ProfileReport
import numpy as np

"""Neste arquivo converti em número a coluna cylindernumber"""
"""Normalizei os campos numéricos"""
"""Gerei novo ProfileReport"""

df = pd.read_csv(r'C:\Users\miche\OneDrive\Cursos\Pos MLP\IML\IML1\Ativ1\atividade-1-precificacao-veiculos.csv')
dfname = df[['name']].copy()
dfprice = df[['price']].copy()
df = df.drop(['ID', 'symboling', 'carheight', 'stroke', 'compressionratio', 'peakrpm', 'name', 'fueltypes', 'aspiration', 'doornumbers', 'carbody', 'drivewheels', 'enginelocation', 'enginetype', 'fuelsystem', 'price'], axis=1)
#print(df.head())
df['cylindernumber'] = df['cylindernumber'].replace({'one': 1, 'two': 2, 'three': 3,'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12})
#print(df[['cylindernumber']])
print(df.dtypes)
#print(dfname)
scaler = MinMaxScaler()
arr = scaler.fit_transform(df)

df2 = pd.DataFrame(arr, columns= ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'cylindernumber', 'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg'])
df2[['name']] = dfname.copy()
df2[['price']] = dfprice.copy()
df2.set_index('name', inplace=True)
print(df2.head())
profile = ProfileReport(df2)
profile.to_file('profVehiclePrice_v3.html')
