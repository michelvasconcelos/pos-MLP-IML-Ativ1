import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas_profiling import ProfileReport
import numpy as np

"""Neste arquivo converti em número a coluna cylindernumber"""
"""Normalizei os campos numéricos"""
"""Gerei novo ProfileReport"""

df = pd.read_csv(r'C:\Users\miche\OneDrive\Cursos\Pos MLP\IML\IML1\Ativ1\atividade-1-precificacao-veiculos.csv')
df = df.drop(['ID', 'symboling', 'carheight', 'stroke', 'compressionratio', 'peakrpm', 'fueltypes', 'aspiration', 'doornumbers', 'carbody', 'drivewheels', 'enginelocation', 'enginetype', 'fuelsystem'], axis=1) #, 'price'
df['cylindernumber'] = df['cylindernumber'].replace({'one': 1, 'two': 2, 'three': 3,'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12})
print(df.dtypes)

df.set_index('name', inplace=True)
print(df.head())

profile = ProfileReport(df)
profile.to_file('profVehiclePrice_v4.html')
