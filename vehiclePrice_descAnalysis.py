import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv(r'C:\Users\miche\OneDrive\Cursos\Pos MLP\IML\IML1\Ativ1\atividade-1-precificacao-veiculos.csv')
df = df.drop(['ID', 'symboling', 'carheight', 'stroke', 'compressionratio', 'peakrpm'], axis=1)
df['cylindernumber'] = df['cylindernumber'].replace({'one': 1, 'two': 2, 'three': 3,'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12})

shape = df.shape
print(shape)
print(df.dtypes)
#print(df.isnull().sum())

profile = ProfileReport(df)
profile.to_file('profVehiclePrice_v2.html')
print(df.head())
