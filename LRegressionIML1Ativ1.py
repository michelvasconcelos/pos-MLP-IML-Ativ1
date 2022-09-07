import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv(r'C:\Users\miche\OneDrive\Cursos\Pos MLP\IML\IML1\Ativ1\atividade-1-precificacao-veiculos.csv')
dfname = df[['name']].copy()
dfprice = df[['price']].copy()
df = df.drop(['ID', 'symboling', 'carheight', 'stroke', 'compressionratio', 'peakrpm', 'name', 'fueltypes', 'aspiration', 'doornumbers', 'carbody', 'drivewheels', 'enginelocation', 'enginetype', 'fuelsystem', 'price'], axis=1) #
df['cylindernumber'] = df['cylindernumber'].replace({'one': 1, 'two': 2, 'three': 3,'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12})

scaler = MinMaxScaler()
arr = scaler.fit_transform(df)

df2 = pd.DataFrame(arr, columns= ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'cylindernumber', 'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg'])
df2[['name']] = dfname.copy()
df2[['price']] = dfprice.copy()
df2.set_index('name', inplace=True)
print(df2.head())

model = LinearRegression()
X = df2[[ 'enginesize','carwidth', 'curbweight', 'horsepower', 'highwaympg']] #'cylindernumber','boreratio', 'wheelbase', 'carlength',  'citympg'
y = df2['price']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 10)
#model.fit(X, y)
model.fit(x_train, y_train)
modelResult = model.score(x_test, y_test)
print('R2: %.4f' % modelResult)
print(type(x_train))
#predictions = model.predict(x_test)
#print('MSE: %.4f' % mean_squared_error(predictions, y_test))