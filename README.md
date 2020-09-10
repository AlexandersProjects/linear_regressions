# linear_regressions
Here I performed some linear, multiple and polynomial regressions

import pandas as pd

df = pd.read_csv('diamonds.csv')

df.head()

df.info()

import seaborn as sns

sns.pairplot(df[['price','carat','x','y','z']]);

### simple linear regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr

x = df['carat']
y = df['price']

x = x.to_numpy()
y = y.to_numpy()

X = x.reshape((-1,1))
Y = y.reshape((-1,1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 0.25,shuffle = True)

model = lr()

model.fit(X_train,Y_train)

model.intercept_

model.coef_

model.score(X_test,Y_test)

import numpy as np
yp = model.predict(np.array([3]).reshape((-1,1)))
yp

yp = model.predict(np.array([2]).reshape((-1,1)))
yp

import matplotlib.pyplot as plt

yp_test = model.predict(X_test)

plt.scatter(X_test, Y_test, c='blue', label='Test')
plt.plot(X_test, yp_test, c='black', label='Regression-line')
plt.scatter(X_train, Y_train, c='orange', label='Train')
# plt.ylim(0,10)
plt.legend();

### Multiple lineare Regression


df.head()

df.describe()

x = df[['x', 'y', 'z']]
y = df['price']

X = x.values
Y = y.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size =0.25)

model = lr()

model.fit(X_train, Y_train)

model.score(X_test, Y_test)

model.intercept_

model.coef_

### Polynomiale Regression



X = df[['x','y', 'z']].values
Y = df[['price']].values

from sklearn.preprocessing import PolynomialFeatures

model = lr()

pf = PolynomialFeatures(degree = 2)

pf.fit(X_train)

X_train_transformed = pf.transform(X_train)
X_test_transformed = pf.transform(X_test)

model.fit(X_train_transformed, Y_train)

print(model.score(X_test_transformed, Y_test))

model.intercept_

model.coef_
