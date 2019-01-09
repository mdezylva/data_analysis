import pandas as pd
import statsmodels.api as sm

from sklearn import linear_model

data = pd.read_csv("data.csv")

data.index
data.columns

data.plot.scatter('Height(Inches)', 'Weight(Pounds)')

X = data["Height(Inches)"]
y = data["Weight(Pounds)"]

model1 = sm.OLS(y, X).fit()
predictions = model1.predict(X)

model1.summary()

lm = linear_model.LinearRegression()

X2 = data
y = data["Weight(Pounds)"]

model2 = lm.fit(X2, y)
predictions = lm.predict(X2)
print(predictions)
lm.score(X2,y) # This seems wrong. Outputs R^2 of 1, not true

lm.coef_

lm.intercept_
