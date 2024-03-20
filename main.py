from Regression.LinReg import LinReg
import pandas as pd
from sklearn import linear_model

# data = pd.read_csv("Datasets/Salary_Data.csv")
# result = LinReg.linear_regression(data["YearsExperience"], data["Salary"])

# print(result)

# print(result.predict(1))


df = pd.read_csv("Datasets/Sample_Multidata.csv")

X = df[['Age', 'Mileage']]
y = df['Price']

regr = linear_model.LinearRegression()
result = regr.fit(X, y)

print(f"pred1 = {result.predict([[2300, 1300]])}")

print(result.coef_)
print(result.intercept_)

result = LinReg.linear_regression(X, y)
print(f"pred2 = {result.predict([2300, 1300])}")
print(result)

# from ANN.Model import Model

# model = Model()
# model.add_layer(2)
# model.add_layer(3, 'relu')
# model.add_layer(3, 'relu')

# model.summary()

# model.fit([2, 5])