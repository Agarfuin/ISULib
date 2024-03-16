# from Regression.LinReg import LinReg
# import pandas as pd

# data = pd.read_csv("Salary_Data.csv")
# result = LinReg.linear_regression(data["YearsExperience"], data["Salary"])

# print(result)

# print(result.predict(1))

from ANN.Model import Model

model = Model()
model.add_layer(2)
model.add_layer(4, 'relu')
# model.add_layer(3, 'relu')

model.summary()