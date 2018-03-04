import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import os
import numpy as np

FILENAME = os.path.join(os.path.dirname(__file__), "data.2.csv")
df = pd.read_csv(FILENAME).astype('float32', errors='ignore').sort_values('MPG.city')

# print(df)
x_values = df[['MPG.city']].values
y_values = df[['Price']].values

body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()