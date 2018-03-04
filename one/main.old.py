import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import os
import numpy as np

FILENAME = os.path.join(os.path.dirname(__file__), "animal-brain-to-body.csv")
df = pd.read_csv(FILENAME).astype('float32', errors='ignore').sort_values('body')

# print(df)
x_values = df['brain'].values
y_values = df['body'].values
order = x_values.argsort()
x_values = [x_values[order]]
y_values = [y_values[order]]

print(x_values)
print(y_values)

body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

print(body_reg.predict(y_values))

# plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
# print([x_values, body_reg.predict(x_values)])
plt.show()