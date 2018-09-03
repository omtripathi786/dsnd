# TODO: Add import statements

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv("CSV_DATA/bmi_and_life_expectancy.csv")

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)
print(laos_life_exp)

x, y = bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']]
y_pred = bmi_life_model.predict(x)
#plt.scatter(x,y, label='Life Expectancy')
#plt.xlabel('BMI')
#plt.ylabel('Life expectancy')
#plt.legend()
#plt.show()
plt.scatter(x, y,  color='black')
plt.xlabel('BMI')
plt.ylabel('Life expectancy')
plt.legend()
plt.plot(x, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()