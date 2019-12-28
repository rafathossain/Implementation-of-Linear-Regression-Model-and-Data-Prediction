import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def predict_co2_vs_alt(data):
	regression_co2 = LinearRegression()
	regression_co2.fit(data[['Altitude']], data.CO2)

	co2_predict = []
	for alt in data.Altitude:
		co2_predict.append(regression_co2.predict([[alt]])[0])

	return co2_predict


def predict_co2_vs_temp(data):
	regression_co2 = LinearRegression()
	regression_co2.fit(data[['Temperature']], data.CO2)

	co2_predict = []
	for temp in data.Temperature:
		co2_predict.append(regression_co2.predict([[temp]])[0])

	return co2_predict


def predict_co2_vs_hum(data):
	regression_co2 = LinearRegression()
	regression_co2.fit(data[['Humidity']], data.CO2)

	co2_predict = []
	for hum in data.Humidity:
		co2_predict.append(regression_co2.predict([[hum]])[0])

	return co2_predict


def predict_co_vs_alt(data):
	regression_co = LinearRegression()
	regression_co.fit(data[['Altitude']], data.CO)

	co_predict = []
	for alt in data.Altitude:
		co_predict.append(regression_co.predict([[alt]])[0])

	return co_predict


def predict_co_vs_temp(data):
	regression_co = LinearRegression()
	regression_co.fit(data[['Temperature']], data.CO)

	co_predict = []
	for temp in data.Temperature:
		co_predict.append(regression_co.predict([[temp]])[0])

	return co_predict


def predict_co_vs_hum(data):
	regression_co = LinearRegression()
	regression_co.fit(data[['Humidity']], data.CO)

	co_predict = []
	for hum in data.Humidity:
		co_predict.append(regression_co.predict([[hum]])[0])

	return co_predict


def predict_air_vs_alt(data):
	regression_air = LinearRegression()
	regression_air.fit(data[['Altitude']], data.Air)

	air_predict = []
	for alt in data.Altitude:
		air_predict.append(regression_air.predict([[alt]])[0])

	return air_predict


def predict_air_vs_temp(data):
	regression_air = LinearRegression()
	regression_air.fit(data[['Temperature']], data.Air)

	air_predict = []
	for temp in data.Temperature:
		air_predict.append(regression_air.predict([[temp]])[0])

	return air_predict


def predict_air_vs_hum(data):
	regression_air = LinearRegression()
	regression_air.fit(data[['Humidity']], data.Air)

	air_predict = []
	for hum in data.Humidity:
		air_predict.append(regression_air.predict([[hum]])[0])

	return air_predict


if __name__ == "__main__":
	dataset = pd.read_csv('Mirpur.csv')
	alt_vs_co2 = predict_co2_vs_alt(dataset)
	temp_vs_co2 = predict_co2_vs_temp(dataset)
	hum_vs_co2 = predict_co2_vs_hum(dataset)

	plt.title('Altitude vs CO2 (Linear Regression) | Location: MIRPUR')
	plt.xlabel('Altitude (Meter)')
	plt.ylabel('CO2 (PPM)')
	plt.scatter(dataset.Altitude, dataset.CO2, color='gray')
	plt.plot(dataset.Altitude, alt_vs_co2, color='red', linewidth=2)
	plt.legend(['Prediction Line', 'Raw Values'])
	plt.tight_layout()
	Fig_alt_vs_co2 = plt.gcf()

	plt.show()

	plt.title('Temperature vs CO2 (Linear Regression) | Location: MIRPUR')
	plt.xlabel('Temperature (Celsius)')
	plt.ylabel('CO2 (PPM)')
	plt.scatter(dataset.Temperature, dataset.CO2, color='gray')
	plt.plot(dataset.Temperature, temp_vs_co2, color='red', linewidth=2)
	plt.legend(['Prediction Line', 'Raw Values'])
	plt.tight_layout()
	Fig_temp_vs_co2 = plt.gcf()

	plt.show()

	plt.title('Humidity vs CO2 (Linear Regression) | Location: MIRPUR')
	plt.xlabel('Humidity (%)')
	plt.ylabel('CO2 (PPM)')
	plt.scatter(dataset.Humidity, dataset.CO2, color='gray')
	plt.plot(dataset.Humidity, hum_vs_co2, color='red', linewidth=2)
	plt.legend(['Prediction Line', 'Raw Values'])
	plt.tight_layout()
	Fig_hum_vs_co2 = plt.gcf()

	plt.show()

	Fig_alt_vs_co2.savefig('Alt vs CO2.png', dpi=1200)
	Fig_temp_vs_co2.savefig('Temp vs CO2.png', dpi=1200)
	Fig_hum_vs_co2.savefig('Hum vs CO2.png', dpi=1200)

	alt_vs_co = predict_co_vs_alt(dataset)
	temp_vs_co = predict_co_vs_temp(dataset)
	hum_vs_co = predict_co_vs_hum(dataset)

	plt.title('Altitude vs CO (Linear Regression) | Location: MIRPUR')
	plt.xlabel('Altitude (Meter)')
	plt.ylabel('CO (PPM)')
	plt.scatter(dataset.Altitude, dataset.CO, color='gray')
	plt.plot(dataset.Altitude, alt_vs_co, color='red', linewidth=2)
	plt.legend(['Prediction Line', 'Raw Values'])
	plt.tight_layout()
	Fig_alt_vs_co = plt.gcf()

	plt.show()

	plt.title('Temperature vs CO (Linear Regression) | Location: MIRPUR')
	plt.xlabel('Temperature (Celsius)')
	plt.ylabel('CO (PPM)')
	plt.scatter(dataset.Temperature, dataset.CO, color='gray')
	plt.plot(dataset.Temperature, temp_vs_co, color='red', linewidth=2)
	plt.legend(['Prediction Line', 'Raw Values'])
	plt.tight_layout()
	Fig_temp_vs_co = plt.gcf()

	plt.show()

	plt.title('Humidity vs CO (Linear Regression) | Location: MIRPUR')
	plt.xlabel('Humidity (%)')
	plt.ylabel('CO (PPM)')
	plt.scatter(dataset.Humidity, dataset.CO, color='gray')
	plt.plot(dataset.Humidity, hum_vs_co, color='red', linewidth=2)
	plt.legend(['Prediction Line', 'Raw Values'])
	plt.tight_layout()
	Fig_hum_vs_co = plt.gcf()

	plt.show()

	Fig_alt_vs_co.savefig('Alt vs CO.png', dpi=1200)
	Fig_temp_vs_co.savefig('Temp vs CO.png', dpi=1200)
	Fig_hum_vs_co.savefig('Hum vs CO.png', dpi=1200)

	alt_vs_air = predict_air_vs_alt(dataset)
	temp_vs_air = predict_air_vs_temp(dataset)
	hum_vs_air = predict_air_vs_hum(dataset)

	plt.title('Altitude vs Air Quality (Linear Regression) | Location: MIRPUR')
	plt.xlabel('Altitude (Meter)')
	plt.ylabel('Air Quality Value')
	plt.scatter(dataset.Altitude, dataset.Air, color='gray')
	plt.plot(dataset.Altitude, alt_vs_air, color='red', linewidth=2)
	plt.legend(['Prediction Line', 'Raw Values'])
	plt.tight_layout()
	Fig_alt_vs_air = plt.gcf()

	plt.show()

	plt.title('Temperature vs Air Quality (Linear Regression) | Location: MIRPUR')
	plt.xlabel('Temperature (Celsius)')
	plt.ylabel('Air Quality Value')
	plt.scatter(dataset.Temperature, dataset.Air, color='gray')
	plt.plot(dataset.Temperature, temp_vs_air, color='red', linewidth=2)
	plt.legend(['Prediction Line', 'Raw Values'])
	plt.tight_layout()
	Fig_temp_vs_air = plt.gcf()

	plt.show()

	plt.title('Humidity vs Air Quality (Linear Regression) | Location: MIRPUR')
	plt.xlabel('Humidity (%)')
	plt.ylabel('Air Quality Value')
	plt.scatter(dataset.Humidity, dataset.Air, color='gray')
	plt.plot(dataset.Humidity, hum_vs_air, color='red', linewidth=2)
	plt.legend(['Prediction Line', 'Raw Values'])
	plt.tight_layout()
	Fig_hum_vs_air = plt.gcf()

	plt.show()

	Fig_alt_vs_air.savefig('Alt vs Air.png', dpi=1200)
	Fig_temp_vs_air.savefig('Temp vs Air.png', dpi=1200)
	Fig_hum_vs_air.savefig('Hum vs Air.png', dpi=1200)
