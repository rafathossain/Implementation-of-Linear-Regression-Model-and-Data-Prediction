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
	dataset_uttara = pd.read_csv('Uttara.csv')
	dataset_aftabnagar = pd.read_csv('Aftabnagar.csv')
	dataset_mirpur = pd.read_csv('Mirpur.csv')

	# Uttara
	alt_vs_co2_uttara = predict_co2_vs_alt(dataset_uttara)
	temp_vs_co2_uttara = predict_co2_vs_temp(dataset_uttara)
	hum_vs_co2_uttara = predict_co2_vs_hum(dataset_uttara)

	# Aftabnagar
	alt_vs_co2_aftabnagar = predict_co2_vs_alt(dataset_aftabnagar)
	temp_vs_co2_aftabnagar = predict_co2_vs_temp(dataset_aftabnagar)
	hum_vs_co2_aftabnagar = predict_co2_vs_hum(dataset_aftabnagar)

	# Mirpur
	alt_vs_co2_mirpur = predict_co2_vs_alt(dataset_mirpur)
	temp_vs_co2_mirpur = predict_co2_vs_temp(dataset_mirpur)
	hum_vs_co2_mirpur = predict_co2_vs_hum(dataset_mirpur)

	params = {'mathtext.default': 'regular'}
	plt.rcParams.update(params)
	plt.rcParams['figure.figsize'] = 10, 8
	plt.title('Altitude vs $CO_2$ (Linear Regression)')
	plt.xlabel('Altitude (Meter)')
	plt.ylabel('$CO_2$ (PPM)')
	plt.scatter(dataset_uttara.Altitude, dataset_uttara.CO2, color='blue', marker='s')
	plt.scatter(dataset_aftabnagar.Altitude, dataset_aftabnagar.CO2, color='green', marker='o')
	plt.scatter(dataset_mirpur.Altitude, dataset_mirpur.CO2, color='goldenrod', marker='x')
	plt.plot(dataset_uttara.Altitude, alt_vs_co2_uttara, color='red', linewidth=2)
	plt.plot(dataset_aftabnagar.Altitude, alt_vs_co2_aftabnagar, color='gold', linewidth=2)
	plt.plot(dataset_mirpur.Altitude, alt_vs_co2_mirpur, color='lime', linewidth=2)
	plt.legend(['Prediction (Uttara)', 'Prediction (Aftabnagar)', 'Prediction (Mirpur)', 'Uttara', 'Aftabnagar', 'Mirpur'])
	plt.tight_layout()
	Fig_alt_vs_co2 = plt.gcf()

	plt.show()

	plt.title('Temperature vs $CO_2$ (Linear Regression)')
	plt.xlabel('Temperature (Celsius)')
	plt.ylabel('$CO_2$ (PPM)')
	plt.scatter(dataset_uttara.Temperature, dataset_uttara.CO2, color='blue', marker='s')
	plt.scatter(dataset_aftabnagar.Temperature, dataset_aftabnagar.CO2, color='green', marker='o')
	plt.scatter(dataset_mirpur.Temperature, dataset_mirpur.CO2, color='goldenrod', marker='x')
	plt.plot(dataset_uttara.Temperature, temp_vs_co2_uttara, color='red', linewidth=2)
	plt.plot(dataset_aftabnagar.Temperature, temp_vs_co2_aftabnagar, color='gold', linewidth=2)
	plt.plot(dataset_mirpur.Temperature, temp_vs_co2_mirpur, color='lime', linewidth=2)
	plt.legend(['Prediction (Uttara)', 'Prediction (Aftabnagar)', 'Prediction (Mirpur)', 'Uttara', 'Aftabnagar', 'Mirpur'])
	plt.tight_layout()
	Fig_temp_vs_co2 = plt.gcf()

	plt.show()

	plt.title('Humidity vs $CO_2$ (Linear Regression)')
	plt.xlabel('Humidity (%)')
	plt.ylabel('$CO_2$ (PPM)')
	plt.scatter(dataset_uttara.Humidity, dataset_uttara.CO2, color='blue', marker='s')
	plt.scatter(dataset_aftabnagar.Humidity, dataset_aftabnagar.CO2, color='green', marker='o')
	plt.scatter(dataset_mirpur.Humidity, dataset_mirpur.CO2, color='goldenrod', marker='x')
	plt.plot(dataset_uttara.Humidity, hum_vs_co2_uttara, color='red', linewidth=2)
	plt.plot(dataset_aftabnagar.Humidity, hum_vs_co2_aftabnagar, color='gold', linewidth=2)
	plt.plot(dataset_mirpur.Humidity, hum_vs_co2_mirpur, color='lime', linewidth=2)
	plt.legend(['Prediction (Uttara)', 'Prediction (Aftabnagar)', 'Prediction (Mirpur)', 'Uttara', 'Aftabnagar', 'Mirpur'])
	plt.tight_layout()
	Fig_hum_vs_co2 = plt.gcf()

	plt.show()

	Fig_alt_vs_co2.savefig('Alt vs CO2.png', dpi=1200)
	Fig_temp_vs_co2.savefig('Temp vs CO2.png', dpi=1200)
	Fig_hum_vs_co2.savefig('Hum vs CO2.png', dpi=1200)

	# Uttara
	alt_vs_co_uttara = predict_co_vs_alt(dataset_uttara)
	temp_vs_co_uttara = predict_co_vs_temp(dataset_uttara)
	hum_vs_co_uttara = predict_co_vs_hum(dataset_uttara)

	# Aftabnagar
	alt_vs_co_aftabnagar = predict_co_vs_alt(dataset_aftabnagar)
	temp_vs_co_aftabnagar = predict_co_vs_temp(dataset_aftabnagar)
	hum_vs_co_aftabnagar = predict_co_vs_hum(dataset_aftabnagar)

	# Mirpur
	alt_vs_co_mirpur = predict_co_vs_alt(dataset_mirpur)
	temp_vs_co_mirpur = predict_co_vs_temp(dataset_mirpur)
	hum_vs_co_mirpur = predict_co_vs_hum(dataset_mirpur)

	params = {'mathtext.default': 'regular'}
	plt.rcParams.update(params)
	plt.rcParams['figure.figsize'] = 10, 8
	plt.title('Altitude vs CO (Linear Regression)')
	plt.xlabel('Altitude (Meter)')
	plt.ylabel('CO (PPM)')
	plt.scatter(dataset_uttara.Altitude, dataset_uttara.CO, color='blue', marker='s')
	plt.scatter(dataset_aftabnagar.Altitude, dataset_aftabnagar.CO, color='green', marker='o')
	plt.scatter(dataset_mirpur.Altitude, dataset_mirpur.CO, color='goldenrod', marker='x')
	plt.plot(dataset_uttara.Altitude, alt_vs_co_uttara, color='red', linewidth=2)
	plt.plot(dataset_aftabnagar.Altitude, alt_vs_co_aftabnagar, color='gold', linewidth=2)
	plt.plot(dataset_mirpur.Altitude, alt_vs_co_mirpur, color='lime', linewidth=2)
	plt.legend(['Prediction (Uttara)', 'Prediction (Aftabnagar)', 'Prediction (Mirpur)', 'Uttara', 'Aftabnagar', 'Mirpur'])
	plt.tight_layout()
	Fig_alt_vs_co = plt.gcf()

	plt.show()

	plt.title('Temperature vs CO (Linear Regression)')
	plt.xlabel('Temperature (Celsius)')
	plt.ylabel('CO (PPM)')
	plt.scatter(dataset_uttara.Temperature, dataset_uttara.CO, color='blue', marker='s')
	plt.scatter(dataset_aftabnagar.Temperature, dataset_aftabnagar.CO, color='green', marker='o')
	plt.scatter(dataset_mirpur.Temperature, dataset_mirpur.CO, color='goldenrod', marker='x')
	plt.plot(dataset_uttara.Temperature, temp_vs_co_uttara, color='red', linewidth=2)
	plt.plot(dataset_aftabnagar.Temperature, temp_vs_co_aftabnagar, color='gold', linewidth=2)
	plt.plot(dataset_mirpur.Temperature, temp_vs_co_mirpur, color='lime', linewidth=2)
	plt.legend(['Prediction (Uttara)', 'Prediction (Aftabnagar)', 'Prediction (Mirpur)', 'Uttara', 'Aftabnagar', 'Mirpur'])
	plt.tight_layout()
	Fig_temp_vs_co = plt.gcf()

	plt.show()

	plt.title('Humidity vs CO (Linear Regression)')
	plt.xlabel('Humidity (%)')
	plt.ylabel('CO (PPM)')
	plt.scatter(dataset_uttara.Humidity, dataset_uttara.CO, color='blue', marker='s')
	plt.scatter(dataset_aftabnagar.Humidity, dataset_aftabnagar.CO, color='green', marker='o')
	plt.scatter(dataset_mirpur.Humidity, dataset_mirpur.CO, color='goldenrod', marker='x')
	plt.plot(dataset_uttara.Humidity, hum_vs_co_uttara, color='red', linewidth=2)
	plt.plot(dataset_aftabnagar.Humidity, hum_vs_co_aftabnagar, color='gold', linewidth=2)
	plt.plot(dataset_mirpur.Humidity, hum_vs_co_mirpur, color='lime', linewidth=2)
	plt.legend(['Prediction (Uttara)', 'Prediction (Aftabnagar)', 'Prediction (Mirpur)', 'Uttara', 'Aftabnagar', 'Mirpur'])
	plt.tight_layout()
	Fig_hum_vs_co = plt.gcf()

	plt.show()

	Fig_alt_vs_co.savefig('Alt vs CO.png', dpi=1200)
	Fig_temp_vs_co.savefig('Temp vs CO.png', dpi=1200)
	Fig_hum_vs_co.savefig('Hum vs CO.png', dpi=1200)

	# Uttara
	alt_vs_air_uttara = predict_air_vs_alt(dataset_uttara)
	temp_vs_air_uttara = predict_air_vs_temp(dataset_uttara)
	hum_vs_air_uttara = predict_air_vs_hum(dataset_uttara)

	# Aftabnagar
	alt_vs_air_aftabnagar = predict_air_vs_alt(dataset_aftabnagar)
	temp_vs_air_aftabnagar = predict_air_vs_temp(dataset_aftabnagar)
	hum_vs_air_aftabnagar = predict_air_vs_hum(dataset_aftabnagar)

	# Mirpur
	alt_vs_air_mirpur = predict_air_vs_alt(dataset_mirpur)
	temp_vs_air_mirpur = predict_air_vs_temp(dataset_mirpur)
	hum_vs_air_mirpur = predict_air_vs_hum(dataset_mirpur)

	params = {'mathtext.default': 'regular'}
	plt.rcParams.update(params)
	plt.rcParams['figure.figsize'] = 10, 8
	plt.title('Altitude vs Air Quality (Linear Regression)')
	plt.xlabel('Altitude (Meter)')
	plt.ylabel('Air Quality Value')
	plt.scatter(dataset_uttara.Altitude, dataset_uttara.Air, color='blue', marker='s')
	plt.scatter(dataset_aftabnagar.Altitude, dataset_aftabnagar.Air, color='green', marker='o')
	plt.scatter(dataset_mirpur.Altitude, dataset_mirpur.Air, color='goldenrod', marker='x')
	plt.plot(dataset_uttara.Altitude, alt_vs_air_uttara, color='red', linewidth=2)
	plt.plot(dataset_aftabnagar.Altitude, alt_vs_air_aftabnagar, color='gold', linewidth=2)
	plt.plot(dataset_mirpur.Altitude, alt_vs_air_mirpur, color='lime', linewidth=2)
	plt.legend(['Prediction (Uttara)', 'Prediction (Aftabnagar)', 'Prediction (Mirpur)', 'Uttara', 'Aftabnagar', 'Mirpur'])
	plt.tight_layout()
	Fig_alt_vs_air = plt.gcf()

	plt.show()

	plt.title('Temperature vs Air Quality (Linear Regression)')
	plt.xlabel('Temperature (Celsius)')
	plt.ylabel('Air Quality Value')
	plt.scatter(dataset_uttara.Temperature, dataset_uttara.Air, color='blue', marker='s')
	plt.scatter(dataset_aftabnagar.Temperature, dataset_aftabnagar.Air, color='green', marker='o')
	plt.scatter(dataset_mirpur.Temperature, dataset_mirpur.Air, color='goldenrod', marker='x')
	plt.plot(dataset_uttara.Temperature, temp_vs_air_uttara, color='red', linewidth=2)
	plt.plot(dataset_aftabnagar.Temperature, temp_vs_air_aftabnagar, color='gold', linewidth=2)
	plt.plot(dataset_mirpur.Temperature, temp_vs_air_mirpur, color='lime', linewidth=2)
	plt.legend(['Prediction (Uttara)', 'Prediction (Aftabnagar)', 'Prediction (Mirpur)', 'Uttara', 'Aftabnagar', 'Mirpur'])
	plt.tight_layout()
	Fig_temp_vs_air = plt.gcf()

	plt.show()

	plt.title('Humidity vs Air Quality (Linear Regression)')
	plt.xlabel('Humidity (%)')
	plt.ylabel('Air Quality Value')
	plt.scatter(dataset_uttara.Humidity, dataset_uttara.Air, color='blue', marker='s')
	plt.scatter(dataset_aftabnagar.Humidity, dataset_aftabnagar.Air, color='green', marker='o')
	plt.scatter(dataset_mirpur.Humidity, dataset_mirpur.Air, color='goldenrod', marker='x')
	plt.plot(dataset_uttara.Humidity, hum_vs_air_uttara, color='red', linewidth=2)
	plt.plot(dataset_aftabnagar.Humidity, hum_vs_air_aftabnagar, color='gold', linewidth=2)
	plt.plot(dataset_mirpur.Humidity, hum_vs_air_mirpur, color='lime', linewidth=2)
	plt.legend(['Prediction (Uttara)', 'Prediction (Aftabnagar)', 'Prediction (Mirpur)', 'Uttara', 'Aftabnagar', 'Mirpur'])
	plt.tight_layout()
	Fig_hum_vs_air = plt.gcf()

	plt.show()

	Fig_alt_vs_air.savefig('Alt vs Air.png', dpi=1200)
	Fig_temp_vs_air.savefig('Temp vs Air.png', dpi=1200)
	Fig_hum_vs_air.savefig('Hum vs Air.png', dpi=1200)
