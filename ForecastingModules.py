import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

###################
## ERROR METRICS ##
###################

#### MAE ####
def MAE(actual: np.ndarray, predicted: np.ndarray):
	""" Mean Absolute Error """
	result=0
	for i in range(len(actual)):
		result += abs(actual[i]-predicted[i])
	result /= len(actual)
	result *= 100
	return result

#### MAPE ####
def MAPE(actual: np.ndarray, predicted: np.ndarray):
	""" Mean Absolute Percentage Error """
	MAPE=0
	for i in range(len(actual)):
		MAPE += abs((actual[i]-predicted[i])/actual[i])
	MAPE /= len(actual)
	MAPE *= 100
	return MAPE

#### MSE ####
def MSE(actual: np.ndarray, predicted: np.ndarray):
	""" Mean Squared Error """
	MSE=mean_squared_error(actual,predicted)
	return MSE
	
#### RMSE ####
def RMSE(actual: np.ndarray, predicted: np.ndarray):
	""" Root Mean Squared Error """
	MSE=mean_squared_error(actual,predicted)
	RMSE=np.sqrt(MSE)
	return RMSE

#### dRMSE ####
def DRMSE(actual: np.ndarray, predicted: np.ndarray):
	dy_true=np.diff(actual, axis=0)
	dy_pred=np.diff(predicted, axis=0)
	DRMSE=0
	for i in range(len(dy_true)):
		dRMSE += ((dy_true[i]-dy_pred[i])/dy_true[i])**2
	DRMSE /= len(dy_true)
	DRMSE = np.sqrt(DRMSE)
	return DRMSE

#### NRMSE ####
def NRMSE(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    NRMSE = RMSE(actual, predicted) / (actual.max() - actual.min())
    return NRMSE

#### RMSLE ####
def RMSLE(actual: np.ndarray, predicted: np.ndarray):
	""" Root Mean Squared Logarithmic Error """
	RMSLE = np.sqrt(mean_squared_log_error(actual, predicted))
	return RMSLE