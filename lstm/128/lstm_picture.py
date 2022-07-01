
# cd C:\chengshu\spring\lstm  ##update
# python lstm_picture.py  ##update

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ##update
import time
import tensorflow as tf
# assert tf.__version__.startswith("2.")  ##update
tf.keras.backend.clear_session()  # For easy reset of notebook state.
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.axislines as axislines
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from tensorflow.python import pywrap_tensorflow
from matplotlib.ticker import PercentFormatter
from tensorflow.keras.utils import plot_model
from scipy.stats import norm


print("\n  Begin ... \n")

#=================================================================================
# Environment
# ===========
t = time.time()
seed = 22
tf.random.set_seed(seed)
np.random.seed(seed)
#---------------------------------------------------------------------------------

#=================================================================================
# Parameters
# ==========
lag = 1   # lag = int(input("input lag (1) = ") )
time_series_length = 1 
each_year_month = 12
#---------------------------------------------------------------------------------

#==================================================================================
# load data
# =========
file_location = r"C:\chengshu\spring\data"
dataset = pd.read_csv(file_location + r'\dataset.csv', header=0, index_col=0)
# from sklearn.utils import shuffle
# dataset = shuffle(dataset)
#---------------------------------------------------------------------------------
values = dataset.values
values = values.astype('float32')  # ensure all data is float
#---------------------------------------------------------------------------------

#=================================================================================
# convert series to supervised learning
# =====================================
def series_to_supervised(data, lag=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(lag, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
#---------------------------------------------------------------------------------
#=================================================================================
# frame as supervised learning
# ============================
reframed = series_to_supervised(values, lag, 1)
#---------------------------------------------------------------------------------

#=================================================================================
# drop columns we don't want to predict
# =====================================
# reframed.drop(reframed.columns[[11,12,13,14,15,16,17,18,19]], axis=1, inplace=True)  # lag=1
print(reframed.head())
#---------------------------------------------------------------------------------

#=================================================================================
# split into train and test sets
# ==============================
values = reframed.values
#---------------------------------------------------------------------------------
# training_number = int(len(dataset) * 0.8)
training_number = 12 * 26 - lag
#---------------------------------------------------------------------------------
train = values[:training_number, :]
validation = values[training_number:, :]
#---------------------------------------------------------------------------------


#=================================================================================
# split into input and outputs
# ============================
x_train, y_train = train[:, :10*lag], train[:, 10*lag].reshape(-1, 1)
x_validation, y_validation = validation[:, :10*lag], validation[:, 10*lag].reshape(-1, 1)
# x_test, y_test = test[:, :4*lag], test[:, 4*lag]
x_all, y_all = values[:, :10*lag], values[:, 10*lag].reshape(-1, 1)
#---------------------------------------------------------------------------------

#=================================================================================
# normalize features
# ==================
x_scaler = MinMaxScaler(feature_range=(0, 1))  # MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
x_train = x_scaler.fit_transform(x_train)
x_validation = x_scaler.fit_transform(x_validation)
x_all = x_scaler.fit_transform(x_all)
#---------------------------------------------------------------------------------
y_scaler = MinMaxScaler(feature_range=(0, 1))  # MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
y_train = y_scaler.fit_transform(y_train)
y_validation = y_scaler.fit_transform(y_validation)
y_all = y_scaler.fit_transform(y_all)
#---------------------------------------------------------------------------------
n_features = values.shape[1]
print("values = ", values)
#---------------------------------------------------------------------------------

#=================================================================================
# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], time_series_length, x_train.shape[1]))
x_validation = x_validation.reshape((x_validation.shape[0], time_series_length, x_validation.shape[1]))
# x_test = x_test.reshape((x_test.shape[0], time_series_length, x_test.shape[1]))
x_all = x_all.reshape((x_all.shape[0], time_series_length, x_all.shape[1]))
#---------------------------------------------------------------------------------
# reshape output to be 2D [samples, features]
y_train = y_train.reshape(y_train.shape[0], 1)
y_validation = y_validation.reshape(y_validation.shape[0], 1)
# y_test = y_test.reshape(y_test.shape[0], 1)
y_all = y_all.reshape(y_all.shape[0], 1)
#---------------------------------------------------------------------------------
print("shape: \n  x_train.shape={0}, y_train.shape{1}, \n  x_validation.shape={2}, y_validation.shape={3}"\
    .format(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape))
#---------------------------------------------------------------------------------

#==================================================================================
# load model
# ==========
# model = keras.models.load_model("./model/model_{epoch:05d}-{val_loss:.5f}.h5")
model = keras.models.load_model("./model/00389-0.001667-0.024294-0.040825-0.001220-0.027269-0.034925.h5")
# model.load_weights("./save/model.ckpt")
# print("layer1.kernel: ", layer1.kernel)
#---------------------------------------------------------------------------------

# #=================================================================================
# # Print weights
# # =============
# checkpoint_file = os.path.join('./model/', "model.ckpt")
# model.load_weights(checkpoint_file)
# model_reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_file)  
# var_dict = model_reader.get_variable_to_shape_map() 
# #---------------------------------------------------------------------------------
# weight_dir = os.path.join("weight")
# if not os.path.exists(weight_dir):
#     os.mkdir(weight_dir)
# for key in var_dict:
# 	w = model_reader.get_tensor(key)
# 	# print(key, w)
# 	# print("variable name: ", key)
# 	if key == "layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUE":
# 		# print(w.shape) 
# 		np.savetxt("./weight/W.csv", w, delimiter=",")
# 	elif key == "layer-0/bias/.ATTRIBUTES/VARIABLE_VALUE":
# 		# print(w.shape)
# 		np.savetxt("./weight/b.csv", w, delimiter=",")
# #---------------------------------------------------------------------------------

#=================================================================================
# predict origin
# ==============
y_pre = model.predict(x_all)
predict = y_scaler.inverse_transform(y_pre)
origin = y_scaler.inverse_transform(y_all)
#---------------------------------------------------------------------------------
diff = (predict - origin)



picture_dir = os.path.join("pictures")
if not os.path.exists(picture_dir):
    os.mkdir(picture_dir)
#=================================================================================
# Picture
# =======
fig = plt.figure(figsize=(16.0, 12.0))
#---------------------------------------------------------------------------------
# subpicture1
ax1 = axislines.Subplot(fig, 311)
fig.add_subplot(ax1)
ax1.axis[:].major_ticks.set_tick_out(True)
n, bins, patches = ax1.hist(diff, bins="auto", density=True, cumulative=False, facecolor="yellowgreen", alpha=0.5, label="histogram")
y = norm.pdf(bins, diff.mean(), diff.std()) # 生成正态分布函数
ax1.plot(bins, y, "b--", label="Gauss Distribution")
ax1.set_title("Histogram of Spring Flow Difference distribution: $\mu = {0:.2f}$, $\sigma={1:.2f}$".format(diff.mean(), diff.std()))
ax1.set_xlabel("Value of Slip Difference(m)")
ax1.set_ylabel("Percent")
ax1.yaxis.set_major_formatter(PercentFormatter(xmax=len(diff)))
ax1.legend(loc="upper left", shadow=True)
plt.savefig("./pictures/self_BP.png")
#---------------------------------------------------------------------------------


#=================================================================================
# Picture 2
# =========
plt.figure()
plt.tick_params(axis='both', which='major', labelsize=11)
# plt.axis[:].major_ticks.set_tick_out(True)
# n, bins, patches = plt.hist(diff, bins=10, density=True, cumulative=False, facecolor="yellowgreen", alpha=0.5, label="histogram")
n, bins, patches = plt.hist(diff, bins=6, range=[-0.30, 0.30], density=True, cumulative=False, facecolor="yellowgreen", alpha=0.5, label="histogram")
# y = norm.pdf(bins, diff.mean(), diff.std()) # 生成正态分布函数
# plt.plot(bins, y, "b--", label="Gauss Distribution")
plt.title("lag={0} Histogram: $\mu = {1:.2f}$, $\sigma={2:.2f}$".format(lag, diff.mean(), diff.std()), fontsize=18)
plt.xlabel("Value of Spring discharge diffrence", fontsize=16)
plt.ylabel("Percent", fontsize=16)
# plt.yaxis.set_major_formatter(PercentFormatter(xmax=len(diff)))
plt.legend(loc="upper right", shadow=True)
plt.savefig("./pictures/hist.png")


from sklearn.linear_model import LinearRegression 
#=================================================================================
# Picture 3
# =========
plt.figure(figsize=(6, 6))
plt.tick_params(axis='both', which='major', labelsize=13)
xticks = range(0, 8, 1)
plt.scatter(predict, origin, marker="o")
a = b = range(0, 8, 1)
plt.plot(a, b, 'b', label="Diagonal")
reg = LinearRegression().fit(predict, origin)
plt.plot(predict, reg.predict(predict), color='red', label="predict")
plt.xlim(2, 7)
plt.ylim(2, 7)
plt.xlabel("Simulated Spring discharge (m$^3$/s)", fontsize=20)
plt.ylabel("Observed Spring discharge (m$^3$/s)", fontsize=20)
plt.legend(loc="upper left", shadow=True, fontsize=15)
plt.savefig("./pictures/fit.png")


training_year = 26
#---------------------------------------------------------------------------------
total_month = 32*each_year_month
def plot_curve(true_data, predicted_data):
	plt.figure()
	xticks = range(lag+1, total_month+1, 1)
	plt.tick_params(axis='both',which='major', labelsize=13)
	plt.plot(xticks, true_data, 'purple', label='True data')
	# plt.plot(xticks, predicted_data, 'white')
	plt.plot(range(lag+1, training_year*each_year_month+1, 1), predicted_data[0:training_year*each_year_month-lag], 'b', label='Training data')
	plt.plot(range(training_year*each_year_month+lag, 32*each_year_month+lag, 1), predicted_data[training_year*each_year_month-lag:32*each_year_month-lag], 'r', label='Testing data')
	plt.xlabel("Time (months)", fontsize=20)
	plt.ylabel("Spring discharge (m$^3$/s)", fontsize=20)
	# plt.axvline(lag+1)
	# plt.axvline(26*each_year_month)
	# plt.axvline(32*each_year_month)
	plt.ylim(1.5, 7.5)
	plt.xticks(rotation=30)
	plt.xticks(xticks, ('1984.12', '1985.12', '1986.12', '1987.12', '1988.12','1989.12','1990.12','1991.12','1992.12','1993.12','1994.12','1995.12','1996.12','1997.12','1998.12','1999.12','2000.12','2001.12','2002.12','2003.12','2004.12','2005.12','2006.12','2007.12','2008.12','2009.12','2010.12','2011.12','2012.12','2013.12','2014.12','2015.12','2016.12','2017.12','2018.12', '2019.12', '2020.12'))
	plt.title("the spring discharge trends of true and forecate values")
	ax=plt.gca() 
	ax.xaxis.set_major_locator(MultipleLocator(each_year_month))
	ax.yaxis.set_major_locator(MultipleLocator(1))
	plt.legend(loc='upper right', shadow=True, fontsize=15)
	plt.grid(axis='x', which='minor')
	plt.savefig('./pictures/spring.png')
	plt.show()
plot_curve(origin[:, -1], predict[:, -1])
#---------------------------------------------------------------------------------


print("Code has been executed!!! It takes {:.2f} minutes.".format((time.time() - t)/60))
print("\n  End ... \n")