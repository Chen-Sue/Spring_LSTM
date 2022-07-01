
# cd C:\chengshu\spring\lstm
# python lstm_main.py  

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ##update
import time
import tensorflow as tf
# assert tf.__version__.startswith("2.")  ##update
tf.keras.backend.clear_session()  # For easy reset of notebook state.
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
from tensorflow.keras import layers, optimizers, metrics, Sequential, initializers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.python import pywrap_tensorflow
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import mpl_toolkits.axisartist.axislines as axislines
from tensorflow import keras

print("\n Begin ... \n")

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
output_node = 1
batch_size = 16
epochs = 1000
reset_number = epochs
lr = 1e-4
learning_rate = optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=epochs,
													  decay_rate=0.99, staircase=False)
#---------------------------------------------------------------------------------
lag = 1   # lag = int(input("input lag (1) = ") )
time_series_length = 1 
batch_size = 1  # len(x_train[0])
reset_number = epochs
training_number = 26*12 - lag
#---------------------------------------------------------------------------------


#==================================================================================
# load data
# =========
file_location = r"C:\chengshu\spring\data"
#---------------------------------------------------------------------------------
# 降雨量
input_data1 = pd.read_csv(file_location + r"/降雨量-化乐1.csv", delimiter=",", usecols=[0], header=None)
input_data2 = pd.read_csv(file_location + r"/降雨量-克城1.csv", delimiter=",", usecols=[0], header=None)
input_data3 = pd.read_csv(file_location + r"/降雨量-山头1.csv", delimiter=",", usecols=[0], header=None)
input_data4 = pd.read_csv(file_location + r"/降雨量-一平垣1.csv", delimiter=",", usecols=[0], header=None)
input_data5 = pd.read_csv(file_location + r"/降雨量-台头1.csv", delimiter=",", usecols=[0], header=None)
input_data6 = pd.read_csv(file_location + r"/降雨量-光华1.csv", delimiter=",", usecols=[0], header=None)
input_data7 = pd.read_csv(file_location + r"/降雨量-河底1.csv", delimiter=",", usecols=[0], header=None)
input_data8 = pd.read_csv(file_location + r"/降雨量-双风渰1.csv", delimiter=",", usecols=[0], header=None)
input_data9 = pd.read_csv(file_location + r"/降雨量-关王庙1.csv", delimiter=",", usecols=[0], header=None)
print(input_data1.shape)  # 384, 9
#---------------------------------------------------------------------------------
# 泉流量
output_data = pd.read_csv(file_location + r"/泉流量1.csv", delimiter=",", usecols=[0], header=None)
print(output_data.shape)  # 384, 1
#---------------------------------------------------------------------------------


#=================================================================================
# concat data
# ===========
dataset = pd.concat([output_data, input_data1, input_data2, input_data3, input_data4, input_data5, input_data6, input_data7, input_data8, input_data9], axis=1) # axis=1：数组加在右边
#---------------------------------------------------------------------------------


#=================================================================================
# manually specify column names
# =============================
dataset.columns = ['spring discharge', 'Huale', 'Kecheng', 'Shantou', 'Yipingyuan', 'Taitou', 'Guanghua', 'Hedi', 'Shuangfengyan', 'Guanwangmiao']
print(dataset.head(5))  # summarize first 5 rows
#---------------------------------------------------------------------------------


#=================================================================================
# save to file
# ============
dataset.to_csv(file_location + r'\dataset.csv')
#---------------------------------------------------------------------------------


#=================================================================================
# load dataset
# ============
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

#=================================================================================
# input_data output_data
# ======================
input_data = values[:, :10*lag]  ##update
output_data = values[:, 10*lag].reshape(-1, 1)  ##update
#---------------------------------------------------------------------------------

# #=================================================================================
# # Normalized data
# # ===============
# scaler_x = MinMaxScaler().fit(input_data)   # MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
# input_data = scaler_x.transform(input_data).reshape(-1, input_data.shape[1])
# scaler_y = MinMaxScaler().fit(output_data)
# output_data = scaler_y.transform(output_data).reshape(-1, 1)
# #---------------------------------------------------------------------------------
# print("reshape: input_data.shape={0}, output_data.shape={1}".format(input_data.shape, output_data.shape))
# #---------------------------------------------------------------------------------

#=================================================================================
# split into input and outputs
# ============================
x_train, y_train = input_data[:training_number, :], output_data[:training_number, ].reshape(-1, 1)
x_validation, y_validation = input_data[training_number:, :], output_data[training_number:, ].reshape(-1, 1)
# x_test, y_test = test[:, :4*lag], test[:, 4*lag]
x_all, y_all = input_data[:, :], output_data[:, ].reshape(-1, 1)
#---------------------------------------------------------------------------------

n_features = values.shape[1]
print("values = ", values)
#---------------------------------------------------------------------------------

#=================================================================================
# Normalized data
# ===============
scaler_x = MinMaxScaler().fit(x_train)   # MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
x_train = scaler_x.transform(x_train).reshape(-1, x_train.shape[1])
x_validation = scaler_x.transform(x_validation).reshape(-1, x_validation.shape[1])
scaler_y = MinMaxScaler().fit(y_train)   # MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
y_train = scaler_y.transform(y_train).reshape(-1, y_train.shape[1])
y_validation = scaler_y.transform(y_validation).reshape(-1, y_validation.shape[1])
#---------------------------------------------------------------------------------
print("reshape: input_data.shape={0}, output_data.shape={1}".format(input_data.shape, output_data.shape))
#---------------------------------------------------------------------------------


#=================================================================================
# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], time_series_length, x_train.shape[1]))
x_validation = x_validation.reshape((x_validation.shape[0], time_series_length, x_validation.shape[1]))
# x_test = x_test.reshape((x_test.shape[0], time_series_length, x_test.shape[1]))
#---------------------------------------------------------------------------------
# reshape output to be 2D [samples, features]
y_train = y_train.reshape(y_train.shape[0], 1)
y_validation = y_validation.reshape(y_validation.shape[0], 1)
# y_test = y_test.reshape(y_test.shape[0], 1)
#---------------------------------------------------------------------------------
print("shape: \n  x_train.shape={0}, y_train.shape{1}, \n  x_validation.shape={2}, y_validation.shape={3}"\
    .format(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape))
#---------------------------------------------------------------------------------


#=================================================================================
# design network
# ==============
model = Sequential()
# stateful lstm
# model.add(layers.LSTM(256, batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]), \
# 	stateful=True, return_sequences=False))
# model.add(layers.LSTM(32, stateful=True, return_sequences=True)) # False
# model.add(layers.LSTM(32, stateful=True, return_sequences=False)) # False
#---------------------------------------------------------------------------------
# stateless lstm
model.add(layers.LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=False))
# model.add(layers.LSTM(128, return_sequences=False)) 
# model.add(Dense(units=32, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(),
# 				bias_initializer=tf.zeros_initializer(), name="layer3"))
# model.add(Dense(units=32, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(),
# 				bias_initializer=tf.zeros_initializer(), name="layer4"))
# model.add(layers.Dropout(0.5))
# model.add(layers.LSTM(output_node, stateful=True, return_sequences=True)) 
model.add(layers.Dense(units=output_node, kernel_initializer=initializers.glorot_normal(),
				bias_initializer=tf.zeros_initializer(), name="layer7"))
# model.add(LSTM(n_hidden, return_sequences=True, activation='tanh',
#                recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
#                recurrent_initializer='orthogonal', bias_initializer='zeros',
#                dropout=rate_dropout, recurrent_dropout=rate_dropout))
#---------------------------------------------------------------------------------
model.build(input_shape=[None, x_train.shape[1]])
print("model.summary(): \n{} ".format(model.summary()))
print("layer nums:", len(model.layers))
#---------------------------------------------------------------------------------


#=================================================================================
# Optimizer
# =========
optimizer = optimizers.Adam(learning_rate=learning_rate)
#---------------------------------------------------------------------------------


#=================================================================================
# checkpoint
# ==========
checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
#---------------------------------------------------------------------------------


#=================================================================================
# Compile
# =======
model.compile(optimizer=optimizer, loss="mean_squared_error",
			  metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")])
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
log_dir = os.path.join("log")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
error_dir = os.path.join("error")
if not os.path.exists(error_dir):
    os.mkdir(error_dir)
model_dir = os.path.join("model")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
weight_dir = os.path.join("weight")
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)
#---------------------------------------------------------------------------------
# output_model_file = os.path.join("./model/the_best_model.h5")
output_model_file = os.path.join("./model/{epoch:05d}-{loss:.6f}-{mae:.6f}-{rmse:.6f}-{val_loss:.6f}-{val_mae:.6f}-{val_rmse:.6f}.h5")
#---------------------------------------------------------------------------------
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir),
				tf.keras.callbacks.CSVLogger(os.path.join(log_dir,'logs.log'),separator=','),
				tf.keras.callbacks.ModelCheckpoint(output_model_file, monitor='val_loss', verbose=1, save_best_only = True, save_weights_only=False),
				tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs, verbose=True)]
#---------------------------------------------------------------------------------


#=================================================================================
# Train on training set
# =====================

# for i in range(reset_number):
#     	print("steps: ", i)
# 	global history
# 	history = model.fit(x_train, y_train, epochs=1, validation_data=(x_validation, y_validation), \
# 						batch_size=1, verbose=2, shuffle=False, callbacks=callbacks)
# 	model.reset_states()
#---------------------------------------------------------------------------------
# batch_size = len(x_train[0])
history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_validation, y_validation), \
						batch_size=batch_size, verbose=2, shuffle=False, callbacks=callbacks)
#---------------------------------------------------------------------------------


#==================================================================================
# save model
# ==========
# model.save("./model/the_save_model.h5")
# checkpoint.restore(tf.train.latest_checkpoint('./model'))
#--------------------------------------------------------------------------------


#=================================================================================
# Print weights
# =============
save_path = model.save_weights('./model/model.ckpt')
model_dir = './model/'
checkpoint_file = os.path.join(model_dir, "model.ckpt")
model.load_weights(checkpoint_file)
model_reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_file)  
var_dict = model_reader.get_variable_to_shape_map() 
#---------------------------------------------------------------------------------


#=================================================================================
# history.pkl
# ======================
file = open("./model/history.pkl", "wb")
history_dict = history.history
print("history_dict.keys(): {}".format(history_dict.keys()))
# print("history dict: \n{}".format(history_dict))
pickle.dump(history_dict, file)
file.close()
#--------------------------------------------------------------------------------


#=================================================================================
# save error
# ==========
loss = history_dict["loss"]
mae = history_dict["mae"]
rmse = history_dict["rmse"]
#---------------------------------------------------------------------------------
val_loss = history_dict["val_loss"]
val_mae = history_dict["val_mae"]
val_rmse = history_dict["val_rmse"]
#---------------------------------------------------------------------------------	
np.savetxt(r"./error/train.csv", np.row_stack((np.array(loss), np.array(mae), np.array(rmse))).T, delimiter=",", newline='\n', header='train')
np.savetxt(r"./error/train.csv", np.row_stack((np.array(val_loss), np.array(val_mae), np.array(val_rmse))).T, delimiter=",", newline='\n', header='train')
#---------------------------------------------------------------------------------


print("Code has been executed!!! It takes {:.2f} minutes.".format((time.time() - t)/60))