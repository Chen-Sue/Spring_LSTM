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
epochs = 10000
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
file_location = r"C:\cshu\spring\data"
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
# reshape input to be 2D [samples, features]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
x_validation = x_validation.reshape((x_validation.shape[0], x_validation.shape[1]))
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))
#---------------------------------------------------------------------------------
# reshape output to be 2D [samples, features]
y_train = y_train.reshape(y_train.shape[0], 1)
y_validation = y_validation.reshape(y_validation.shape[0], 1)
# y_test = y_test.reshape(y_test.shape[0], 1)
#---------------------------------------------------------------------------------
print("shape: \n  x_train.shape={0}, y_train.shape{1}, \n  x_validation.shape={2}, y_validation.shape={3}"\
    .format(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape))
#---------------------------------------------------------------------------------

x_test = x_validation
y_test = y_validation

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# 4.1 支持向量机模型进行学习和预测
# 线性核函数配置支持向量机
linear_svr = SVR(kernel="rbf")
# 训练
linear_svr.fit(x_train, y_train)
# 预测 保存预测结果
linear_svr_y_predict = linear_svr.predict(x_test)
linear_svr_y_predict_train = linear_svr.predict(x_train)


# plt.errorbar(np.arange(26*12, 32*12), y_test, yerr=standard_error, fmt='o:',ecolor='hotpink',elinewidth=3,ms=5,mfc='wheat',mec='salmon',capsize=3)
# plt.show()

# import skill_metrics as sm
# import scipy.stats as stats
# from sklearn.metrics import mean_squared_error
# sdev = np.std(y_test,ddof=0)
# crmsd = np.sqrt(mean_squared_error(scaler_y.inverse_transform(linear_svr.predict(x_test).reshape(-1, 1)),scaler_y.inverse_transform(y_test.reshape(-1, 1)))) # centered root-mean-square error deviation
# x1 = scaler_y.inverse_transform(linear_svr.predict(x_test).reshape(-1, 1))
# x2 = scaler_y.inverse_transform(y_test.reshape(-1, 1))
# print(len(x1), len(x2))
# print(x1, x2)
# x1 = [i for item in x1 for i in item] # list(_flatten(a))
# x2 = [i for item in x2 for i in item]
# ccoef = stats.pearsonr(x1, x2)[0]
# print(ccoef)
# # ccoef = pd.Series(x1).corr(pd.Series(x2),method="pearson") # Correlation coefficient
# sm.taylor_diagram(sdev, crmsd, ccoef, colRMS='g', styleRMS=':', widthRMS=2.0, titleRMS='on',\
# 	colSTD='b', styleSTD='-.', widthSTD=1.0, titleSTD ='on',\
# 		colCOR='k', styleCOR='--', widthCOR=1.0, titleCOR='on')
# label = {'ERA-5': 'r', 'TRMM': 'b'}
# '''
# Produce the Taylor diagram for the first dataset
# '''
# sm.taylor_diagram(0.8, 
# 					0.5, 
# 					0.6, markercolor ='r', alpha = 0.0,
# 					titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0])

# '''
# Overlay the second dataset
# '''
# sm.taylor_diagram(0.4, 
# 					0.5, 
# 					0.7, markercolor ='b', alpha = 0.0,
# 					overlay = 'on', markerLabel = label)
# sm.target_diagram(bias,crmsd,rmsd, MarkerDisplayed = 'colorBar',
#                       titleColorbar = 'RMSD', cmapzdata = rmsd,
#                       markerLabel = label, markerLabelColor = 'b',
#                       markerColor = 'b', markerLegend = 'on',
#                       ticks = np.arange(-50,60,10), axismax = 50,
#                       xtickLabelPos = [np.arange(-50,20,20), 40, 50],
#                       circles = [20, 40, 50],
#                       circleLineSpec = '-.b', circleLineWidth = 1.5)
# plt.show()

ss_y = StandardScaler()
# 5 模型评估
# 线性核函数 模型评估
print("RBF核函数支持向量机的默认评估值为：", linear_svr.score(x_test, y_test))
print("RBF核函数支持向量机的R_squared值为：", r2_score(y_test, linear_svr_y_predict))
print("RBF核函数支持向量机的均方误差MSE为:", mean_squared_error(scaler_y.inverse_transform(y_train.reshape(-1, 1)),
                                              scaler_y.inverse_transform(linear_svr_y_predict_train.reshape(-1, 1))))
print("RBF核函数支持向量机的平均绝对误差MAE为:", mean_absolute_error(scaler_y.inverse_transform(y_train.reshape(-1, 1)),
                                                 scaler_y.inverse_transform(linear_svr_y_predict_train.reshape(-1, 1))))
print("RBF核函数支持向量机的均方根误差RMSE为:", np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_train.reshape(-1, 1)),
                                              scaler_y.inverse_transform(linear_svr_y_predict_train.reshape(-1, 1)))))

print("RBF核函数支持向量机的均方误差MSE为:", mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)),
                                              scaler_y.inverse_transform(linear_svr_y_predict.reshape(-1, 1))))
print("RBF核函数支持向量机的平均绝对误差MAE为:", mean_absolute_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)),
                                                 scaler_y.inverse_transform(linear_svr_y_predict.reshape(-1, 1))))
print("RBF核函数支持向量机的均方根误差RMSE为:", np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)),
                                              scaler_y.inverse_transform(linear_svr_y_predict.reshape(-1, 1)))))



# #=================================================================================
# # design network
# # ==============
# model = Sequential()
# model.add(Dense(units=256, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(),
# 				bias_initializer=tf.zeros_initializer(), name="layer1"))
# # model.add(BatchNormalization())
# # model.add(LeakyReLU())
# model.add(Dense(units=128, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(),
# 				bias_initializer=tf.zeros_initializer(), name="layer2"))
# # model.add(BatchNormalization())
# # model.add(LeakyReLU())
# model.add(Dense(units=64, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(),
# 				bias_initializer=tf.zeros_initializer(), name="layer3"))
# # model.add(BatchNormalization())
# # model.add(LeakyReLU())
# model.add(Dense(units=32, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(),
# 				bias_initializer=tf.zeros_initializer(), name="layer4"))
# # model.add(BatchNormalization())
# # model.add(LeakyReLU())
# model.add(Dense(units=16, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(),
# 				bias_initializer=tf.zeros_initializer(), name="layer5"))
# # model.add(BatchNormalization())
# # model.add(LeakyReLU())
# model.add(Dense(units=8, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(),
# 				bias_initializer=tf.zeros_initializer(), name="layer6"))
# # model.add(BatchNormalization())
# # model.add(LeakyReLU())
# model.add(Dense(units=output_node, kernel_initializer=initializers.glorot_normal(),
# 				bias_initializer=tf.zeros_initializer(), name="layer7"))
# #---------------------------------------------------------------------------------
# model.build(input_shape=[None, x_train.shape[1]])
# print("model.summary(): \n{} ".format(model.summary()))
# print("layer nums:", len(model.layers))
# #---------------------------------------------------------------------------------

# #=================================================================================
# # Optimizer
# # =========
# optimizer = optimizers.Adam(learning_rate=learning_rate)
# #---------------------------------------------------------------------------------


# #=================================================================================
# # checkpoint
# # ==========
# checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
# #---------------------------------------------------------------------------------


# #=================================================================================
# # Compile
# # =======
# model.compile(optimizer=optimizer, loss="mean_squared_error",
# 			  metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")])
# #---------------------------------------------------------------------------------


# #---------------------------------------------------------------------------------
# log_dir = os.path.join("log")
# if not os.path.exists(log_dir):
#     os.mkdir(log_dir)
# error_dir = os.path.join("error")
# if not os.path.exists(error_dir):
#     os.mkdir(error_dir)
# model_dir = os.path.join("model")
# if not os.path.exists(model_dir):
#     os.mkdir(model_dir)
# weight_dir = os.path.join("weight")
# if not os.path.exists(weight_dir):
#     os.mkdir(weight_dir)
# #---------------------------------------------------------------------------------
# output_model_file = os.path.join("./model/model_{epoch:05d}-{loss:.6f}-{mae:.6f}-{rmse:.6f}-{val_loss:.6f}-{val_mae:.6f}-{val_rmse:.6f}.h5")
# #---------------------------------------------------------------------------------
# callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir),
# 				tf.keras.callbacks.CSVLogger(os.path.join(log_dir,'logs.log'),separator=','),
# 				tf.keras.callbacks.ModelCheckpoint(output_model_file, monitor='val_loss', verbose=1, save_best_only = True, save_weights_only=False),
# 				tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs, verbose=True)]
# #---------------------------------------------------------------------------------


# #=================================================================================
# # Train on training set
# # =====================
# history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_validation, y_validation), \
# 						batch_size=batch_size, verbose=2, shuffle=False, callbacks=callbacks)
# #---------------------------------------------------------------------------------


# #==================================================================================
# # save model
# # ==========
# # model.save("./model/the_save_model.h5")
# # checkpoint.restore(tf.train.latest_checkpoint('./model'))
# #--------------------------------------------------------------------------------


# #=================================================================================
# # Print weights
# # =============
# save_path = model.save_weights('./model/model.ckpt')
# model_dir = './model/'
# checkpoint_file = os.path.join(model_dir, "model.ckpt")
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


# #=================================================================================
# # history.pkl
# # ======================
# file = open("./history.pkl", "wb")
# history_dict = history.history
# print("history_dict.keys(): {}".format(history_dict.keys()))
# # print("history dict: \n{}".format(history_dict))
# pickle.dump(history_dict, file)
# file.close()
# #--------------------------------------------------------------------------------


# #=================================================================================
# # save error
# # ==========
# loss = history_dict["loss"]
# mae = history_dict["mae"]
# rmse = history_dict["rmse"]
# #---------------------------------------------------------------------------------
# val_loss = history_dict["val_loss"]
# val_mae = history_dict["val_mae"]
# val_rmse = history_dict["val_rmse"]
# #---------------------------------------------------------------------------------	
# np.savetxt(r"./error/train.csv", np.row_stack((np.array(loss), np.array(mae), np.array(rmse))).T, delimiter=",", newline='\n', header='train')
# np.savetxt(r"./error/train.csv", np.row_stack((np.array(val_loss), np.array(val_mae), np.array(val_rmse))).T, delimiter=",", newline='\n', header='train')
# #---------------------------------------------------------------------------------


# #=================================================================================
# # do the train prediction
# # =======================
# y_train_predicted = model.predict(x_train)
# x_train = np.squeeze(x_train) 
# train_predict_temp = np.concatenate((y_train_predicted, x_train[:, -n_features:-1]), axis=1)
# predict_train = scaler.inverse_transform(train_predict_temp)
# predict_train = predict_train[:, 0]
# # do the train data
# train_origin_temp = np.concatenate((y_train, x_train[:, -n_features:-1]), axis=1)
# origin_train = scaler.inverse_transform(train_origin_temp)
# origin_train = origin_train[:, 0]
# #---------------------------------------------------------------------------------

# #=================================================================================
# # do the validation prediction
# # =======================
# y_validation_predicted = model.predict(x_validation)
# x_validation = np.squeeze(x_validation) 
# validation_predict_temp = np.concatenate((y_validation_predicted, x_validation[:, -n_features:-1]), axis=1)
# predict_validation = scaler.inverse_transform(validation_predict_temp)
# predict_validation = predict_validation[:, 0]
# # do the validation data
# validation_origin_temp = np.concatenate((y_validation, x_validation[:, -n_features:-1]), axis=1)
# origin_validation = scaler.inverse_transform(validation_origin_temp)
# origin_validation = origin_validation[:, 0]
# #---------------------------------------------------------------------------------


# #=================================================================================
# # do the new prediction
# # =====================
# x_new = x_validation[-1][:].reshape(1, x_validation.shape[1])
# y_new_predicted = model.predict(x_new)
# x_new = x_new.reshape(1, x_validation.shape[1])
# print("y_new_predicted, x_new",y_new_predicted.shape, x_new.shape)
# # do the new data
# y_new_predicted = np.concatenate((y_new_predicted, x_new[:, -n_features:-1]), axis=1)
# predict_new = scaler.inverse_transform(y_new_predicted)
# print("predict_new ", predict_new)
# #---------------------------------------------------------------------------------

# #=================================================================================
# # do the difference
# # =====================
# origin = np.append(origin_train, origin_validation)
# predict = np.append(predict_train, predict_validation)
# origin = origin.reshape(-1, 1)
# predict = predict.reshape(-1, 1)
# diff = predict - origin
# #---------------------------------------------------------------------------------


# #=================================================================================
# # Picture 1
# # =========
# fig = plt.figure()
# xticks = range(1, epochs+1, 1)
# #---------------------------------------------------------------------------------
# # subpicture1
# ax1 = axislines.Subplot(fig, 212)
# fig.add_subplot(ax1)
# ax1.axis[:].major_ticks.set_tick_out(True)
# ax1.set_xticks(xticks)
# ax1.axis["bottom"].label.set_text("Epochs")
# ax1.axis["left"].label.set_text("error") 
# ax1.set_title("lag={} MAE".format(lag))
# mae = history_dict["mae"]
# val_mae = history_dict["val_mae"]
# ax1.plot(xticks, mae, "b", label="Training")
# ax1.plot(xticks, val_mae, "r", label="Validation")
# # ax1=plt.gca() #ax为两条坐标轴的实例
# ax1.xaxis.set_major_locator(MultipleLocator(epochs/100)) #把x轴的主刻度设置为epochs/10的倍数
# ax1.yaxis.set_major_locator(MultipleLocator(0.05)) #把y轴的主刻度设置为0.05的倍数
# ax1.set_ylim(0, 0.20)
# ax1.legend(loc="upper right", shadow=True)
# #---------------------------------------------------------------------------------
# # subpicture2
# ax2 = axislines.Subplot(fig, 211)
# fig.add_subplot(ax2)
# ax2.axis[:].major_ticks.set_tick_out(True)
# ax2.set_xticks(xticks)
# ax2.axis["bottom"].label.set_text("Epochs")
# ax2.axis["left"].label.set_text("Error") 
# ax2.set_title("lag={} RMSE".format(lag))
# rmse = history_dict["rmse"]
# val_rmse = history_dict["val_rmse"]
# ax2.plot(xticks, rmse, "b", label="Training")
# ax2.plot(xticks, val_rmse, "r", label="Validation")
# # ax2=plt.gca() #ax为两条坐标轴的实例
# ax2.xaxis.set_major_locator(MultipleLocator(epochs/10)) #把x轴的主刻度设置为epochs/10的倍数
# ax2.yaxis.set_major_locator(MultipleLocator(0.05)) #把y轴的主刻度设置为0.05的倍数
# ax2.set_ylim(0, 0.20)
# ax2.legend(loc="upper right", shadow=True)
# #---------------------------------------------------------------------------------
# plt.savefig("./error.png")
# fig.show()
# #---------------------------------------------------------------------------------

# #=================================================================================
# # Picture 2
# # =========
# plt.figure()
# plt.tick_params(axis='both',which='major',labelsize=11)
# # plt.axis[:].major_ticks.set_tick_out(True)
# # n, bins, patches = plt.hist(diff, bins=10, density=True, cumulative=False, facecolor="yellowgreen", alpha=0.5, label="histogram")
# # lag=3
# n, bins, patches = plt.hist(diff, bins=6, range=[-0.30, 0.30], density=True, cumulative=False, facecolor="yellowgreen", alpha=0.5, label="histogram")
# # y = norm.pdf(bins, diff.mean(), diff.std()) # 生成正态分布函数
# # plt.plot(bins, y, "b--", label="Gauss Distribution")
# plt.title("lag={0} Histogram: $\mu = {1:.2f}$, $\sigma={2:.2f}$".format(lag, diff.mean(), diff.std()), fontsize=18)
# plt.xlabel("Value of Spring discharge diffrence", fontsize=16)
# plt.ylabel("Percent", fontsize=16)
# # plt.yaxis.set_major_formatter(PercentFormatter(xmax=len(diff)))
# plt.legend(loc="upper right", shadow=True)
# plt.savefig("./hist.png")
# plt.show()

# from sklearn.linear_model import LinearRegression 
# #=================================================================================
# # Picture 3
# # =========
# plt.figure(figsize=(6, 6))
# plt.tick_params(axis='both', which='major', labelsize=13)
# xticks = range(0, 8, 1)
# plt.scatter(predict, origin, marker="o")
# a = b = range(0, 8, 1)
# plt.plot(a, b, 'b', label="Diagonal")
# reg = LinearRegression().fit(predict, origin)
# plt.plot(predict, reg.predict(predict), color='red', label="predict")
# plt.xlim(2, 7)
# plt.ylim(2, 7)
# plt.xlabel("Simulated Spring discharge (m$^3$/s)", fontsize=20)
# plt.ylabel("Observed Spring discharge (m$^3$/s)", fontsize=20)
# plt.legend(loc="upper left", shadow=True, fontsize=15)
# plt.savefig("./fit.png")
# plt.show()

# # training_year = 24
# # validating_year = 6
# # testing_year = 2
# # n_train_mouth = (each_year_month+validating_year)*each_year_month
# #---------------------------------------------------------------------------------
# total_month = 32*each_year_month
# def plot_curve(true_data, predicted_data):
# 	plt.figure()
# 	xticks = range(lag+1, total_month+1, 1)
# 	plt.tick_params(axis='both',which='major', labelsize=13)
# 	plt.plot(xticks, true_data, 'purple', label='True data')
# 	plt.plot(xticks, predicted_data, 'white')
# 	plt.plot(range(lag+1, training_year*each_year_month+1, 1), predicted_data[0:training_year*each_year_month-1], 'b', label='Training data')
# 	plt.plot(range(training_year*each_year_month, (training_year+validating_year)*each_year_month+1, 1), predicted_data[training_year*each_year_month-2:(training_year+validating_year)*each_year_month-1], 'g', label='Validation data')
# 	plt.plot(range((training_year+validating_year)*each_year_month, 32*each_year_month+1, 1), predicted_data[(training_year+validating_year)*each_year_month-2:32*each_year_month-1], 'r', label='Testing data')
# 	plt.xlabel("Time (months)", fontsize=20)
# 	plt.ylabel("Spring discharge (m$^3$/s)", fontsize=20)
# 	# plt.scatter(xticks, predict_new, c="y", marker="o") #画点
# 	# plt.scatter(xticks, predicted_data, c="r", marker="^") #画点
# 	plt.scatter(total_month+1, predict_new[0][0], c="r", marker="^")
# 	# plt.axvline(lag+1)
# 	# plt.axvline(24*each_year_month)
# 	# plt.axvline(30*each_year_month)
# 	# plt.axvline(32*each_year_month)
# 	# plt.xlim(0, 385)
# 	plt.ylim(1.5, 7.5)
# 	plt.xticks(rotation=30)
# 	plt.xticks(xticks, ('1984.12', '1985.12', '1986.12', '1987.12', '1988.12','1989.12','1990.12','1991.12','1992.12','1993.12','1994.12','1995.12','1996.12','1997.12','1998.12','1999.12','2000.12','2001.12','2002.12','2003.12','2004.12','2005.12','2006.12','2007.12','2008.12','2009.12','2010.12','2011.12','2012.12','2013.12','2014.12','2015.12','2016.12','2017.12','2018.12', '2019.12', '2020.12'))
#     # plt.title("the spring discharge trends of true and forecate values")
# 	x_major_locator=MultipleLocator(each_year_month) 
# 	y_major_locator=MultipleLocator(1) 
# 	ax=plt.gca() #ax为两条坐标轴的实例
# 	ax.xaxis.set_major_locator(x_major_locator) #把x轴的主刻度设置为each_year_month的倍数
# 	ax.yaxis.set_major_locator(y_major_locator) #把y轴的主刻度设置为1的倍数
# 	plt.legend(loc='upper right', shadow=True, fontsize=15)
# 	plt.grid(axis='x', which='minor')
# 	plt.savefig('./spring.png')
# 	plt.show()
# plot_curve(origin[:, -1], predict[:, -1])
# #---------------------------------------------------------------------------------

# #---------------------------------------------------------------------------------
# print("Code has been executed!!! It takes {:.2f} minutes.".format((time.time() - t)/60))
# #---------------------------------------------------------------------------------
