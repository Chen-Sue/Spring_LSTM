
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # 忽略tensorflow警告信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 只使用第一块GPU
os.environ["PATH"] += os.pathsep + r"C:\Users\cs\Anaconda3\Library\bin\graphviz"
import time
import tensorflow as tf
import numpy as np 
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import mpl_toolkits.axisartist.axislines as axislines
from scipy.stats import norm
from six.moves import cPickle 
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
from tensorflow import keras
from tensorflow.keras import layers, optimizers, metrics, Sequential, initializers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, LeakyReLU, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from sklearn.utils import shuffle
from datetime import datetime

print("\n******")
print("  Begin to execute main_mean.py... \n")

#=================================================================================
# Environment
# ===========
t = time.time()
seed = 22
tf.random.set_seed(seed)  
np.random.seed(seed) 
tf.keras.backend.clear_session()
#---------------------------------------------------------------------------------


file_location = r"C:\chengshu\Person\spring\data"

# 降雨量
input_data1 = pd.read_csv(file_location + r"\降雨量-化乐1.csv", delimiter=",", usecols=[0], header=None)
input_data2 = pd.read_csv(file_location + r"\降雨量-克城1.csv", delimiter=",", usecols=[0], header=None)
input_data3 = pd.read_csv(file_location + r"\降雨量-山头1.csv", delimiter=",", usecols=[0], header=None)
input_data4 = pd.read_csv(file_location + r"\降雨量-一平垣1.csv", delimiter=",", usecols=[0], header=None)
input_data5 = pd.read_csv(file_location + r"\降雨量-台头1.csv", delimiter=",", usecols=[0], header=None)
input_data6 = pd.read_csv(file_location + r"\降雨量-光华1.csv", delimiter=",", usecols=[0], header=None)
input_data7 = pd.read_csv(file_location + r"\降雨量-河底1.csv", delimiter=",", usecols=[0], header=None)
input_data8 = pd.read_csv(file_location + r"\降雨量-双风渰1.csv", delimiter=",", usecols=[0], header=None)
input_data9 = pd.read_csv(file_location + r"\降雨量-关王庙1.csv", delimiter=",", usecols=[0], header=None)
print(input_data1.shape) # 384, 9
input_data1 = np.array(input_data1)  # shape=(384, 9)
input_data2 = np.array(input_data2)
input_data3 = np.array(input_data3)
input_data4 = np.array(input_data4)
input_data5 = np.array(input_data5)
input_data6 = np.array(input_data6)
input_data7 = np.array(input_data7)
input_data8 = np.array(input_data8)
input_data9 = np.array(input_data9)

# 泉流量
output_data = pd.read_csv(file_location + r"\泉流量1.csv", delimiter=",", usecols=[0], header=None)
print(output_data.shape) # 384, 1
output_data = np.array(output_data)

# 滞后
x, z1, z2, z3, z4, z5, z6, z7, z8, z9, y = [], [], [], [], [], [], [], [], [], [], []
# 若泉流量滞后 lag_spring 个月 
# lag_spring = 6 
lag_spring = input("input spring lag = ") 
for i in range(384-lag_spring):
    y = np.append(y, output_data[i+lag_spring], axis=0) 
    for j in range(lag_spring):
        x = np.append(x, output_data[i+j], axis=0)   # axis=0：数组加在下面
# 若降雨量滞后 lag_precipitation 个月 
# lag_precipitation = 6 
lag_precipitation = input("input precipitation lag = ") 
for i in range(384-lag_precipitation):
    for j in range(lag_precipitation):
        z1 = np.append(z1, input_data1[i+j], axis=0)
        z2 = np.append(z2, input_data2[i+j], axis=0)
        z3 = np.append(z3, input_data3[i+j], axis=0)
        z4 = np.append(z4, input_data4[i+j], axis=0)
        z5 = np.append(z5, input_data5[i+j], axis=0)
        z6 = np.append(z6, input_data6[i+j], axis=0)
        z7 = np.append(z7, input_data7[i+j], axis=0)
        z8 = np.append(z8, input_data8[i+j], axis=0)
        z9 = np.append(z9, input_data9[i+j], axis=0)

x = x.reshape(-1, lag_spring)
z1 = z1.reshape(-1, lag_precipitation)
z2 = z2.reshape(-1, lag_precipitation)
z3 = z3.reshape(-1, lag_precipitation)
z4 = z4.reshape(-1, lag_precipitation)
z5 = z5.reshape(-1, lag_precipitation)
z6 = z6.reshape(-1, lag_precipitation)
z7 = z7.reshape(-1, lag_precipitation)
z8 = z8.reshape(-1, lag_precipitation)
z9 = z9.reshape(-1, lag_precipitation)
y = y.reshape(-1, 1)
print("x.shape=", x.shape, "z.shape=", z1.shape, "y.shape=", y.shape)

x = np.concatenate((x, z1, z2, z3, z4, z5, z6, z7, z8, z9), axis=1)
print("x.shape=", x.shape)


#=================================================================================
# Parameters
# ==========
input_node = lag_spring*1 + lag_precipitation*9  
output_node = 1
batch_size = 64
epochs = 10000
initial_learning_rate = 1e-3
learning_rate = optimizers.schedules.ExponentialDecay(initial_learning_rate,
													decay_steps=10000,
													decay_rate=0.96,
													staircase=False)										
#---------------------------------------------------------------------------------

# 打乱数据集
randnum = 1000
np.random.seed(randnum)
input_data = np.array(shuffle(list(x)))
np.random.seed(randnum)
output_data = np.array(shuffle(list(y)))

input_data = np.array(input_data, dtype=np.float32).reshape(-1, l)
output_data = np.array(output_data, dtype=np.float32).reshape(-1, 1)
#=================================================================================
# Normalized data
# ===============d
scaler_x = MinMaxScaler().fit(input_data)   # MaxAbsScaler
input_data = scaler_x.transform(input_data)
#---------------------------------------------------------------------------------
scaler_y = MinMaxScaler().fit(output_data)  # MinMaxScaler
output_data = scaler_y.transform(output_data)
#---------------------------------------------------------------------------------
print("******")
print("  Normalized shape: input_data={0}, output_data={1}".format(input_data.shape, output_data.shape))
input_data = input_data.reshape(-1, l*1+l*9)

num1 = int(378 * 0.7)
x_train, x_test = input_data[:num1, :], input_data[num1:, :]
y_train, y_test = output_data[:num1], output_data[num1:]
print("******")
print("  Spliting shape: x_train={0}, y_train={1}".format(x_train.shape, y_train.shape))
# print("  Spliting shape: x_validation={0}, y_validation={1}".format(x_validation.shape, y_validation.shape))
print("  Spliting shape: x_test={0}, y_test={1}".format(x_test.shape, y_test.shape))

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)  
    y = tf.cast(y, dtype=tf.float32) 
    return x, y
#---------------------------------------------------------------------------------
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(preprocess).batch(batch_size)
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
test_dataset =  tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(preprocess).batch(batch_size)
#---------------------------------------------------------------------------------

model = Sequential()

model.add(Dense(256, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer()))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(128, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer()))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(64, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer()))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(output_node, kernel_initializer=initializers.glorot_normal(), bias_initializer=tf.zeros_initializer()))
model.add(Activation("relu"))

#---------------------------------------------------------------------------------
model.build(input_shape=[None, x_train.shape[1]])
#---------------------------------------------------------------------------------
print("******************************************************************")
print("model.summary(): \n{} ".format(model.summary()))
print("  layer nums:", len(model.layers))
print("******************************************************************")
#---------------------------------------------------------------------------------


#=================================================================================
# Checkpoint
# ==========
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", \
	verbose=1, save_best_only=True, mode="auto", save_freq=1)
reducelr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, \
	patience=5, verbose=1, mode="auto", cooldown=0, min_lr=0)
callback_lists = [checkpoint, reducelr]
#---------------------------------------------------------------------------------


#=================================================================================
# Optimizer
# =========
optimizer = optimizers.Adam(learning_rate=learning_rate)
#---------------------------------------------------------------------------------


#=================================================================================
# Compile
# =======
model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
#---------------------------------------------------------------------------------


#=================================================================================
# Train on training set
# =====================
print("******************************************************************")
history = model.fit(x_train, y_train, epochs=epochs,  \
	verbose=2, validation_data=(x_test, y_test)) #, callbacks=callback_lists)
# history = model.fit(train_dataset, epochs=epochs, \
# 	validation_data=validation_dataset, verbose=2)
print("******************************************************************")
#---------------------------------------------------------------------------------


#=================================================================================
# history of the loss values and metric values during training
# ======================
file = open("./history.pkl", "wb")
history_dict = history.history
print("******")
print("history_dict.keys(): {}".format(history_dict.keys()))
print("history dict: \n{}".format(history_dict))
print("******")
pickle.dump(history_dict, file)
file.close()
#---------------------------------------------------------------------------------


#=================================================================================
# Evaluate on test set
# ====================
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("******")
print("Evaluate on test data: ")
print("  test_mse={0:.4f}%, test_mae={1:.4f}%".\
	format(test_loss*100, test_accuracy*100))
print("******")
#---------------------------------------------------------------------------------


#=================================================================================
# Prediction on new data
# ======================
# x_new_location = r"/mnt/c/chengshu/ShiYaolin/Program"
# x_new = pd.read_csv(x_new_location + r"/Code/x_new.csv", \
# 					delimiter=",", \
# 					usecols=[0, 1, 2, 3, 4], \
# 					header=None)
# x_new = np.array(x_new, dtype="float32")[:, 2:].reshape(1, 82, 3)
# x_new[:, 2] = -x_new[:, 2]
# # x_new[:, 4] = -x_new[:, 4]
# x_new = x_new.reshape(-1, 3)
# # x_new = x_new.reshape(-1, 5)
# x_new_scaler = scaler_x.transform(x_new)
# #---------------------------------------------------------------------------------
# x_new = x_new.reshape(-1, 82*3)
# # x_new = x_new.reshape(-1, 82*5)
# #---------------------------------------------------------------------------------
# y_new = model.predict(x_new)
# print("  y_new={}".format(y_new))
# y_new[:, 0] = 0.02 + (1.5 - 0.02)/2 * (y_new[:, 0]+1)
# y_new[:, 1] = 400 + (400 - (-400))/2 * (y_new[:, 1]+1)
# y_new[:, 2] = 400 + (400 - (-400))/2 * (y_new[:, 2]+1)
# y_new[:, 3] = 0 + (20 - 0)/2 * (y_new[:, 3]+1)
# y_new[:, 4] = 6 + (60 - 6)/2 * (y_new[:, 4]+1)
# y_new[:, 5] = 4 + (20 - 4)/2 * (y_new[:, 5]+1)
# y_new[:, 6] = 0 + (360 - 0)/2 * (y_new[:, 6]+1)
# y_new[:, 7] = 0 + (90 - 0)/2 * (y_new[:, 7]+1)
# y_new[:, 8] = 0 + (360 - 0)/2 * (y_new[:, 8]+1)

# # y_new = scaler_y.inverse_transform(y_new)
# print("******")
# print("Prediction on new data: ")
# print("  y_new={}".format(y_new))
# print("******")
# #---------------------------------------------------------------------------------


#=================================================================================
# Test data
# ======================
y_test_pre = model.predict(x_test)
targets, predictions = scaler_y.inverse_transform(y_test), scaler_y.inverse_transform(y_test_pre)
#---------------------------------------------------------------------------------
diff = (targets - predictions)
print("******")
print("  diff={}".format(diff[:5,:]))
#---------------------------------------------------------------------------------

from matplotlib.ticker import PercentFormatter
from tensorflow.keras.utils import plot_model

# cd C:\chengshu\Person
# python predict.py
#=================================================================================
# Picture
# =======
fig = plt.figure(figsize=(22.0, 15.0))
xticks = range(1, epochs+1, 1)
# yticks = np.arange(0, 0.33, 0.03)
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

# subpicture2
ax2 = axislines.Subplot(fig, 312)
fig.add_subplot(ax2)
ax2.axis[:].major_ticks.set_tick_out(True)
ax2.set_xticks(xticks)
# ax2.set_yticks(yticks)
ax2.axis["bottom"].label.set_text("Epochs")
ax2.axis["left"].label.set_text("Erroe") 
ax2.set_title("mse (Mean square error)")
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
ax2.plot(xticks, loss_values, "--b*", label="Training")
ax2.plot(xticks, val_loss_values, "--r*", label="Validation")
ax2.legend(loc="upper right", shadow=True)
#---------------------------------------------------------------------------------
# subpicture3
ax3 = axislines.Subplot(fig, 313)
fig.add_subplot(ax3)
ax3.axis[:].major_ticks.set_tick_out(True)
ax3.set_xticks(xticks)
# ax3.set_yticks(yticks)
ax3.axis["bottom"].label.set_text("Epochs")
ax3.axis["left"].label.set_text("error") 
ax3.set_title("mae (Mean absolute error)")
acc = history_dict["mae"]
val_acc = history_dict["val_mae"]
ax3.plot(xticks, acc, "--b*", label="Training")
ax3.plot(xticks, val_acc, "--r*", label="Validation")
ax3.legend(loc="upper right", shadow=True)
#---------------------------------------------------------------------------------
plt.savefig("./pictures/self_BP.png")
fig.show()
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
print("Code has been executed!!! It takes {:.2f} minutes.".format((time.time() - t)/60))
#---------------------------------------------------------------------------------