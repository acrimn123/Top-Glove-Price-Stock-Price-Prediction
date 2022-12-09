
#%%
#packages
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.layers import SimpleRNN,LSTM,GRU,Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential,Input
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import os

#%%
#1.Data loading
train_fname = 'Top_Glove_Stock_Price_Train(Modified).csv'
path_name = os.path.join(os.getcwd(),'dataset',train_fname)

df_train = pd.read_csv(path_name)

#%%
#2.Data inspection

#%%
#2.1 Check datatype

df_train.info()

#%%
# 2.2 Check first 10 dataframe

df_train.head(10)

#%%
# 2.3 Describe dataframe
df_train.describe()

#%%
#3.Data cleaning

#%%
#3.1 Check missing values
missing = df_train['Open'].isna().sum()
print(f'Missing values: {missing}')

#3.2 Plot to see missing values
plt.figure(figsize=(10,10))                         
plt.plot(df_train['Open'])
plt.show()

#3.3 Fill the missing values
df_train['Open'] = df_train['Open'].interpolate(method='polynomial',order=3)

#3.4 Plot to see the fill missing values
plt.figure(figsize=(10,10))                         
plt.plot(df_train['Open'])
plt.show()

#3.5 Recheck fill missing values
missing = df_train['Open'].isna().sum()
print(f'Missing values: {missing}')

#%%
#4.Feature selection
#%%
#5.Data preprocessing

#5.1 Set Normalization
scalar = MinMaxScaler()

#5.2 Convert Series to Array
data = df_train['Open'].values
#%%
#5.3 expand dimensions
data = data.reshape(-1, 1)

#5.3 Scaled data
scalar.fit(data)
data = scalar.transform(data)

# %%
#6.Split data to x_train and y_train

#6.1 create window,empty list
window = 60
X_train = []
Y_train = []

#6.2 loop window and append list
for i in range (window,len(data)):
    X_train.append(data[i-window:i])
    Y_train.append(data[i])

#6.3 Convert list to array
X_train = np.array(X_train)
Y_train = np.array(Y_train)

#6.4 train test split
x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,shuffle=True,random_state=12345)

# %%
#7. Buiding model

model = Sequential()
model.add(Input(shape=(window,1)))
model.add(SimpleRNN(8,return_sequences=True))
model.add(GRU(8))
model.add(Dense(1,activation='linear'))

model.summary()
# %%
#%%
#8. Compile model

Log_path = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime('%Y%M%d-%H%M%S'))

es = EarlyStopping(monitor='val_loss',patience=5)
tb = TensorBoard(log_dir=Log_path)

model.compile(optimizer='adam',loss='mse',metrics='mse')
hist = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,callbacks=[es,tb])
# %%
#9.Model evaluation

TEST_CSV_FNAME = 'Top_Glove_Stock_Price_Test.csv'
CSV_PATH = os.path.join(os.getcwd(),'dataset',TEST_CSV_FNAME)

df_test = pd.read_csv(CSV_PATH,names=df_train.columns) 
# %%
#10.Test data concat

concat = pd.concat((df_train['Open'],df_test['Open']),ignore_index=True)
concat = concat[len(concat)-window-len(df_test):]

concat = scalar.transform(concat[::,None])
# %%
#11.Test data concat

x_test_list = []
y_test_list = []

# loop window and append list
for i in range (window,len(concat)):
    x_test_list.append(concat[i-window:i])
    y_test_list.append(concat[i])

X_testtest = np.array(x_test_list)
Y_testtest = np.array(y_test_list)
# %%
#predict model

predicted_test = model.predict(X_testtest)

#plot predictet price using testing

plt.figure(1)
plt.plot(predicted_test,color='r')
plt.plot(Y_testtest,color='b')
plt.legend(['Predicted Price','Actual Price'])
plt.xlabel('time')
plt.ylabel('Stock price')
plt.show()

predicted_test = scalar.inverse_transform(predicted_test)
Y_testtest = scalar.inverse_transform(Y_testtest)

plt.figure(2)
plt.plot(predicted_test,color='r')
plt.plot(Y_testtest,color='b')
plt.legend(['Predicted Price','Actual Price'])
plt.xlabel('time')
plt.ylabel('Stock price')
plt.show()

#metrics to evaluate the performance

from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error

print(mean_absolute_percentage_error(Y_testtest,predicted_test))
print(mean_absolute_error(Y_testtest,predicted_test))


# %%
