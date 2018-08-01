# Multiclass Classification with the NSL-KDD Dataset
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


#%matplotlib inline


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Attck type and Class
Normal = ['normal']
DoS = ['back', 'neptune', 'pod', 'smurf', 'teardrop','land']
Probe = ['ipsweep', 'nmap', 'portsweep','satan']
R2L = ['warezclient','ftp_write','guess_passwd','multihop','phf','spy','warezmaster']
U2R = ['buffer_overflow', 'loadmodule','perl','rootkit']
Unknown = ['unknown']

# load training dataset 
dataframe = pd.read_csv("KDDTrain.csv", header=None)
#dataset = dataframe.values
# The read data informaton
#dataframe.info()
#dataframe.describe()
#  
#dataframe.drop(['PassengerId','Ticket'],axis=1,inplace=True)  

# transfer to Multi-Dimention Array
ndata = np.array(dataframe)

# load test dataset 
dataframe_test =pd.read_csv("KDDTest.csv", header=None)
ndata_test = np.array(dataframe_test)

ndata[:,0]=ndata[:,0].astype(float)
ndata[:,4:41]=ndata[:,4:41].astype(float) 

ndata_test[:,0]=ndata_test[:,0].astype(float)
ndata_test[:,4:41]=ndata_test[:,4:41].astype(float)

Y_label = ndata[:,41]
Y_label_test = ndata_test[:,41]

Out_label=[]
Out_label_test=[]

# transform the string to value

for i in range(1,4):
  encoder = LabelEncoder()
  encoder.fit(ndata[:,i])
  ndata[:,i] = encoder.transform(ndata[:,i])
  
  encoder = LabelEncoder()
  encoder.fit(ndata_test[:,i])
  ndata_test[:,i]  = encoder.transform(ndata_test[:,i])
  ####

# classification of Output
Normal_flag=DoS_flag=Probe_flag=R2L_flag=U2R_flag=Unknown_flag=False

for i in range(0,Y_label.size):
  if Y_label[i] in DoS:
     if DoS_flag is False:
       DoS_flag=True
       Out_label= Out_label+['DoS']
        
          
  elif Y_label[i] in Probe:
     if Probe_flag is False:
       Probe_flag=True
       Out_label= Out_label+['Probe']
 
  elif Y_label[i] in R2L:
     if R2L_flag is False:
       R2L_flag=True
       Out_label= Out_label+['R2L']

  elif Y_label[i] in U2R:  
     if U2R_flag is False:
       U2R_flag=True
       Out_label= Out_label+['U2R']
   
  elif Y_label[i] in Normal:  
     if Normal_flag is False:
       Normal_flag=True
       Out_label= Out_label+['Normal']
  else:
     if Unknown_flag is False:
       Unknown_flag=True
       Out_label= Out_label+['Unknown']
####

# classification of Output
Normal_flag=DoS_flag=Probe_flag=R2L_flag=U2R_flag=Unknown_flag=False

for i in range(0,Y_label_test.size):
  if Y_label_test[i] in DoS:
     if DoS_flag is False:
       DoS_flag=True
       Out_label_test= Out_label_test+['DoS']
        
          
  elif Y_label_test[i] in Probe:
     if Probe_flag is False:
       Probe_flag=True
       Out_label_test= Out_label_test+['Probe']
 
  elif Y_label_test[i] in R2L:
     if R2L_flag is False:
       R2L_flag=True
       Out_label_test= Out_label_test+['R2L']

  elif Y_label_test[i] in U2R:  
     if U2R_flag is False:
       U2R_flag=True
       Out_label_test= Out_label_test+['U2R']

  elif Y_label_test[i] in Normal:  
     if Normal_flag is False:
       Normal_flag=True
       Out_label_test= Out_label_test+['Normal']
   
  else:
       if Unknown_flag is False:
         Unknown_flag=True
         Out_label_test= Out_label_test+['Unknown']

####

## encode class values as integers
encoder = LabelEncoder()
encoder.fit(Out_label)
encoded_Y = encoder.transform(Out_label)
# encoded_Y = [1, 0, 3, 2, 4, 5]

encoder = LabelEncoder()
encoder.fit(Out_label_test)
encoded_Y_test = encoder.transform(Out_label_test)

for i in range(len(Out_label)):
  if Out_label[i] == 'DoS':
    DoS_class=encoded_Y[i]
  elif Out_label[i] == 'Probe':
      Probe_class=encoded_Y[i]
  elif Out_label[i] == 'R2L':
      R2L_class=encoded_Y[i]
  elif Out_label[i] == 'U2R':
      U2R_class=encoded_Y[i] 
  elif Out_label[i] == 'Normal':
      Normal_class=encoded_Y[i]
  else:
      Unknown_class=encoded_Y[i]


for i in range(len(Out_label_test)):
  if Out_label_test[i] == 'DoS':
    DoS_class=encoded_Y_test[i]
  elif Out_label[i] == 'Probe':
      Probe_class=encoded_Y_test[i]
  elif Out_label[i] == 'R2L':
      R2L_class=encoded_Y_test[i]
  elif Out_label[i] == 'U2R':
      U2R_class=encoded_Y_test[i] 
  elif Out_label[i] == 'Normal':
      Normal_class=encoded_Y_test[i]
  else:
      Unknown_class=encoded_Y_test[i]
   

### Convert class from string to value
for i in range(Y_label.size):
  if Y_label[i] in DoS:
     Y_label[i]=DoS_class
     
          
  elif Y_label[i] in Probe:
     Y_label[i]=Probe_class
 
  elif Y_label[i] in R2L:
     Y_label[i]=R2L_class

  elif Y_label[i] in U2R:
     Y_label[i]=U2R_class
  
  elif Y_label[i] in Normal:
     Y_label[i]=Normal_class
   
  else:
     Y_label[i]=Unknown_class

### Convert class from string to value
for i in range(Y_label_test.size):
  if Y_label_test[i] in DoS:
     Y_label_test[i]=DoS_class
     
          
  elif Y_label_test[i] in Probe:
     Y_label_test[i]=Probe_class
 
  elif Y_label_test[i] in R2L:
     Y_label_test[i]=R2L_class

  elif Y_label_test[i] in U2R:
     Y_label_test[i]=U2R_class

  elif Y_label_test[i] in Normal:
     Y_label_test[i]=Normal_class
   
  else:
     Y_label_test[i]=Unknown_class
     

# convert integers to dummy variables (i.e. one hot encoded)
Output_y = np_utils.to_categorical(Y_label)
Output_y_test = np_utils.to_categorical(Y_label_test)


# normalized the train data and test 
X_train=ndata[:,0:41]
X_train_test=ndata_test[:,0:41]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X_train = scaler.fit_transform(X_train)

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X_train_test = scaler.fit_transform(X_train_test)

#print(rescaled_X_train)
#print(Output_y)

# Design the DNN
model = Sequential()
model.add(Dense(50, input_dim=41, kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='softmax'))

	# Compile model        
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])  


# batch_size  100, 1000, 5000, 10000, 10000 and epochs=20
train_history=model.fit(rescaled_X_train,Output_y,batch_size=1000,epochs=150)

#print(train_history.history)
#print(train_history.history.keys())

# After training a model, you can get the weights for each layer by calling the .get_weights() method.

all_weights = []
for layer in model.layers:
   w = layer.get_weights()
   all_weights.append(w)
   



plt.plot(train_history.history['acc'])
#plt.plot(train_history.history['val_acc'])
plt.title('Train History')
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.legend(['train','validation'],loc='upper left')
plt.show()

plt.plot(train_history.history['loss'])
#plt.plot(train_history.history['val_loss'])
plt.title('Train History')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train','loss'],loc='upper left')
plt.show()


scores, acc =model.evaluate(rescaled_X_train_test,Output_y_test)
print('Test score:', scores)
print('Test accuracy:', acc)

'''
Score is the evaluation of the loss function for a given input.

Training a network is finding parameters that minimize a loss function (or cost function).

The cost function here is the binary_crossentropy.

For a target T and a network output O, the binary crossentropy can defined as

f(T,O) = -(T*log(O) + (1-T)*log(1-O) )

So the score you see is the evaluation of that.

If you feed it a batch of inputs it will most likely return the mean loss.

So yeah, if your model has lower loss (at test time), it should often have lower prediction error.
'''









