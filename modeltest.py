import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import KFold
from keras.models import load_model
import h5py
from keras.models import load_model

KDD_model = load_model('train_model.h5')


dataframe = read_csv("KDDTest.csv", header=None)
dataset1 = dataframe.values
dataset_test = dataset1[:,0:42]

X_test = dataset_test[:,0:41]
Y_test = dataset_test[:,41]


for i in range(1,4):
	encoder = LabelEncoder()
	encoder.fit(dataset_test[:,i])
	encoder_Z = encoder.transform(dataset_test[:,i])
	dataset_test[:,i] = encoder_Z

for i in range(len(Y_test)):
	if Y_test[i] == 'normal':
		Y_test[i] = 'Normal'
	elif Y_test[i] == 'back' or Y_test[i] == 'neptune' or Y_test[i] == 'pod' or Y_test[i] == 'smurf' or Y_test[i] == 'teardrop' or Y_test[i] == 'land':
		Y_test[i] = 'DoS'
	elif Y_test[i] == 'ipsweep' or Y_test[i] == 'nmap' or Y_test[i] == 'portsweep' or Y_test[i] == 'satan':
		Y_test[i] = 'Probe'
	elif Y_test[i] == 'warezclient' or Y_test[i] == 'ftp_write' or Y_test[i] == 'guess_passwd' or Y_test[i] == 'multihop' or Y_test[i] == 'phf'  or Y_test[i] == 'spy' or Y_test[i] == 'warezmaster':
		Y_test[i] = 'R2L'
	elif Y_test[i] == 'buffer_overflow' or Y_test[i] == 'loadmodule' or Y_test[i] == 'perl' or Y_test[i] == 'rootkit':
		Y_test[i] = 'U2R'
	else :
		Y_test[i] = 'Unknown'

encoderY = LabelEncoder()
encoderY.fit(Y_test)
encoded_Y = encoderY.transform(Y_test)
dummy_y_test = np_utils.to_categorical(encoded_Y) 
print(dummy_y_test)
scaler = preprocessing.StandardScaler().fit(X_test)
rescaler = scaler.transform(X_test)
scaler = preprocessing.StandardScaler().fit(rescaler)

score = KDD_model.evaluate(X_test, dummy_y_test, batch_size=1000)
print('Test loss: ' , score[0])
print('Test acc: ', score[1]*100)

