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

speed = 7
numpy.random.seed(speed)

dataframe = read_csv("KDDTrain.csv", header=None)
dataset1 = dataframe.values
dataset = dataset1[:,0:42]

X = dataset[:,0:41]
Y = dataset[:,41]


for i in range(1,4):
	encoder = LabelEncoder()
	encoder.fit(dataset[:,i])
	encoder_Z = encoder.transform(dataset[:,i])
	dataset[:,i] = encoder_Z

for i in range(len(Y)):
	if Y[i] == 'normal':
		Y[i] = 'Normal'
	elif Y[i] == 'back' or Y[i] == 'neptune' or Y[i] == 'pod' or Y[i] == 'smurf' or Y[i] == 'teardrop' or Y[i] == 'land':
		Y[i] = 'DoS'
	elif Y[i] == 'ipsweep' or Y[i] == 'nmap' or Y[i] == 'portsweep' or Y[i] == 'satan':
		Y[i] = 'Probe'
	elif Y[i] == 'warezclient' or Y[i] == 'ftp_write' or Y[i] == 'guess_passwd' or Y[i] == 'multihop' or Y[i] == 'phf'  or Y[i] == 'spy' or Y[i] == 'warezmaster':
		Y[i] = 'R2L'
	elif Y[i] == 'buffer_overflow' or Y[i] == 'loadmodule' or Y[i] == 'perl' or Y[i] == 'rootkit':
		Y[i] = 'U2R'
	else :
		Y[i] = 'Unknown'	

encoderY = LabelEncoder()
encoderY.fit(Y)
encoded_Y = encoderY.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y) 

scaler = preprocessing.StandardScaler().fit(X)
rescaler = scaler.transform(X)
scaler = preprocessing.StandardScaler().fit(rescaler)


'''
def KDD_model():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=41, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model  

estimator = KerasClassifier(build_fn=KDD_model, epochs=200, batch_size=10000, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=speed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
'''
model = Sequential()
model.add(Dense(60, input_dim=41, kernel_initializer='normal', activation='relu'))
model.add(Dense(30, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, dummy_y, epochs=150, batch_size=1000)
sorce = model.evaluate(X, dummy_y)
print(sorce)
print("\n%s: %.2f%%" % (model.metrics_names[1], sorce[1]*100))

#save model
#model= KDD_model
model.save('train_model.h5')











