from keras.utils import np_utils
import joblib
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def get_data_label(fea):
	if fea == 'all':
		fea = 'seq+str+dyn'

	data = np.load('data/' + fea + '/test_data_1d.npy', allow_pickle=True).astype(float)
	label = np.load('data/' + fea + '/test_label_1d.npy').astype(float)

	label = np_utils.to_categorical(label)

	return data, label


def predict(data,fea):
	if fea == 'all':
		fea = 'seq+str+dyn'
		
	rf_model = joblib.load(fea + '/rf.pkl')
	pred_prob = rf_model.predict_proba(data)
	
	return pred_prob


if __name__ == '__main__':
	fea = 'all'

	data, label = get_data_label(fea)
	pred_prob = predict(data, fea)
	pred_label = np.argmax(pred_prob, axis=1)

	print(pred_label)


