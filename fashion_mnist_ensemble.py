import struct
import numpy as np
import keras
import matplotlib.pyplot as plt 
from keras.layers import LSTM
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp
from keras.models import Model, Sequential 
from keras.layers import Input, Dense, Activation, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Average, Dropout, Flatten, Add

#loading data
mnist_img = 'mnist/train-images_mnist.idx3-ubyte'
mnist_lbl = 'mnist/train-labels_mnist.idx1-ubyte'
fashion_mnist_train_img = 'fashion_mnist/train-images_fashion-idx3-ubyte'
fashion_mnist_train_lbl = 'fashion_mnist/train-labels_fashion-idx1-ubyte'
fashion_mnist_test_img = 'fashion_mnist/t10k-images-idx3-ubyte'
fashion_mnist_test_lbl = 'fashion_mnist/t10k-labels-idx1-ubyte'

with open(mnist_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)

with open(mnist_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

with open(fashion_mnist_train_lbl , 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    fashion_train_lbl = np.fromfile(flbl, dtype=np.int8)

with open(fashion_mnist_train_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    fashion_train_img = np.fromfile(fimg, dtype=np.uint8).reshape(len(fashion_train_lbl), rows, cols)

with open(fashion_mnist_test_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    fashion_test_lbl = np.fromfile(flbl, dtype=np.int8)

with open(fashion_mnist_test_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    fashion_test_img = np.fromfile(fimg, dtype=np.uint8).reshape(len(fashion_test_lbl), rows, cols)

# Training parameters.
batch_size = 32
num_classes = 10
epochs = 10 #for transfer learning model
epochs_CNN = 6 #for fashion model
epochs_LSTM = 3 #chosen basesd on acuracy plot after trials. After 3 overfitting happens
epochs_ensemble = 3

#functions for accuracy, loss and ROC plots
#Note: with just 1 epoch you will not get the plot. increase epochs to see plot
def accuracy_plot(epochs, train_acc, val_acc, name):
	plt.plot(epochs, train_acc, label='Training accuracy')  #label here is for legend
	plt.plot(epochs, val_acc, label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig(name)
	plt.show()

def loss_plot(epochs, train_loss, val_loss, name):
	plt.plot(epochs, train_loss, label='Training loss')
	plt.plot(epochs, val_loss, label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(name)
	plt.show()

def ROC_plot(test_pred, name):
	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(num_classes):
		fpr[i], tpr[i], _ = roc_curve(fashion_test_y[:, i], test_pred[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(fashion_test_y.ravel(), test_pred.ravel())		#.ravel() returns a flattened array
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	# Compute macro-average ROC curve and ROC area
	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(num_classes):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])
	# Finally average it and compute AUC
	mean_tpr /= num_classes

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	plt.figure()
	plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})' 
		''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

	plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'
        ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
	lw = 2
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'yellow', 'violet', 'brown', 'pink', 'purple', 'olive'])
	for i, color in zip(range(num_classes), colors):
		plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'
            ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic (ROC)')
	plt.legend(loc="lower right")
	plt.savefig(name)
	plt.show()

#training on mnist data and saving weights to create a customized model for fashion_mnist using transfer learning

mnist_train_x = np.asarray(img)
mnist_train_x = np.reshape(mnist_train_x , [len(mnist_train_x),28,28,1])	#len(mnist_train_x) is no.of images in training set, 60,000
mnist_train_y  = keras.utils.np_utils.to_categorical(lbl,num_classes)		# Convert labels to categorical one-hot encoding, num_classes we have in dataset is 10
mnist_train_y = np.asarray(mnist_train_y) 
#print(mnist_train_x.shape)
print("MNIST train data's input shape:", mnist_train_x.shape)	#input is images, output is labels
print("MNIST train data's output shape:", mnist_train_y.shape)

mnist_shape = mnist_train_x[0].shape
mnist_input = Input(shape=mnist_shape)

def mnist_cnn():

    x = Sequential()
    x = Conv2D(64, (3,3),  activation='relu', name = "conv1" )(mnist_input)
    x = MaxPooling2D(pool_size=(2, 2), name = "pool1")(x)
    x = Dropout(0.25)(x)	#Dropout avoids overfitting
    x = Conv2D(32, (3, 3), activation='relu', name = "conv2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name = "pool2")(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation = 'softmax')(x)	#In final layer the number of class labels (here 10) is given
  
    model = Model(mnist_input, x, name='custom_cnn')
    
    return model

mnist_model = mnist_cnn()

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
mnist_model.compile(loss='categorical_crossentropy', optimizer=opt,  metrics=['accuracy'])
mnist_model.fit(mnist_train_x , mnist_train_y  , batch_size= batch_size , epochs = epochs ,  shuffle=True)
mnist_model.save('models/mnist_model.h5') #saves in models folder
mnist_model.save_weights("weights/mnist_original_model_weights.h5")

#create training and testing data for fashion_mnist
fashion_train_x = np.asarray(fashion_train_img)
fashion_train_x = np.reshape(fashion_train_x , [len(fashion_train_x),28,28,1])     		 #instead of len(fashion_train_x) you can directly give 60,000 also,i.e., no. of images in training set
fashion_train_y = keras.utils.np_utils.to_categorical(fashion_train_lbl,num_classes)	 # Convert labels to categorical one-hot encoding
fashion_train_y = np.asarray(fashion_train_y)
print("Fashion MNIST train data's input shape:", fashion_train_x.shape)
print("Fashion MNIST train data's output shape:", fashion_train_y.shape)

fashion_test_x = np.asarray(fashion_test_img)
fashion_test_x = np.reshape(fashion_test_x , [len(fashion_test_x),28,28,1])         #instead of len(fashion_test_x) you can directly give 10,000 also,i.e., no. of images in testing set
fashion_test_y = keras.utils.np_utils.to_categorical(fashion_test_lbl,num_classes)			# Convert labels to categorical one-hot encoding
fashion_test_y = np.asarray(fashion_test_y)
print("Fashion MNIST test data's input shape:", fashion_test_x.shape)
print("Fashion MNIST test data's output shape:", fashion_test_y.shape)

#to save accuracy of all models
accuracy_scores = { }

#input shape is going to be same for all fashion_mnist models
input_shape = fashion_train_x[0].shape
model_input = Input(shape=input_shape)

#customized CNN for fashion_mnist using transfer learning, i.e using the originial mnist weights saved above.
def custom_cnn():

    x = Sequential()
    x = Conv2D(64, (3,3),  activation='relu', name = "conv1" )(model_input)
    x = MaxPooling2D(pool_size=(2, 2), name = "pool1")(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation = 'softmax')(x)
  
    model = Model(model_input, x, name='custom_cnn')
    
    return model


custom_model = custom_cnn()

custom_model.load_weights("weights/mnist_original_model_weights.h5",by_name=True)
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
custom_model.compile(loss='categorical_crossentropy', optimizer=opt,  metrics=['accuracy'])
custom_history = custom_model.fit( fashion_train_x , fashion_train_y  , batch_size= batch_size ,epochs = epochs ,validation_data=(fashion_test_x,fashion_test_y) )
custom_model.save('models/custom_model.h5')
custom_model.save_weights("weights/custom_model_weights.h5")

custom_train_predict = custom_model.predict(fashion_train_x, batch_size = batch_size)	#this will be used for training ensemble model
custom_test_predict = custom_model.predict(fashion_test_x, batch_size = batch_size)		#.predict and .predict_proba give probabilities, .predict_classes gives class labels
#print(custom_test_predict)
custom_scores = custom_model.evaluate(fashion_test_x,fashion_test_y, batch_size=batch_size)
print('Validation loss of transfer learning model:', custom_scores[0])
print('Validation accuracy of transfer learning model:', custom_scores[1])
accuracy_scores["custom_fashion_mnist"] = custom_history.history["val_acc"][-1]		#This value is same as the validation accuracy printed above. (custom_scores[1])

custom_confusion_matrix = metrics.confusion_matrix(fashion_test_y.argmax(axis=1), custom_test_predict.argmax(axis=1))	#argmax converts one hot encoding back to labels(single vector)
print ("Confusion Matrix of transfer learning model:")
print(custom_confusion_matrix)

custom_train_accuracy = custom_history.history['acc']
custom_val_accuracy = custom_history.history['val_acc']
custom_train_loss = custom_history.history['loss']
custom_val_loss = custom_history.history['val_loss']
custom_epochs = range(len(custom_train_accuracy))

accuracy_plot(custom_epochs, custom_train_accuracy, custom_val_accuracy, 'plots/custom_accuracy_plot.png') #function definition is above
loss_plot(custom_epochs, custom_train_loss, custom_val_loss, 'plots/custom_loss_plot.png')
ROC_plot(custom_test_predict, 'plots/custom_ROC_plot.png')

#training our own CNN
def fashion_cnn():

    x = Sequential()
    x = Conv2D(64, (3,3),  activation='tanh')(model_input)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation = 'softmax')(x)
  
    model = Model(model_input, x, name='fashion_cnn')
    
    return model

fashion_model = fashion_cnn()

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
fashion_model.compile(loss='categorical_crossentropy', optimizer=opt,  metrics=['accuracy'])
fashion_history = fashion_model.fit( fashion_train_x , fashion_train_y , batch_size= batch_size, epochs = epochs_CNN, validation_data=(fashion_test_x,fashion_test_y) )
fashion_model.save('models/fashion_model.h5')
fashion_model.save_weights("weights/fashion_model_weights.h5")

fashion_train_predict = fashion_model.predict(fashion_train_x, batch_size = batch_size)		#this will be used for training ensemble model
fashion_test_predict = fashion_model.predict(fashion_test_x, batch_size = batch_size)
fashion_scores = fashion_model.evaluate(fashion_test_x,fashion_test_y, batch_size=batch_size)
print('Validation loss of CNN model:', fashion_scores[0])
print('Validation accuracy of CNN model:', fashion_scores[1])
accuracy_scores["own_cnn_fashion_mnist"] = fashion_history.history["val_acc"][-1]

fashion_confusion_matrix = metrics.confusion_matrix(fashion_test_y.argmax(axis=1), fashion_test_predict.argmax(axis=1))	#argmax converts one hot encoding back to labels(single vector)
print ("Confusion Matrix of CNN model:")
print(fashion_confusion_matrix)

fashion_train_accuracy = fashion_history.history['acc']
fashion_val_accuracy = fashion_history.history['val_acc']
fashion_train_loss = fashion_history.history['loss']
fashion_val_loss = fashion_history.history['val_loss']
fashion_epochs = range(len(fashion_train_accuracy))

accuracy_plot(fashion_epochs, fashion_train_accuracy, fashion_val_accuracy, 'plots/fashion_accuracy_plot.png')
loss_plot(fashion_epochs, fashion_train_loss, fashion_val_loss, 'plots/fashion_loss_plot.png')
ROC_plot(fashion_test_predict, 'plots/fashion_ROC_plot.png')

#training LSTM model
def fashion_LSTM():
	# Embedding dimensions.
	row_hidden = 128
	col_hidden = 128
	# Encodes a row of pixels using TimeDistributed Wrapper.
	encoded_rows = TimeDistributed(LSTM(row_hidden))(model_input)
	# Encodes columns of encoded rows.
	encoded_columns = LSTM(col_hidden)(encoded_rows)
	# Final predictions and model.
	prediction = Dense(num_classes, activation='softmax')(encoded_columns)
	model = Model(model_input, prediction)

	return model

LSTM_model = fashion_LSTM()

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
LSTM_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
LSTM_history = LSTM_model.fit(fashion_train_x, fashion_train_y, batch_size=batch_size, epochs=epochs_LSTM, validation_data=(fashion_test_x, fashion_test_y))
LSTM_model.save('models/LSTM_model.h5')
LSTM_model.save_weights("weights/LSTM_model_weights.h5")

LSTM_train_predict = LSTM_model.predict(fashion_train_x, batch_size = batch_size)		#this will be used for training ensemble model
LSTM_test_predict = LSTM_model.predict(fashion_test_x, batch_size = batch_size)
LSTM_scores = LSTM_model.evaluate(fashion_test_x, fashion_test_y)
print('Validation loss of LSTM model:', LSTM_scores[0])
print('Validation accuracy of LSTM model:', LSTM_scores[1])
accuracy_scores["LSTM_fashion_mnist"] = LSTM_history.history["val_acc"][-1]

LSTM_confusion_matrix = metrics.confusion_matrix(fashion_test_y.argmax(axis=1), LSTM_test_predict.argmax(axis=1))	#argmax converts one hot encoding back to labels(single vector)
print ("Confusion Matrix of LSTM model:")
print(LSTM_confusion_matrix)

LSTM_train_accuracy = LSTM_history.history['acc']
LSTM_val_accuracy = LSTM_history.history['val_acc']
LSTM_train_loss = LSTM_history.history['loss']
LSTM_val_loss = LSTM_history.history['val_loss']
LSTM_epochs = range(len(LSTM_train_accuracy))

accuracy_plot(LSTM_epochs, LSTM_train_accuracy, LSTM_val_accuracy, 'plots/LSTM_accuracy_plot.png')
loss_plot(LSTM_epochs, LSTM_train_loss, LSTM_val_loss, 'plots/LSTM_loss_plot.png')
ROC_plot(LSTM_test_predict, 'plots/LSTM_ROC_plot.png')


#ensemble model 

def ensemble(models, model_input):
    
    outputs = [model.outputs[0] for model in models]
    #y = Add()(outputs)
    y = Average()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    
    return model

custom_model.load_weights('weights/custom_model_weights.h5')
fashion_model.load_weights('weights/fashion_model_weights.h5')
LSTM_model.load_weights("weights/LSTM_model_weights.h5")

models = [custom_model, fashion_model, LSTM_model] 

ensemble_model = ensemble(models, model_input)

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
ensemble_model.compile(loss='categorical_crossentropy', optimizer=opt,  metrics=['accuracy'])
ensemble_history = ensemble_model.fit( fashion_train_x , fashion_train_y , batch_size= batch_size, epochs = epochs_ensemble, validation_data=(fashion_test_x,fashion_test_y) )
ensemble_model.save('models/ensemble_model.h5')
ensemble_model.save_weights("weights/ensemble_model_weights.h5")

ensemble_test_predict = ensemble_model.predict(fashion_test_x, batch_size = batch_size)
ensemble_scores = ensemble_model.evaluate(fashion_test_x,fashion_test_y, batch_size=batch_size)
print('Validation loss of ensemble model:', ensemble_scores[0])
print('Validation accuracy of ensemble model:', ensemble_scores[1])
accuracy_scores["ensemble_fashion_mnist"] = ensemble_history.history["val_acc"][-1]

ensemble_confusion_matrix = metrics.confusion_matrix(fashion_test_y.argmax(axis=1), ensemble_test_predict.argmax(axis=1))	#argmax converts one hot encoding back to labels(single vector)
print ("Confusion Matrix of ensemble model:")
print(ensemble_confusion_matrix)

ensemble_train_accuracy = ensemble_history.history['acc']
ensemble_val_accuracy = ensemble_history.history['val_acc']
ensemble_train_loss = ensemble_history.history['loss']
ensemble_val_loss = ensemble_history.history['val_loss']
ensemble_epochs = range(len(ensemble_train_accuracy))

accuracy_plot(ensemble_epochs, ensemble_train_accuracy, ensemble_val_accuracy, 'plots/ensemble_accuracy_plot.png')
loss_plot(ensemble_epochs, ensemble_train_loss, ensemble_val_loss, 'plots/ensemble_loss_plot.png')
ROC_plot(ensemble_test_predict, 'plots/ensemble_ROC_plot.png')

#view accuracy of all models
for i in accuracy_scores:
    print(i, accuracy_scores[i])