import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from tensorflow.keras.layers import Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from encoding_vector import load_pre_train_data

x, y, embedding_matrix = load_pre_train_data()

print('Loading data')


# x.shape -> (10662, 56)
# y.shape -> (10662, 2)
# len(vocabulary) -> 18765
# len(vocabulary_inv) -> 18765

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

# X_train.shape -> (8529, 56)
# y_train.shape -> (8529, 2)
# X_test.shape -> (2133, 56)
# y_test.shape -> (2133, 2)


# sequence_length = x.shape[1] # 56 bởi vì x.shape -> (10662, 56)
sequence_length = 2494 # 56 bởi vì x.shape -> (10662, 56)
print("sequence_length = ",sequence_length)#câu dài nhất có 2494 từ
# vocabulary_size = 18765
vocabulary_size = 57448 #bộ dataset mới
embedding_dim = 50
filter_sizes = [3,4,5]
num_filters = 100
drop = 0.5

epochs = 10
batch_size = 50

# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding_static = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length, weights = [embedding_matrix],trainable= False)(inputs)
embedding_non_static = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length,weights = [embedding_matrix], trainable= True)(inputs)

reshape_static = Reshape((sequence_length,embedding_dim,1))(embedding_static)
reshape_non_static = Reshape((sequence_length,embedding_dim,1))(embedding_non_static)

print("inputs = ",inputs)
print("embedding_static = ",embedding_static)
print("embedding_non_static = ",embedding_non_static)
print("reshape_static = ",reshape_static)
print("reshape_non_static = ",reshape_non_static)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu',)(reshape_non_static)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape_non_static)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape_non_static)

conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', trainable= False)(reshape_static)
conv_4 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', trainable= False)(reshape_static)
conv_5 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', trainable= False)(reshape_static)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

maxpool_3 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_3)
maxpool_4 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_4)
maxpool_5 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_5)

maxpool_6 = tf.math.maximum(maxpool_0, maxpool_3)
maxpool_7 = tf.math.maximum(maxpool_1, maxpool_4)
maxpool_8 = tf.math.maximum(maxpool_2, maxpool_5)
#print(maxpool_2, maxpool_0, maxpool_1);
concatenated_tensor = Concatenate(axis=1)([maxpool_6, maxpool_7, maxpool_8])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=2, activation='softmax', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5))(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,  validation_data=(X_test, y_test))  # starts training
y_predict_prohibit = model.predict(X_test)
y_predict_label = list(map(lambda v: v > 0.5, y_predict_prohibit))
accuracy = accuracy_score(y_test, y_predict_label)
print("--Độ chính xác = ",accuracy)

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model CNN-multichannels')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
