import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from encoding_vector import load_one_hot_vector

print('Loading data')
print("Gián đoạn! Từ model_random.py Quay về trang encoding.vector")
x, y = load_one_hot_vector()
print("Đã xong, từ encoding.vector quay trở lại model_random.py")
print("x = ",x,x.shape)
print('y = ',y,y.shape)
# x.shape -> (10662, 56)
# y.shape -> (10662, 2)
# len(vocabulary_inv) -> 18765

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
'''
X_train =  [[    0     0     0 ...    16  1389    93]
 [    0     0     0 ...     9   111   191]
 [    0     0     0 ...   883     5    59]
 ...
 [    0     0     0 ...   221   517   182]
 [    0     0     0 ... 10747   542   171]
 [    0     0     0 ...  1028   316   321]] (8529, 56)
X_test =  [[   0    0    0 ...  403   11 9658]
 [   0    0    0 ...  242    5  468]
 [   0    0    0 ...   62  417  457]
 ...
 [   0    0    0 ...    5  291  331]
 [   0    0    0 ...    7 2015   11]
 [   0    0    0 ... 4437 3298  379]] (2133, 56)
y_train =  [[1 0]
 [1 0]
 [1 0]
 ...
 [1 0]
 [0 1]
 [1 0]] (8529, 2)
y_test =  [[1 0]
 [1 0]
 [0 1]
 ...
 [0 1]
 [0 1]
 [1 0]] (2133, 2)
'''
print('X_train = ',X_train,X_train.shape)
print('X_test = ',X_test,X_test.shape)
print('y_train = ',y_train,y_train.shape)
print('y_test = ',y_test,y_test.shape)

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
'''
sequence_length = 56
vocabulary_size = 18765
num_filters = 100
'''
inputs = Input(shape=(sequence_length,), dtype='int32')#sequence_length = 56
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)
print("inputs = ",inputs)
print("embedding = ",embedding)
print("reshape = ",reshape)

#filter_sizes = [3,4,5]
'''
embedding_dim = 50
sequence_length = 56
vocabulary_size = 18765
num_filters = 100
filter_sizes = [3,4,5]
activation = 'relu': hàm kích hoạt
f(x) =  x nếu x >  0
        0 nếu x <= 0
kernel_size: kích thước bộ lọc: 3x50,4x50,5x50
'''
#Lớp tích chập
conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

print("conv_0 = ",conv_0)
print("conv_1 = ",conv_1)
print("conv_2 = ",conv_2)
#filter_sizes = [3,4,5] -> sequence_length - filter_sizes + 1 = [54,53,52]


#Lớp pooling
maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
print("maxpool_0 = ",maxpool_0)
print("maxpool_1 = ",maxpool_1)
print("maxpool_2 = ",maxpool_2)



concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)#drop = 0.5
output = Dense(units=2, activation='softmax', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5))(dropout)
print("concatenated_tensor = ",concatenated_tensor)
print("flatten = ",flatten)
print("dropout = ",dropout)
print("output = ",output)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model_compile = model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("model = ", model.get_weights())
print("checkpoint = ",checkpoint)
print("adam = ",adam.get_weights())
print("model_compile = ",model_compile)#None

print("Traning Model...")
#batch_size = 50
#epochs = 10
'''
X_train =  [[    0     0     0 ...    16  1389    93]
 [    0     0     0 ...     9   111   191]
 [    0     0     0 ...   883     5    59]
 ...
 [    0     0     0 ...   221   517   182]
 [    0     0     0 ... 10747   542   171]
 [    0     0     0 ...  1028   316   321]] (8529, 56)
'''
'''
y_train =  [[1 0]
 [1 0]
 [1 0]
 ...
 [1 0]
 [0 1]
 [1 0]] (8529, 2)
'''
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,  validation_split=0.1)  # starts training

'''
X_test =  [[   0    0    0 ...  403   11 9658]
 [   0    0    0 ...  242    5  468]
 [   0    0    0 ...   62  417  457]
 ...
 [   0    0    0 ...    5  291  331]
 [   0    0    0 ...    7 2015   11]
 [   0    0    0 ... 4437 3298  379]] (2133, 56)
'''
y_predict_prohibit = model.predict(X_test)
'''
y_predict_prohibit =  [[0.933806   0.06619404]
 [0.1734788  0.8265213 ]
 [0.7857168  0.2142832 ]
 ...
 [0.07488674 0.92511326]
 [0.22351727 0.7764827 ]
 [0.6578327  0.34216732]] (2133, 2)
'''
#y_predict_prohibit là 1 mảng 2133 x 2, nếu cột 0 > 0.5 thì vị trí tương ứng trong y_predict_label là true, trái lại là false
y_predict_label = list(map(lambda v: v > 0.5, y_predict_prohibit))
# for i in range(1, 15):
#     print(y_predict_label[i])

'''
y_test =  [[1 0]
 [1 0]
 [0 1]
 ...
 [0 1]
 [0 1]
 [1 0]] (2133, 2)
 '''
accuracy = accuracy_score(y_test, y_predict_label)


print("history = ",history)
print("y_predict_prohibit = ",y_predict_prohibit, y_predict_prohibit.shape)
print("y_predict_label = ",y_predict_label[len(y_predict_label)-1],len(y_predict_label))
print("accuracy = ",accuracy)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
