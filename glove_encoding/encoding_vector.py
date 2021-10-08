import re
# lets import some stuff
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_glove():
    embeddings_index = {}
    f = open("glove.6B.50d.txt", "r", encoding="utf8")#mỗi từ bao gồm 1 vector R^50
    # print(f.readlines())
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open('../data/rt-polarity.pos', "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open('../data/rt-polarity.neg', "r", encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    print("Số câu tích cực là positive_labels = ",positive_examples[len(positive_examples)-1], len(positive_examples))
    print("Số nhãn tích cực là positive_labels = ",positive_labels[len(positive_labels)-1], len(positive_labels))
    negative_labels = [[1, 0] for _ in negative_examples]
    print("Số câu tiêu cực là negative_labels = ",negative_examples[len(negative_examples)-1], len(negative_examples))
    print("Số nhãn tiêu cực là negative_labels = ",negative_labels[len(negative_labels)-1], len(negative_labels))
    y = np.concatenate([positive_labels, negative_labels], 0)
    '''
    Số câu tích cực là positive_labels =  ...provides a porthole into that noble , trembling incoherence that defines us all . 5331
    Số nhãn tích cực là positive_labels =  ...[0, 1] 5331
    Số câu tiêu cực là negative_labels =  ...enigma is well-made , but it's just too dry and too placid . 5331
    Số nhãn tiêu cực là negative_labels =  ...[1, 0] 5331
    
    x_text = [
    ... 10659 ['as', 'it', 'stands', ',', 'crocodile', 'hunter', 'has', 'the', 'hurried', ',', 'badly', 'cobbled', 'look', 'of', 'the', '1959', 'godzilla', ',', 'which', 'combined', 'scenes', 'of', 'a', 'japanese', 'monster', 'flick', 'with', 'canned', 'shots', 'of', 'raymond', 'burr', 'commenting', 'on', 'the', 'monster', "'s", 'path', 'of', 'destruction']
    10660 ['the', 'thing', 'looks', 'like', 'a', 'made', 'for', 'home', 'video', 'quickie']
    10661 ['enigma', 'is', 'well', 'made', ',', 'but', 'it', "'s", 'just', 'too', 'dry', 'and', 'too', 'placid']
    ]
    len(x) = 10662
    '''
    return [x_text, y]


def tokenizer_data(data):
    max_features = 20000  # this is the number of words we care about
    # sequence_length = 56
    # sequence_length = 2494
    sequence_length = max(len(x) for x in data)
    print("Chiều dài lớn nhất của 1 câu là sequence_length = ",sequence_length)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)

    print("Đưa data = ", data[len(data)-1], len(data), " vào tokenizer_data ta được")
    '''
    Đưa data = ... ['enigma', 'is', 'well', 'made', ',', 'but', 'it', "'s", 'just', 'too', 'dry', 'and', 'too', 'placid'] 10662  vào tokenizer_data ta được
    '''
    # this takes our sentences and replaces each word with an integer
    X = tokenizer.texts_to_sequences(data)
    print("X = tokenizer.texts_to_sequences(data) = ",len(X))
    '''
    X = tokenizer.texts_to_sequences(data) =  [
        [1, 565, 7, 2633, 6, 22, 1, 3369, 887, 8, 100, 5598, 4, 11, 65, 8, 240, 6, 73, 3, 3913, 57, 2948, 34, 1489, 2393, 2, 2394, 10111, 1708, 7197, 42, 937, 10112]
        [1, 3370, 2181, 7198, 5, 1, 3914, 5, 1, 2949, 4631, 7, 39, 1075, 11, 3, 7199, 5, 823, 888, 4632, 2634, 685, 246, 66, 1076, 889, 8, 4633, 686, 5, 1241, 1709, 1709, 5599, 8, 480, 1490]
        [716, 13, 44, 2635, 2636]
        [40, 21, 267, 28, 6, 191, 6, 1, 101, 6, 37, 132, 2, 5600, 7, 3, 51, 328, 6, 660]
        ...
        [1, 168, 439, 28, 3, 97, 16, 349, 275, 6442]
        [3337, 7, 63, 97, 2, 13, 9, 8, 49, 44, 956, 4, 44, 8339] (length = 10662 bao gồm 5331 câu + và 5331 câu -)
    ]
    '''
    # we then pad the sequences so they're all the same length (sequence_length)

    X = pad_sequences(X, maxlen=sequence_length)
    print("X = pad_sequences(X, maxlen=sequence_length=",sequence_length,") = ",X,X[len(X) - 1])
    '''
    Vì các câu trên có độ dài ko = nhau nên ta thêm các số 0 vào bên trái cho đủ 56 phần tử trong mảng
    VD: [   0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
     3337    7   63   97    2   13    9    8   49   44  956    4   44 8339]
    '''
    word_index = tokenizer.word_index
    '''
    word_index = tokenizer.word_index = {'the': 1, ',': 2, 'a': 3, 'and': 4, 'of': 5, 'to': 6,...,'definitions': 18759, "'time": 18760, "waster'": 18761, 'hurried': 18762, '1959': 18763, 'commenting': 18764}
    '''
    # y = pd.get_dummies(data['Sentiment']).values

    # where there isn't a test set, Kim keeps back 10% of the data for testing, I'm going to do the same since we have an ok amount to play with
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # print("test set size " + str(len(X_test)))
    return X, tokenizer


def create_pretrain_vectors(X):
    print("DỮ LIỆU ĐẦU VÀO X = ",X[len(X)-1],len(X))
    '''
    DỮ LIỆU ĐẦU VÀO X =  ['enigma', 'is', 'well', 'made', ',', 'but', 'it', "'s", 'just', 'too', 'dry', 'and', 'too', 'placid'] 10662
    '''
    embeddings_index = load_glove()#400000 từ
    # print("embeddings_index = load_glove() = ",embeddings_index)
    '''
    embeddings_index = 
    ...,'cubagua': array([ 0.57109 , -0.63577 , -0.20191 ,  0.28871 , -0.41515 ,  0.23298 ,
        0.18418 ,  1.0639  ,  0.21431 ,  0.3221  ,  0.25714 , -0.43339 ,
        1.421   ,  0.32212 , -0.27569 ,  0.18809 , -0.10578 , -0.38507 ,
        0.75146 ,  0.033099,  0.61688 , -0.013702,  0.21781 ,  0.13186 ,
        0.4783  ,  1.1894  ,  0.59397 ,  0.64924 ,  0.40029 , -0.41445 ,
       -1.254   , -0.090746, -0.18584 ,  0.26396 , -0.24033 , -0.36358 ,
       -0.60275 , -0.28451 , -0.34378 , -0.085481, -0.18766 , -0.49134 ,
        0.43289 , -0.72533 , -0.43876 , -0.48901 ,  0.40512 , -0.071588,
       -0.028151, -0.65714 ], dtype=float32),
       
       'nerites': array([-0.26624  , -0.074572 , -0.31718  ,  0.49647  ,  0.32977  ,
       -0.84361  , -0.14784  ,  0.52659  , -0.010153 , -0.9129   ,
       -0.0051815,  0.1303   ,  0.92102  , -0.097595 ,  0.073276 ,
        0.20236  , -0.61804  ,  0.14897  ,  0.67469  ,  0.30773  ,
       -0.23417  , -0.17095  , -0.69459  ,  0.27412  ,  0.25507  ,
        1.4761   ,  0.11786  , -0.61123  ,  0.15601  ,  0.58901  ,
       -1.7497   ,  0.063957 ,  0.27766  ,  0.28113  , -0.067202 ,
        0.36202  , -0.35695  , -0.35329  , -0.11329  , -0.2308   ,
        0.18509  , -1.0952   , -0.14384  ,  0.83838  ,  0.15106  ,
       -0.083365 ,  0.269    , -0.35663  , -0.30989  ,  0.13176  ], dtype=float32),
      
      'swartland': array([-0.47618 , -0.8201  , -0.22952 , -0.04365 , -0.7108  , -0.40953 ,
        0.33023 ,  0.85019 ,  0.15985 ,  0.11389 ,  0.55382 , -0.20868 ,
        1.3581  ,  0.5527  , -0.85067 ,  0.015544,  0.72046 ,  0.21893 ,
        0.49171 ,  0.51826 , -0.51624 , -0.56922 , -0.23038 ,  0.59274 ,
       -0.087474,  1.3969  ,  0.069287,  0.35148 , -0.16813 ,  0.17421 ,
       -1.1481  , -0.6704  ,  0.58367 , -0.13645 ,  0.16125 ,  0.70473 ,
       -0.1997  , -0.78529 ,  0.0451  ,  0.22057 ,  0.082777, -0.12633 ,
        0.102   , -0.4043  , -0.73585 ,  0.37395 ,  0.40996 ,  0.7603  ,
       -0.26234 , -0.39384 ], dtype=float32),...
    '''
    X_train, tokenizer = tokenizer_data(X)
    '''
    X_train =  [[    0     0     0 ...    42   937 10112]
     [    0     0     0 ...     8   480  1490]
     [    0     0     0 ...    44  2635  2636]
     ...
     [    0     0     0 ...  2042     5  4345]
     [    0     0     0 ...   349   275  6442]
     [    0     0     0 ...     4    44  8339]] (10662, 56) 10662 câu bao gồm 5331 câu + và - mỗi loại
     tokenizer.word_index = {'the': 1, ',': 2, 'a': 3, 'and': 4, 'of': 5, 'to': 6,...,'definitions': 18759, "'time": 18760, "waster'": 18761, 'hurried': 18762, '1959': 18763, 'commenting': 18764}
    '''
    print("X_train = ",X_train,X_train.shape,len(X_train))
    # print("tokenizer = ",tokenizer,tokenizer.word_index)#<keras_preprocessing.text.Tokenizer object at 0x000002723200FD08>
    num_words = len(tokenizer.word_index)   #18764
    '''
    word_index = tokenizer.word_index = {'the': 1, ',': 2, 'a': 3, 'and': 4, 'of': 5, 'to': 6,...,'definitions': 18759, "'time": 18760, "waster'": 18761, 'hurried': 18762, '1959': 18763, 'commenting': 18764}
    '''
    print("num_words = ",num_words)#18764
    embedding_dim = 50
    # first create a matrix of zeros, this is our embedding matrix
    embedding_matrix = np.random.uniform(-0.25, 0.25, [num_words + 1, embedding_dim])
    '''np.random.uniform(-0.25, 0.25, [18765, 50]) -> 1 ma trận kích cỡ 18765 x 50 gồm toàn các số từ [-0.25;0.25]'''
    print("embedding_matrix = ",embedding_matrix,embedding_matrix.shape)#1 ma trận kích cỡ 18765 x 50 gồm toàn các số từ [-0.25;0.25]
    for word, i in tokenizer.word_index.items():#duyệt từng từ 1 trong số 18764 từ kia, i chạy từ 1 đến 18764
        # if(i > num_words): continue
        embedding_vector = embeddings_index.get(word)#với word = 1 -> 18764, tìm từng từ 1 (từ word) trong tổng số 400000 từ trong load_glove()
        '''
        (word, i) =  (commenting, 18764)
        embedding_vector =  [-1.2052e-01 -4.0552e-02 -7.2908e-01  1.9339e-01  1.0442e-02 -2.2662e-01
         -1.6251e-02  1.8911e-01 -4.0526e-01  1.3020e-01 -1.8348e-01  3.3031e-01
         -3.6352e-01  9.0073e-03  7.0418e-01  3.5828e-02 -4.4733e-01 -6.5598e-01
          3.7794e-01 -1.1968e-01  6.7509e-01  4.7399e-01  6.1681e-01 -9.1943e-04
          8.8798e-01 -8.6004e-01 -4.2329e-02  6.3347e-01 -4.7317e-01  4.8146e-02
          1.2688e+00 -3.4170e-01  3.1129e-01 -7.0181e-01 -5.0539e-01 -4.9846e-01
          3.9921e-01 -2.6459e-01 -1.1099e+00 -2.0320e-03  1.0371e-01 -2.2844e-02
         -4.6529e-01 -2.4713e-01  2.9375e-01 -5.2541e-01 -1.9085e-01  1.2925e+00
          4.4365e-01  4.6728e-01] (get from glove)
        '''

        '''
        (word, i) =  (roteiro, 17086)
        embedding_vector =  None
        
        '''
        # print(embedding_vector)
        if embedding_vector is not None:#embedding_vector lấy từ trong glove nếu tìm thấy từ word
            # we found the word - add that words vector to the matrix: Tìm thấy từ word trong glove
            for j in range(0, 50, 1):   #j từ 0,1,2,...,48,49
                embedding_matrix[i][j] = embedding_vector[j]
                '''
                ...
                embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 39 ] =  0.9753299951553345
                embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 40 ] =  -0.32012999057769775
                embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 41 ] =  -0.3647800087928772
                embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 42 ] =  -0.5879700183868408
                embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 43 ] =  0.7087399959564209
                embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 44 ] =  -0.7298799753189087
                embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 45 ] =  -1.1770999431610107
                embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 46 ] =  2.0943000316619873
                embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 47 ] =  -0.14892999827861786
                embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 48 ] =  0.5447499752044678
                embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 49 ] =  0.24785999953746796
                '''
        else:   #ko tìm thấy từ word trong glove
            # doesn't exist, assign a random vector
            embedding_matrix[i] = np.random.randn(embedding_dim)#embedding_dim = 50 -> random 1 vector 50 chiều và gán cho embedding_matrix[i]
            '''
            embedding_matrix[i] =  embedding_matrix[ 17086 ] =  [-0.08282215  2.3330552   0.48603692 -0.79119773 -0.65865271 -0.78010235
         -1.08084054  1.81642096 -1.44727082  1.61452508  1.00989054 -0.55773045
         -1.17652845 -1.60835573 -0.23047082  1.85180883 -0.18887875 -0.38421895
          0.99428963  1.22978062  0.62705164  0.77319553 -1.00248422 -1.04089605
         -1.48384281 -1.19981462  0.01139322  0.03671663 -1.52797511 -1.22683945
          0.94704993  1.01088809 -0.10236021 -0.66223604 -1.10825006  0.58423114
          0.06373448 -0.8246087   0.69869425  0.76869243  1.39097501  0.35198539
          0.59237324  1.00642328 -1.45817682 -0.97036542 -1.59260274 -1.00682717
         -0.30411879  1.21423292]  (embedding_matrix[i] = np.random.randn(embedding_dim))
            '''
    return X_train, embedding_matrix


def load_pre_train_data():
    x, y = load_data_and_labels()
    print("tại load_data_and_labels() thì x = ", x[len(x)-1], len(x))
    '''
    ...
    10659 ['as', 'it', 'stands', ',', 'crocodile', 'hunter', 'has', 'the', 'hurried', ',', 'badly', 'cobbled', 'look', 'of', 'the', '1959', 'godzilla', ',', 'which', 'combined', 'scenes', 'of', 'a', 'japanese', 'monster', 'flick', 'with', 'canned', 'shots', 'of', 'raymond', 'burr', 'commenting', 'on', 'the', 'monster', "'s", 'path', 'of', 'destruction']
    10660 ['the', 'thing', 'looks', 'like', 'a', 'made', 'for', 'home', 'video', 'quickie']
    10661 ['enigma', 'is', 'well', 'made', ',', 'but', 'it', "'s", 'just', 'too', 'dry', 'and', 'too', 'placid']
    len(x) = 10662
    '''
    print("tại load_data_and_labels() thì y = ", y[len(y)-1], y.shape)#tại load_data_and_labels() thì y =  [1 0] (10662, 2)
    # X_train, word_index = tokenizer_data(x)
    X_train, embedding_matrix = create_pretrain_vectors(x)
    '''
            X_train =  [[    0     0     0 ...    42   937 10112]
             [    0     0     0 ...     8   480  1490]
             [    0     0     0 ...    44  2635  2636]
             ...
             [    0     0     0 ...  2042     5  4345]
             [    0     0     0 ...   349   275  6442]
             [    0     0     0 ...     4    44  8339]] (10662, 56) 10662 câu bao gồm 5331 câu + và - mỗi loại
             
             y: (10662,2)
                 Số nhãn tích cực là positive_labels =  ...[0, 1] 5331
                 Số nhãn tiêu cực là negative_labels =  ...[1, 0] 5331

             
        embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 39 ] =  0.9753299951553345
                        embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 40 ] =  -0.32012999057769775
                        embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 41 ] =  -0.3647800087928772
                        embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 42 ] =  -0.5879700183868408
                        embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 43 ] =  0.7087399959564209
                        embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 44 ] =  -0.7298799753189087
                        embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 45 ] =  -1.1770999431610107
                        embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 46 ] =  2.0943000316619873
                        embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 47 ] =  -0.14892999827861786
                        embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 48 ] =  0.5447499752044678
                        embedding_matrix[i][j] =  embedding_matrix[ 18508 ][ 49 ] =  0.24785999953746796
        embedding_matrix[i] =  embedding_matrix[ 17086 ] =  [-0.08282215  2.3330552   0.48603692 -0.79119773 -0.65865271 -0.78010235
                 -1.08084054  1.81642096 -1.44727082  1.61452508  1.00989054 -0.55773045
                 -1.17652845 -1.60835573 -0.23047082  1.85180883 -0.18887875 -0.38421895
                  0.99428963  1.22978062  0.62705164  0.77319553 -1.00248422 -1.04089605
                 -1.48384281 -1.19981462  0.01139322  0.03671663 -1.52797511 -1.22683945
                  0.94704993  1.01088809 -0.10236021 -0.66223604 -1.10825006  0.58423114
                  0.06373448 -0.8246087   0.69869425  0.76869243  1.39097501  0.35198539
                  0.59237324  1.00642328 -1.45817682 -0.97036542 -1.59260274 -1.00682717
                 -0.30411879  1.21423292]  (embedding_matrix[i] = np.random.randn(embedding_dim))
    '''
    return [X_train, y, embedding_matrix]


def load_one_hot_vector():
    x, y = load_data_and_labels()
    print("tại load_data_and_labels() thì x = ", x[len(x)-1], len(x))
    '''
    ...
    10659 ['as', 'it', 'stands', ',', 'crocodile', 'hunter', 'has', 'the', 'hurried', ',', 'badly', 'cobbled', 'look', 'of', 'the', '1959', 'godzilla', ',', 'which', 'combined', 'scenes', 'of', 'a', 'japanese', 'monster', 'flick', 'with', 'canned', 'shots', 'of', 'raymond', 'burr', 'commenting', 'on', 'the', 'monster', "'s", 'path', 'of', 'destruction']
    10660 ['the', 'thing', 'looks', 'like', 'a', 'made', 'for', 'home', 'video', 'quickie']
    10661 ['enigma', 'is', 'well', 'made', ',', 'but', 'it', "'s", 'just', 'too', 'dry', 'and', 'too', 'placid']
    len(x) = 10662
    '''
    print("tại load_data_and_labels() thì y = ", y[len(y)-1], y.shape)#tại load_data_and_labels() thì y =  [1 0] (10662, 2)



    X_train, tokenizer = tokenizer_data(x)
    print("tại load_data_and_labels() thì X_train = ", X_train, X_train.shape)
    '''
    tại load_data_and_labels() thì X_train =  [[    0     0     0 ...    42   937 10112]
     [    0     0     0 ...     8   480  1490]
     [    0     0     0 ...    44  2635  2636]
     ...
     [    0     0     0 ...  2042     5  4345]
     [    0     0     0 ...   349   275  6442]
     [    0     0     0 ...     4    44  8339]] (10662, 56)
    '''
    print("tại load_data_and_labels() thì tokenizer = ", tokenizer)
    '''
    tokenizer.word_index = {'the': 1, ',': 2, 'a': 3, 'and': 4, 'of': 5, 'to': 6,...,'definitions': 18759, "'time": 18760, "waster'": 18761, 'hurried': 18762, '1959': 18763, 'commenting': 18764}
    '''
    return [X_train, y]


if __name__ == '__main__':
    #3 dòng dưới đây thực chất là load_pre_train_data():
    # x, y = load_data_and_labels()
    # # X_train, word_index = tokenizer_data(x)
    # X_train, embedding_matrix = create_pretrain_vectors(x)



    load_pre_train_data()
    #######################
    # print(X_train[0])
    '''
        [    0     0     0     0     0     0     0     0     0     0     0     0
         0     0     0     0     0     0     0     0     0     0     1   565
         7  2633     6    22     1  3369   887     8   100  5598     4    11
        65     8   240     6    73     3  3913    57  2948    34  1489  2393
         2  2394 10111  1708  7197    42   937 10112]
    '''
    # X_train, y = load_one_hot_vector()
    # print(y[0])#[0 1]
