import csvReader as Cr
from keras.layers import Dense, GRU, Embedding
from keras.models import Sequential
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.models import load_model as load


while True:
    print('-'*300)
    if int(input('Choose:\n    0-Generate new model\n    1-Load previous model\n0/1: ')) == 1:
        model = load('PhoneReviewRu')
        while True:
            result = model.predict(pad_sequences(
                token.texts_to_sequences([input('Enter your prompt: ').lower().replace('[\.,123456789!?()\-\$%"\']', '')]),
                maxlen=200))
            print('Estimated estimate is: ', np.argmax(result)+1)
            print('-'*300)
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=200))
    model.add(GRU(128))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(xTrainDiv, yTrainDiv, batch_size=70, epochs=16)
    model.save('PhoneReviewRu')

print('Data loading...')

xTrain = Cr.readColumn('new1.csv', 0, True)
yTrain = Cr.readColumn('new1.csv', 1, False)

xTrainDiv = []
yTrainDiv = []
for obj in range(13000):
    if yTrain[obj] == 1:
        yTrainDiv.append([1, 0, 0, 0, 0])
    elif yTrain[obj] == 2:
        yTrainDiv.append([0, 1, 0, 0, 0])
    elif yTrain[obj] == 3:
        yTrainDiv.append( [0, 0, 1, 0, 0])
    elif yTrain[obj] == 4:
        yTrainDiv.append([0, 0, 0, 1, 0])
    elif yTrain[obj] == 5:
        yTrainDiv.append([0, 0, 0, 0, 1])
    xTrainDiv.append(xTrain[obj])

xTrainDiv = np.array(xTrainDiv)
yTrainDiv = np.array(yTrainDiv)
token = Tokenizer(num_words=5000)
token.fit_on_texts(xTrainDiv)
xTrainDiv = pad_sequences(token.texts_to_sequences(xTrainDiv), maxlen=100)