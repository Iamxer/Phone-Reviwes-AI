import csvReader as Cr
from keras.optimizers import Adam as adam
from keras.layers import Dense, GRU, Embedding
from keras.models import Sequential
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.models import load_model as load


while True:

    print('Data loading...')

    xTrain1 = Cr.readColumn('smallData.csv', 0, True)
    yTrain1 = Cr.readColumn('smallData.csv', 1, False)
    xTrain2 = Cr.readColumn('mediumData.csv', 0, True)
    yTrain2 = Cr.readColumn('mediumData.csv', 1, False)
    xTrain3 = Cr.readColumn('bigData.csv', 0, True)
    yTrain3 = Cr.readColumn('bigData.csv', 1, False)

    xTrainDiv1 = []
    yTrainDiv1 = []
    xTrainDiv2 = []
    yTrainDiv2 = []
    xTrainDiv3 = []
    yTrainDiv3 = []
    for obj in range(10000):
        if yTrain1[obj] == 1:
            yTrainDiv1.append([1, 0, 0, 0, 0])
        elif yTrain1[obj] == 2:
            yTrainDiv1.append([0, 1, 0, 0, 0])
        elif yTrain1[obj] == 3:
            yTrainDiv1.append( [0, 0, 1, 0, 0])
        elif yTrain1[obj] == 4:
            yTrainDiv1.append([0, 0, 0, 1, 0])
        elif yTrain1[obj] == 5:
            yTrainDiv1.append([0, 0, 0, 0, 1])
        xTrainDiv1.append(xTrain1[obj])
    xTrainDiv1 = np.array(xTrainDiv1)
    yTrainDiv1 = np.array(yTrainDiv1)
    token1 = Tokenizer(num_words=5000)
    token1.fit_on_texts(xTrainDiv1)
    xTrainDiv1 = pad_sequences(token1.texts_to_sequences(xTrainDiv1), maxlen=150)

    for obj in range(10000):
        if yTrain2[obj] == 1:
            yTrainDiv2.append([1, 0, 0, 0, 0])
        elif yTrain2[obj] == 2:
            yTrainDiv2.append([0, 1, 0, 0, 0])
        elif yTrain2[obj] == 3:
            yTrainDiv2.append( [0, 0, 1, 0, 0])
        elif yTrain2[obj] == 4:
            yTrainDiv2.append([0, 0, 0, 1, 0])
        elif yTrain2[obj] == 5:
            yTrainDiv2.append([0, 0, 0, 0, 1])
        xTrainDiv2.append(xTrain2[obj])
    xTrainDiv2 = np.array(xTrainDiv2)
    yTrainDiv2 = np.array(yTrainDiv2)
    token2 = Tokenizer(num_words=5000)
    token2.fit_on_texts(xTrainDiv2)
    xTrainDiv2 = pad_sequences(token2.texts_to_sequences(xTrainDiv2), maxlen=350)

    for obj in range(10000):
        if yTrain3[obj] == 1:
            yTrainDiv3.append([1, 0, 0, 0, 0])
        elif yTrain3[obj] == 2:
            yTrainDiv3.append([0, 1, 0, 0, 0])
        elif yTrain3[obj] == 3:
            yTrainDiv3.append( [0, 0, 1, 0, 0])
        elif yTrain3[obj] == 4:
            yTrainDiv3.append([0, 0, 0, 1, 0])
        elif yTrain3[obj] == 5:
            yTrainDiv3.append([0, 0, 0, 0, 1])
        xTrainDiv3.append(xTrain3[obj])
    xTrainDiv3 = np.array(xTrainDiv3)
    yTrainDiv3 = np.array(yTrainDiv3)
    token3 = Tokenizer(num_words=5000)
    token3.fit_on_texts(xTrainDiv3)
    xTrainDiv3 = pad_sequences(token3.texts_to_sequences(xTrainDiv3), maxlen=600)


    print('-'*300)
    if int(input('Choose:\n    0-Generate new model\n    1-Load previous model\n0/1: ')) == 1:
        smallModel = load('smallPhoneReviewRu')
        mediumModel = load('mediumPhoneReviewRu')
        bigModel = load('bigPhoneReviewRu')
        while True:
            r = input('Enter your prompt: ').lower().replace('[\.,1234567890!?()\-\$%"\']', '')
            if len(r) < 150:
                print("Small text prediction")
                print(r)
                result = smallModel.predict(pad_sequences(token1.texts_to_sequences(r), maxlen=150))
            elif len(r) > 150 and len(r) < 350:
                print("Medium text prediction")
                result = mediumModel.predict(pad_sequences(token2.texts_to_sequences(r), maxlen=350))
            elif len(r) > 300 and len(r) < 650:
                print("Big text prediction")
                result = bigModel.predict(pad_sequences(token3.texts_to_sequences(r), maxlen=600))
            print('Estimated estimate is: ', np.argmax(result[0])+1)
            print('-'*300)

    smallModel = Sequential()
    smallModel.add(Embedding(input_dim=5000, output_dim=200))
    smallModel.add(GRU(128))
    smallModel.add(Dense(128, activation='relu'))
    smallModel.add(Dense(64, activation='relu'))
    smallModel.add(Dense(32, activation='relu'))
    smallModel.add(Dense(5, activation='sigmoid'))
    smallModel.compile(optimizer='adam', loss='categorical_crossentropy')
    smallModel.fit(xTrainDiv1, yTrainDiv1, batch_size=50, epochs=20)
    smallModel.save('smallPhoneReviewRu')

    mediumModel = Sequential()
    mediumModel.add(Embedding(input_dim=5000, output_dim=200))
    mediumModel.add(GRU(256))
    mediumModel.add(Dense(256, activation='relu'))
    mediumModel.add(Dense(128, activation='relu'))
    mediumModel.add(Dense(64, activation='relu'))
    mediumModel.add(Dense(32, activation='relu'))
    mediumModel.add(Dense(5, activation='sigmoid'))
    mediumModel.compile(optimizer='adam', loss='categorical_crossentropy')
    mediumModel.fit(xTrainDiv2, yTrainDiv2, batch_size=50, epochs=20)
    mediumModel.save('mediumPhoneReviewRu')

    bigModel = Sequential()
    bigModel.add(Embedding(input_dim=5000, output_dim=200))
    bigModel.add(GRU(512))
    bigModel.add(Dense(512, activation='relu'))
    bigModel.add(Dense(256, activation='relu'))
    bigModel.add(Dense(128, activation='relu'))
    bigModel.add(Dense(64, activation='relu'))
    bigModel.add(Dense(32, activation='relu'))
    bigModel.add(Dense(5, activation='sigmoid'))
    bigModel.compile(optimizer='adam', loss='categorical_crossentropy')
    bigModel.fit(xTrainDiv3, yTrainDiv3, batch_size=50, epochs=20)
    bigModel.save('bigPhoneReviewRu')