import keras.optimizers
import numpy as np
import csvReader as Cr
from keras.layers import Dense, GRU, Embedding
from keras.models import Sequential, load_model
from keras_preprocessing.text import Tokenizer
from random import randint

evaluation = Cr.readColumn("data.csv", 1, False)
reviews = Cr.readColumn("data.csv", 0, False)
token = Tokenizer(num_words=8000)
token.fit_on_texts(reviews)

reviews = token.texts_to_sequences(reviews)

xTrain = []
yTrain = []

i = 0
gi = 0

for review in reviews:
    wordIndexInSentence = 5
    for j in range(len(review)-5):
        wordIndex = review[wordIndexInSentence]
        xTrain.append([])
        xTrain[gi] = [review[wordIndexInSentence-5], review[wordIndexInSentence-4], review[wordIndexInSentence-3], review[wordIndexInSentence-2], review[wordIndexInSentence-1]]

        yTrain.append(np.zeros(len(token.word_index)))
        yTrain[gi][review[wordIndexInSentence]-1] = 1
        wordIndexInSentence += 1

        gi += 1
    i += 1

while True:
    usersChoice = input('Choose:\n    0-Generate new model\n    1-Load previous model\n0/1: ')
    if usersChoice == '1':
        model = load_model('PhoneReviewEng-Lang')
        if input("Choose:\n    0-Load previous model\n    1-Load previous model and check it's loss(need some time)\n0/1: ") == '1':
            xTrain = np.array(xTrain)
            yTrain = np.array(yTrain)
            print('loss: ', model.evaluate(xTrain, yTrain))
        while True:
            userInput = input('Prompt: ')
            numOfWords = int(input('Number of words: '))
            temp = int(input('Temperature(recommended value from 3 to 10): '))
            for i in range(numOfWords):
                userInput = userInput + ' ' + token.index_word[np.argsort(model.predict([[token.word_index[userInput.split(' ')[-5]], token.word_index[userInput.split(' ')[-4]], token.word_index[userInput.split(' ')[-3]], token.word_index[userInput.split(' ')[-2]], token.word_index[userInput.split(' ')[-1]]]])[0])[-randint(1, temp)]+1]
            print(userInput)
    elif usersChoice == '0':
        while True:
            usersChoice = input('Choose:\n    0-Use default settings\n    1-Manual configuration\n0/1: ')
            if usersChoice == '1':
                usersLearningRate = float(input('learning rate: '))
                usersEpochs = int(input('number of epochs: '))
                usersBatchSize = int(input('batch size: '))
                break
            elif usersChoice == '0':
                usersLearningRate = 0.0001
                usersEpochs = 100
                usersBatchSize = 100
                break
            print('Your answer is not valid, try again!')

        xTrain = np.array(xTrain)
        yTrain = np.array(yTrain)

        model = Sequential()
        model.add(Embedding(input_dim=8904, output_dim=150))
        model.add(GRU(256, return_sequences=True))
        model.add(GRU(128))
        model.add(Dense(8904, activation='softmax'))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=usersLearningRate), loss='categorical_crossentropy')
        model.fit(xTrain, yTrain, epochs=usersEpochs, batch_size=usersBatchSize)
        model.save('PhoneReviewEng-Lang')