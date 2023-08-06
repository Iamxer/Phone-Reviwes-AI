import keras.optimizers
import numpy as np
import csvReader as Cr
from keras.layers import Dense, GRU, Embedding
from keras.models import Sequential, load_model
from keras_preprocessing.text import Tokenizer

evaluation = Cr.readColumn("data.csv", 1, False)
reviews = Cr.readColumn("data.csv", 0, False)
token = Tokenizer(num_words=8900)
token.fit_on_texts(reviews)

reviews = token.texts_to_sequences(reviews)

xTrain = []
yTrain = []

i = 0
gi = 0

for review in reviews:
    wordIndexInSentence = 3
    for j in range(len(review) - 3):
        wordIndex = review[wordIndexInSentence]
 V        xTrain.append([])
        xTrain[gi] = [review[wordIndexInSentence - 3], review[wordIndexInSentence - 2], review[wordIndexInSentence - 1]]

        yTrain.append(np.zeros(len(token.word_index)))
        yTrain[gi][review[wordIndexInSentence] - 1] = 1
        wordIndexInSentence += 1

        gi += 1
    i += 1

while True:
    usersChoice = input('Choose:\n    0-Generate new model\n    1-Load previous model\n0/1: ')
    if usersChoice == '1':
        model = load_model('PhoneReviewEng-Lang')
        while True:
            userInput = input('Prompt: ')
            for i in range(50):
                userInput = userInput, token.index_word[np.argmax(model.predict([token.texts_to_sequences(
                    [userInput.split(' ')[-3], userInput.split(' ')[-2], userInput.split(' ')[-1]])])) + 1]
                userInput = ' '.join(userInput)
            print(userInput)
    elif usersChoice == '0':
        while True:
            usersChoice = input('Choose:\n    0-Use default settings\n    1-Manual configuration\n0/1: ')
            if usersChoice == '1':
                usersLearningRate = int(input('learning rate: '))
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

        print(len(token.word_index))
        print(len(xTrain))
        print(len(yTrain))

        model = Sequential()
        model.add(Embedding(input_dim=8904, output_dim=150))
        model.add(GRU(256))
        model.add(Dense(8904, activation='softmax'))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=usersLearningRate), loss='categorical_crossentropy')
        model.fit(xTrain, yTrain, epochs=usersEpochs, batch_size=usersBatchSize)
        model.save('PhoneReviewEng-Lang')