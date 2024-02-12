from keras.models import Model
from tensorflow.keras.layers import Embedding
from keras.layers import Input, Embedding, LSTM as LSTM_model, Dense, Average
from tensorflow.keras import utils
from sklearn.metrics import classification_report
import numpy as np
from keras.utils import plot_model
import matplotlib.pyplot as plt
from constants import *


class LSTM:

    def __init__(
        self,
        train_title,
        train_text,
        train_rating,
        val_title,
        val_text,
        val_rating,
        vocab_size,
        embedding_matrix
    ):
        self.vocab_size = vocab_size
        self.title_input = None
        self.text_input = None
        self.train_title = train_title
        self.train_text = train_text
        self.train_rating = train_rating
        self.val_title = val_title
        self.val_text = val_text
        self.val_rating = val_rating
        self.embedding_matrix = embedding_matrix
        self.output = self.getOutput()
        self.model = self.buildModel()

    def getOutput(self):
        hidden_size = 256

        self.title_input = Input(shape=(self.train_title.shape[1],))
        title_embedding = Embedding(self.vocab_size, EMBEDDING_DIM, weights=[self.embedding_matrix], input_length=self.train_title.shape[1], trainable=False)(self.title_input)
        title_lstm = LSTM_model(hidden_size)(title_embedding)

        self.text_input = Input(shape=(self.train_text.shape[1],))
        text_embedding = Embedding(self.vocab_size, EMBEDDING_DIM, weights=[self.embedding_matrix], input_length=self.train_text.shape[1], trainable=False)(self.text_input)
        text_lstm = LSTM_model(hidden_size)(text_embedding)

        average = Average()([title_lstm, text_lstm])

        # dense1 = Dense(512, activation='relu')(combined)
        return Dense(self.train_rating.shape[1], activation='softmax')(average)

    def buildModel(self):

        # Xây dựng mô hình
        model_LSTM = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_LSTM.summary()
        plot_model(model_LSTM, to_file=PATH + MODEL_IMAGE + LSTM_IMAGE, show_shapes=True, show_layer_names=True)
        
        return model_LSTM

    def trainModel(self):

        history = self.model.fit(
            [np.array(self.train_title), np.array(self.train_text)],
            self.train_rating,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=([np.array(self.val_title), np.array(self.val_text)], self.val_rating)
        )

        self.model.save(PATH + MODEL + LSTM_MODEL)

        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('LSTM Model')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(PATH + CHART + LSTM_CHART)
        plt.close()

    def testModel(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        pred = np.argmax(y_pred,axis=1)
        report = classification_report(y_test, utils.to_categorical(pred, num_classes=3))
        print(report)

        with open(PATH + REPORT + LSTM_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + REPORT + LSTM_REPORT}..................")
