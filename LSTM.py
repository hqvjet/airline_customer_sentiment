from keras.models import Model
from tensorflow.keras.layers import Embedding
from keras.layers import Input, Embedding, LSTM as LSTM_model, Dropout, Dense, concatenate
from tensorflow.keras import utils
from sklearn.metrics import classification_report
import numpy as np
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
        vocab_size
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
        self.output = self.getOutput()
        self.model = self.buildModel()

    def getOutput(self):
        hidden_size = 256

        # Đầu vào cho title
        self.title_input = Input(shape=(self.train_title.shape[1],))
        title_embedding = Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.train_title.shape[1])(self.title_input)
        title_lstm = LSTM_model(hidden_size, return_sequences=True)(title_embedding)
        title_lstm_dropout = Dropout(0.2)(title_lstm)
        title_lstm_final = LSTM_model(hidden_size)(title_lstm_dropout)

        # Đầu vào cho text
        self.text_input = Input(shape=(self.train_text.shape[1],))
        text_embedding = Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.train_text.shape[1])(self.text_input)
        text_lstm = LSTM_model(hidden_size, return_sequences=True)(text_embedding)
        text_lstm_dropout = Dropout(0.2)(text_lstm)
        text_lstm_final = LSTM_model(hidden_size)(text_lstm_dropout)

        # Kết hợp hai đầu vào
        combined = concatenate([title_lstm_final, text_lstm_final])

        # Các bước còn lại của mô hình
        dense1 = Dense(512, activation='relu')(combined)
        return Dense(self.train_rating.shape[1], activation='softmax')(dense1)

    def buildModel(self):

        # Xây dựng mô hình
        model_LSTM = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_LSTM.summary()

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

        self.model.save(PATH + LSTM_MODEL)

    def testModel(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        pred = np.argmax(y_pred,axis=1)
        report = classification_report(y_test, utils.to_categorical(pred, num_classes=3))

        print(report)

        with open(PATH + LSTM_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + LSTM_REPORT}...................")