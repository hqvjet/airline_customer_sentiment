from keras.models import Model
from tensorflow.keras.layers import Embedding
from keras.layers import Input, Embedding, LSTM, Dropout, Dense, concatenate
import numpy as np


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
        embedding_dim,
        epoch,
        batch_size
    ):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.title_input = None
        self.text_input = None
        self.train_title = train_title
        self.train_text = train_text
        self.train_rating = train_rating
        self.val_title = val_title
        self.val_text = val_text
        self.val_rating = val_rating
        self.output = self.getOutput()

    def getOutput(self):
        hidden_size = 256

        # Đầu vào cho title
        title_input = Input(shape=(self.train_title.shape[1],))
        title_embedding = Embedding(self.vocab_size, self.embedding_dim, input_length=self.train_title.shape[1])(title_input)
        title_lstm = LSTM(hidden_size, return_sequences=True)(title_embedding)
        title_lstm_dropout = Dropout(0.2)(title_lstm)
        title_lstm_final = LSTM(hidden_size)(title_lstm_dropout)

        # Đầu vào cho text
        text_input = Input(shape=(self.train_text.shape[1],))
        text_embedding = Embedding(self.vocab_size, self.embedding_dim, input_length=self.train_text.shape[1])(text_input)
        text_lstm = LSTM(hidden_size, return_sequences=True)(text_embedding)
        text_lstm_dropout = Dropout(0.2)(text_lstm)
        text_lstm_final = LSTM(hidden_size)(text_lstm_dropout)

        # Kết hợp hai đầu vào
        combined = concatenate([title_lstm_final, text_lstm_final])

        # Các bước còn lại của mô hình
        dense1 = Dense(128, activation='relu')(combined)
        output = Dense(self.train_rating.shape[1], activation='softmax')(dense1)

        return output

    def buildModel(self):

        # Xây dựng mô hình
        model_LSTM = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_LSTM.summary()

        return model_LSTM

    def trainModel(
        self,
        model,
        epoch,
        batch_size
    ):

        history = model.fit(
            [np.array(self.train_title), np.array(self.train_text)],
            self.train_rating,
            epochs=epoch,
            batch_size=batch_size,
            verbose=1,
            validation_data=([np.array(self.val_title), np.array(self.val_text)], self.val_rating)
        )

        return history
