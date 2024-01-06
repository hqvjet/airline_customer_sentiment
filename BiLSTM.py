from keras.layers import Input, Bidirectional, LSTM, Dense, GlobalMaxPooling1D
from keras.models import Model
from tensorflow.keras.layers import Embedding
from keras.layers import concatenate
import numpy as np


class BiLSTM:

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
        # Define input layers for the title and text inputs
        self.title_input = Input(shape=(self.train_title.shape[1],))
        self.text_input = Input(shape=(self.train_text.shape[1],))

        # Embedding layer for title
        title_embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, trainable=True)(self.title_input)
        # Embedding layer for text
        text_embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, trainable=True)(self.text_input)

        # Bidirectional LSTM layer for title
        title_bilstm = Bidirectional(LSTM(64, return_sequences=True))(title_embedding)
        # Bidirectional LSTM layer for text
        text_bilstm = Bidirectional(LSTM(64, return_sequences=True))(text_embedding)

        # Global Max Pooling layer for title
        title_pooling = GlobalMaxPooling1D()(title_bilstm)
        # Global Max Pooling layer for text
        text_pooling = GlobalMaxPooling1D()(text_bilstm)

        # Concatenate title and text pooling layers
        concatenated_pooling = concatenate([title_pooling, text_pooling])

        # Dense layer for final prediction
        output_layer = Dense(5, activation='softmax')(concatenated_pooling)

        return output_layer

        # # Create model
        # model = Model(inputs=[title_input, text_input], outputs=output_layer)

        # return model

    def buildModel(self):
        # Build the model
        model_BiLSTM = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        model_BiLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_BiLSTM.summary()

        return model_BiLSTM

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
