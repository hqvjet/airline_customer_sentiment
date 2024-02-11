from keras.layers import Input, Bidirectional, LSTM, Dense, GlobalMaxPooling1D
from keras.models import Model
from tensorflow.keras.layers import Embedding, Average, Concatenate
from keras.layers import concatenate
from sklearn.metrics import classification_report
from tensorflow.keras import utils
import numpy as np
from constants import *
from keras.utils import plot_model
import matplotlib.pyplot as plt


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
        # Define input layers for the title and text inputs
        self.title_input = Input(shape=(self.train_title.shape[1],))
        self.text_input = Input(shape=(self.train_text.shape[1],))

        # Embedding layer for title
        title_embedding = Embedding(input_dim=self.vocab_size, output_dim=EMBEDDING_DIM, weights=[self.embedding_matrix], trainable=TRAINABLE)(self.title_input)
        # Embedding layer for text
        text_embedding = Embedding(input_dim=self.vocab_size, output_dim=EMBEDDING_DIM, weights=[self.embedding_matrix], trainable=TRAINABLE)(self.text_input)

        # Bidirectional LSTM layer for title
        title_bilstm = Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True))(title_embedding)
        # Bidirectional LSTM layer for text
        text_bilstm = Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True))(text_embedding)

        # Global Max Pooling layer for title
        title_pooling = GlobalMaxPooling1D()(title_bilstm)
        # Global Max Pooling layer for text
        text_pooling = GlobalMaxPooling1D()(text_bilstm)

        # Concatenate title and text pooling layers
        average_pooling = Average()([title_pooling, text_pooling])

        # Dense layer for final prediction
        output_layer = Dense(3, activation='softmax')(average_pooling)

        return output_layer

    def buildModel(self):
        # Build the model
        model_BiLSTM = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        model_BiLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_BiLSTM.summary()

        plot_model(model_BiLSTM, to_file=PATH + MODEL_IMAGE + BILSTM_IMAGE, show_shapes=True, show_layer_names=True)

        return model_BiLSTM

    def trainModel(self):
        history = self.model.fit(
            [np.array(self.train_title), np.array(self.train_text)],
            self.train_rating,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=([np.array(self.val_title), np.array(self.val_text)], self.val_rating)
        )

        self.model.save(PATH + MODEL + BILSTM_MODEL)

        # Plot training accuracy and loss values in the same plot
        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('BiLSTM Model')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(PATH + CHART + BILSTM_CHART)  # Lưu biểu đồ vào file
        plt.close()

        return self.model
    
    def testModel(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        pred = np.argmax(y_pred,axis=1)
        report = classification_report(y_test, utils.to_categorical(pred, num_classes=3))

        print(report)

        with open(PATH + REPORT + BILSTM_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + REPORT + BILSTM_REPORT}")
