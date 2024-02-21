from keras.models import Model, load_model
from keras import regularizers
from tensorflow.keras.layers import Embedding
from keras.layers import Input, Embedding, LSTM as LSTM_model, Dropout, Dense, concatenate, Average
from tensorflow.keras import utils
from sklearn.metrics import classification_report, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from constants import *
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt


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
        DROP = 0.3

        self.title_input = Input(shape=(self.train_title.shape[1],))
        title_embedding = Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.train_title.shape[1], trainable=TRAINABLE)(self.title_input)
        title_lstm = LSTM_model(hidden_size)(title_embedding)
        title_lstm = Dropout(DROP)(title_lstm)

        self.text_input = Input(shape=(self.train_text.shape[1],))
        text_embedding = Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.train_text.shape[1], trainable=TRAINABLE)(self.text_input)
        text_lstm = LSTM_model(hidden_size)(text_embedding)
        text_lstm = Dropout(DROP)(text_lstm)

        average = Average()([title_lstm, text_lstm])

        final = Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(average)
        final = Dropout(DROP)(final)

        return Dense(3, activation='softmax')(final)

    def buildModel(self):

        # Xây dựng mô hình
        model_LSTM = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        class_weights = compute_class_weight('balanced', classes=np.unique(self.train_rating), y=self.train_rating)
        model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], class_weight=class_weights)
        model_LSTM.summary()
        
        plot_model(model_LSTM, to_file=PATH + MODEL_IMAGE + LSTM_IMAGE, show_shapes=True, show_layer_names=True)

        return model_LSTM

    def trainModel(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=STOP_PATIENCE, verbose=0, mode='min')
        checkpoint = ModelCheckpoint(PATH + MODEL + LSTM_MODEL, save_best_only=True, monitor='val_accuracy', mode='max')
        history = self.model.fit(
            [np.array(self.train_title), np.array(self.train_text)],
            self.train_rating,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=([np.array(self.val_title), np.array(self.val_text)], self.val_rating),
            callbacks=[early_stopping, checkpoint]
        )

        # self.model.save(PATH + LSTM_MODEL)

        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('LSTM MODEL')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(PATH + CHART + LSTM_CHART)
        plt.close()

    def testModel(self, x_test, y_test):
        self.model = load_model(PATH + MODEL + LSTM_MODEL)
        y_pred = self.model.predict(x_test)
        pred = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, utils.to_categorical(pred, num_classes=3))
        acc = accuracy_score(y_test, utils.to_categorical(pred, num_classes=3))
        acc_line = f'Accuracy: {acc}\n'
        report += acc_line
        print(report)

        with open(PATH + REPORT + LSTM_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + REPORT + LSTM_REPORT}..................")
