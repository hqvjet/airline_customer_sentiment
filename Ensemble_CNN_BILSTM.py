from keras.layers import Input, Dense, Average
from keras.models import Model, load_model
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from constants import *
from tensorflow.keras import utils
from keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


class CNN_BILSTM:

    def __init__(
            self,
            train_title,
            train_text,
            train_rating,
            val_title,
            val_text,
            val_rating,
            vocab_size,
            bilstm,
            cnn
    ):
        self.title_input = None
        self.text_input = None
        self.train_title = train_title
        self.train_text = train_text
        self.train_rating = train_rating
        self.val_title = val_title
        self.val_text = val_text
        self.val_rating = val_rating
        self.vocab_size = vocab_size
        self.bilstm = bilstm
        self.cnn = cnn
        self.output = self.getOutput()
        self.model = self.buildModel()

    def getOutput(self):
        # Define input layers for the title and text inputs
        self.title_input = Input(shape=(self.train_title.shape[1],))
        self.text_input = Input(shape=(self.train_text.shape[1],))

        # Get the predictions from the BiLSTM model
        lstm_predictions = self.bilstm([self.title_input, self.text_input])

        # Get the predictions from the CNN model
        cnn_predictions = self.cnn([self.title_input, self.text_input])

        # Average predictions
        average_predictions = Average()([lstm_predictions, cnn_predictions])

        # Add a dense layer
        dense_layer = Dense(256, activation='relu')(average_predictions)

        # Add another dense layer for the final output
        return Dense(3, activation='softmax')(dense_layer)

    def buildModel(self):
        # Build the model
        cnn_bilstm_model = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        opt = Adam(learning_rate=0.00001)
        cnn_bilstm_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        cnn_bilstm_model.summary()

        plot_model(cnn_bilstm_model, to_file=PATH + MODEL_IMAGE + ENSEMBLE_CNN_BILSTM_IMAGE, show_shapes=True, show_layer_names=True)

        return cnn_bilstm_model

    def trainModel(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=STOP_PATIENCE, verbose=0, mode='min')
        checkpoint = ModelCheckpoint(PATH + MODEL + ENSEMBLE_CNN_BILSTM_MODEL, save_best_only=True, monitor='val_accuracy', mode='max')
        history = self.model.fit(
            [np.array(self.train_title), np.array(self.train_text)],
            self.train_rating,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=([np.array(self.val_title), np.array(self.val_text)], self.val_rating),
            callbacks=[early_stopping, checkpoint]
        )

        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('ENSEMBLE CNN + BiLSTM Model')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(PATH + CHART + ENSEMBLE_CNN_BILSTM_CHART)
        plt.close()
    
    def testModel(self, x_test, y_test):
        self.model = load_model(PATH + MODEL + ENSEMBLE_CNN_BILSTM_MODEL)
        y_pred = self.model.predict(x_test)
        pred = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, utils.to_categorical(pred, num_classes=3))
        acc = accuracy_score(y_test, utils.to_categorical(pred, num_classes=3))
        acc_line = f'Accuracy: {acc}\n'
        report += acc_line
        print(report)

        with open(PATH + REPORT + ENSEMBLE_CNN_BILSTM_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + REPORT + ENSEMBLE_CNN_BILSTM_REPORT}..................")
