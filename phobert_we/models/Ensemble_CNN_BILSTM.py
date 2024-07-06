from keras.layers import Input, Dense, Average, Concatenate
from keras.models import Model, load_model
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from phobert_we.constants import *
from tensorflow.keras import utils
from keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


class Ensemble_CNN_BILSTM:

    def __init__(
            self,
            train_text,
            train_rating,
            val_text,
            val_rating,
    ):
        self.text_input = None
        self.train_text = train_text
        self.train_rating = train_rating
        self.val_text = val_text
        self.val_rating = val_rating
        self.output = self.getOutput()
        self.model = self.buildModel()

    def getOutput(self):
        # Define input layers for the title and text inputs
        self.text_input = Input(shape=(self.train_text.shape[1], self.train_text.shape[2]))
            
        cnn = load_model(PATH + MODEL + CNN_MODEL)

        # Rename the first model
        cnn_renamed = Model(inputs=cnn.inputs, outputs=cnn.outputs, name='cnn_model')
        cnn_renamed.set_weights(cnn.get_weights())
        cnn_renamed.compile(optimizer=cnn.optimizer, loss=cnn.loss, metrics=cnn.metrics)

        # Load the second model
        bilstm = load_model(PATH + MODEL + BILSTM_MODEL)

        # Rename the second model
        bilstm_renamed = Model(inputs=bilstm.inputs, outputs=bilstm.outputs, name='bilstm_model')
        bilstm_renamed.set_weights(bilstm.get_weights())
        bilstm_renamed.compile(optimizer=bilstm.optimizer, loss=bilstm.loss, metrics=bilstm.metrics)

        # Get the predictions from the BiLSTM model
        lstm_predictions = bilstm_renamed(self.text_input)

        # Get the predictions from the CNN model
        cnn_predictions = cnn_renamed(self.text_input)

        # Add a dense layer
        dense_layer = Dense(128, activation='relu')(cnn_predictions)
        dense_layer = Dense(32, activation='relu')(dense_layer)

        # Add another dense layer for the final output
        return Dense(2, activation='softmax')(dense_layer)

    def buildModel(self):
        # Build the model
        cnn_bilstm_model = Model(inputs=self.text_input, outputs=self.output)

        opt = Adam(learning_rate=0.0001)
        cnn_bilstm_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        cnn_bilstm_model.summary()

        plot_model(cnn_bilstm_model, to_file=PATH + MODEL_IMAGE + ENSEMBLE_CNN_BILSTM_IMAGE, show_shapes=True, show_layer_names=True)

        return cnn_bilstm_model

    def trainModel(self):
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=STOP_PATIENCE, verbose=1, mode='max')
        checkpoint = ModelCheckpoint(PATH + MODEL + ENSEMBLE_CNN_BILSTM_MODEL, save_best_only=True, monitor='val_accuracy', mode='max')
        history = self.model.fit(
            np.array(self.train_text),
            self.train_rating,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=(np.array(self.val_text), self.val_rating),
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
        report = classification_report(y_test, utils.to_categorical(pred, num_classes=2))
        acc = accuracy_score(y_test, utils.to_categorical(pred, num_classes=2))
        acc_line = f'Accuracy: {acc}\n'
        report += acc_line
        print(report)

        with open(PATH + REPORT + ENSEMBLE_CNN_BILSTM_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + REPORT + ENSEMBLE_CNN_BILSTM_REPORT}..................")
