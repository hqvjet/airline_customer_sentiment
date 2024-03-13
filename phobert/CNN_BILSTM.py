from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, concatenate, Average
from keras.models import Model, load_model
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from constants import *
from tensorflow.keras import utils
from keras.utils import plot_model
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
            cnn_model,
            bilstm_mode
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
        self.cnn_model = cnn_model
        self.bilstm_model = bilstm_mode
        self.output = self.getOutput() 
        self.model = self.buildModel()

    def getOutput(self):
      # Define input layers for the title and text inputs
      self.title_input = Input(shape=(self.train_title.shape[1],))
      self.text_input = Input(shape=(self.train_text.shape[1],))

      # Get the predictions from the BiLSTM model
      bilstm_predictions = self.bilstm_model([self.title_input, self.text_input])

      # Get the predictions from the CNN model
      cnn_predictions = self.cnn_model([self.title_input, self.text_input])

      # Average predictions
      average_predictions = Average()([bilstm_predictions, cnn_predictions])

      # Add a dense layer
      dense_layer = Dense(EMBEDDING_DIM, activation='relu')(average_predictions)

      # Add another dense layer for the final output
      return Dense(3, activation='softmax')(dense_layer)

    def buildModel(self):
        # Build the model
        cnn_bilstm_model = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        cnn_bilstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        cnn_bilstm_model.summary()
        
        plot_model(cnn_bilstm_model, to_file=PATH + MODEL_IMAGE + CNN_BILSTM_IMAGE, show_shapes=True, show_layer_names=True)

        return cnn_bilstm_model

    def trainModel(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=STOP_PATIENCE, verbose=0, mode='min')
        checkpoint = ModelCheckpoint(PATH + MODEL + CNN_BILSTM_MODEL, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=LR_PATIENCE, verbose=1, epsilon=1e-4, mode='min')
        history = self.model.fit(
            [np.array(self.train_title), np.array(self.train_text)],
            self.train_rating,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=([np.array(self.val_title), np.array(self.val_text)], self.val_rating),
            callbacks=[early_stopping, checkpoint, reduce_lr_loss]
        )

        # self.model.save(PATH + CNN_BILSTM_MODEL)

        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('CNN + BiLSTM Model')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(PATH + CHART + CNN_BILSTM_CHART)
        plt.close()
    
    def testModel(self, x_test, y_test):
        self.model = load_model(PATH + MODEL + CNN_BILSTM_MODEL)
        y_pred = self.model.predict(x_test)
        pred = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, utils.to_categorical(pred, num_classes=3))
        acc = accuracy_score(y_test, utils.to_categorical(pred, num_classes=3))
        acc_line = f'Accuracy: {acc}\n'
        report += acc_line
        print(report)

        with open(PATH + REPORT + CNN_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + REPORT + CNN_BILSTM_REPORT}..................")