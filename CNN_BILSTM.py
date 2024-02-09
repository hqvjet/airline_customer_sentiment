from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, concatenate, Average
from keras.models import Model
import numpy as np
from sklearn.metrics import classification_report
from constants import *
from tensorflow.keras import utils
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
      lstm_predictions = self.bilstm_model([self.title_input, self.text_input])

      # Get the predictions from the CNN model
      cnn_predictions = self.cnn_model([self.title_input, self.text_input])

      # Average predictions
      average_predictions = Average()([lstm_predictions, cnn_predictions])

      # Add a dense layer
      dense_layer = Dense(EMBEDDING_DIM, activation='relu')(average_predictions)

      # Add another dense layer for the final output
      return Dense(3, activation='softmax')(dense_layer)

    def buildModel(self):
        # Build the model
        cnn_bilstm_model = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        cnn_bilstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        cnn_bilstm_model.summary()

        return cnn_bilstm_model

    def trainModel(self):
        history = self.model.fit(
            [np.array(self.train_title), np.array(self.train_text)],
            self.train_rating,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=([np.array(self.val_title), np.array(self.val_text)], self.val_rating),
        )

        self.model.save(PATH + CNN_BILSTM_MODEL)

        # Plot training accuracy and loss values in the same plot
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('Model Train Accuracy and Loss')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(PATH + 'BiLSTM_chart.png')  # Lưu biểu đồ vào file
        # plt.show()
    
    def testModel(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        pred = np.argmax(y_pred,axis=1)
        report = classification_report(y_test, utils.to_categorical(pred, num_classes=3))
        print(report)

        with open(PATH + CNN_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + CNN_REPORT}..................")