from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, concatenate
from keras.models import Model
import numpy as np
from sklearn.metrics import classification_report
from constants import *
from tensorflow.keras import utils


class CNN:

    def __init__(
            self,
            train_title,
            train_text,
            train_rating,
            val_title,
            val_text,
            val_rating,
            vocab_size,
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
        self.output = self.getOutput() 
        self.model = self.buildModel()

    def getOutput(self):
      # Define input layers for the title and text inputs
      self.title_input = Input(shape=(self.train_title.shape[1],))
      self.text_input = Input(shape=(self.train_text.shape[1],))

      # Get the predictions from the BiLSTM model
      lstm_predictions = model_BiLSTM([self.title_input, self.text_input])

      # Get the predictions from the CNN model
      cnn_predictions = model_CNN([self.title_input, self.text_input])

      # Concatenate the predictions
      concatenated_predictions = Concatenate()([lstm_predictions, cnn_predictions])

      # Add a dense layer
      dense_layer = Dense(64, activation='relu')(concatenated_predictions)

      # Add another dense layer for the final output
      output_layer = Dense(5, activation='softmax')(dense_layer)

    def buildModel(self):
        # Build the model
        model_CNN = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        model_CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_CNN.summary()

        return model_CNN

    def trainModel(self):
        history = self.model.fit(
            [np.array(self.train_title), np.array(self.train_text)],
            self.train_rating,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=([np.array(self.val_title), np.array(self.val_text)], self.val_rating),
        )

        self.model.save(PATH + CNN_MODEL)
    
    def testModel(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        pred = np.argmax(y_pred,axis=1)
        report = classification_report(y_test, utils.to_categorical(pred, num_classes=3))
        print(report)

        with open(PATH + CNN_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + CNN_REPORT}..................")


def build_ensemble_model(model_BiLSTM, model_CNN):
    

    ensemble_model = Model(inputs=[title_input, text_input], outputs=output_layer)

    return ensemble_model


# Build the ensemble model
model_ensemble_bilstm_cnn = build_ensemble_model(model_BiLSTM, model_CNN)
model_ensemble_bilstm_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ensemble_bilstm_cnn.summary()

history = model_ensemble_bilstm_cnn.fit(
    [np.array(x_train_title_pad), np.array(x_train_text_pad)],
    y_train,
    epochs=EPOCH,
    batch_size=4,
    verbose=1,
    validation_data=([np.array(x_test_title_pad), np.array(x_test_text_pad)], y_test)
)