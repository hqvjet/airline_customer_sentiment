from keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from keras.models import Model, load_model
from tensorflow.keras.layers import Embedding, Average, Concatenate
from keras.layers import concatenate
from sklearn.metrics import classification_report, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import utils
import numpy as np
import tensorflow as tf
from constants import *
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import Callback
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
        hidden_size = 256
        DROP = 0.5

        self.title_input = Input(shape=(self.train_title.shape[1],))
        title_embedding = Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.train_title.shape[1], weights=[self.embedding_matrix], trainable=TRAINABLE)(self.title_input)
        title_lstm = Bidirectional(LSTM(hidden_size))(title_embedding)
        title_lstm = Dropout(DROP)(title_lstm)

        self.text_input = Input(shape=(self.train_text.shape[1],))
        text_embedding = Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.train_text.shape[1], weights=[self.embedding_matrix], trainable=TRAINABLE)(self.text_input)
        text_lstm = Bidirectional(LSTM(hidden_size))(text_embedding)
        text_lstm = Dropout(DROP)(text_lstm)

        average = Average()([title_lstm, text_lstm])

        final = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(average)
        final = Dropout(DROP)(final)

        return Dense(3, activation='softmax')(final)

    def buildModel(self):
        # Build the model
        model_BiLSTM = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        model_BiLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_BiLSTM.summary()

        plot_model(model_BiLSTM, to_file=PATH + MODEL_IMAGE + BILSTM_IMAGE, show_shapes=True, show_layer_names=True)

        return model_BiLSTM

    def trainModel(self):
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=STOP_PATIENCE, verbose=0, mode='max')
        checkpoint = ModelCheckpoint(PATH + MODEL + BILSTM_MODEL, save_best_only=True, monitor='val_loss', mode='min')
        gradient_callback = PrintGradientCallback()
        history = self.model.fit(
            [np.array(self.train_title), np.array(self.train_text)],
            self.train_rating,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=([np.array(self.val_title), np.array(self.val_text)], self.val_rating),
            callbacks=[early_stopping, checkpoint, gradient_callback]
        )

        # self.model.save(PATH + MODEL + BILSTM_MODEL)

        # Plot training accuracy and loss values in the same plot
        plt.figure()
        plt.plot(history.history['val_accuracy'], label='Accuracy')
        plt.plot(history.history['val_loss'], label='Loss')
        plt.title('BiLSTM Model')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(PATH + CHART + BILSTM_CHART)  # Lưu biểu đồ vào file
        plt.close()

        return self.model
    
    def testModel(self, x_test, y_test):
        self.model = load_model(PATH + MODEL + BILSTM_MODEL)
        y_pred = self.model.predict(x_test)
        pred = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, utils.to_categorical(pred, num_classes=3))
        acc = accuracy_score(y_test, utils.to_categorical(pred, num_classes=3))
        acc_line = f'Accuracy: {acc}\n'
        report += acc_line
        print(report)

        with open(PATH + REPORT + BILSTM_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + REPORT + BILSTM_REPORT}..................")


class PrintGradientCallback(Callback):

  def on_epoch_end(self, epoch, logs=None):
    
    gradients = K.gradients(self.model.total_loss, self.model.trainable_weights)
    mean_grad = K.mean(K.stack([K.mean(grad) for grad in gradients]))
    print("Epoch {}: Mean Gradient = {}".format(epoch + 1, mean_grad))
