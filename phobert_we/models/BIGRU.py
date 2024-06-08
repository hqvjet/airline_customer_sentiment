from keras.layers import Input, Bidirectional, GRU, Dense, SpatialDropout1D, Dropout, GlobalAveragePooling1D
from keras.models import Model, load_model
from tensorflow.keras.layers import Embedding, Average, Concatenate
from keras.layers import concatenate
from sklearn.metrics import classification_report, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import utils
import numpy as np
from constants import *
from keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam


class BIGRU:

    def __init__(
        self,
        train_title,
        train_text,
        train_rating,
        val_title,
        val_text,
        val_rating,
    ):
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
        DROP = 0.2
        # Define input layers for the title and text inputs
        self.title_input = Input(shape=(self.train_title.shape[1], self.train_title.shape[2]))
        self.text_input = Input(shape=(self.train_text.shape[1], self.train_text.shape[2]))

        # Embedding layer for title
        # title_embedding = Embedding(input_dim=self.vocab_size, output_dim=EMBEDDING_DIM, trainable=TRAINABLE)(self.title_input)
        # Embedding layer for text
        # text_embedding = Embedding(input_dim=self.vocab_size, output_dim=EMBEDDING_DIM, trainable=TRAINABLE)(self.text_input)

        title_avg = SpatialDropout1D(DROP)(self.title_input)
        text_avg = SpatialDropout1D(DROP)(self.text_input)

        # Bidirectional GRU layer for title
        title_bilstm = Bidirectional(GRU(hidden_size, return_sequences=True))(title_avg)
        # Bidirectional GRU layer for text
        text_bilstm = Bidirectional(GRU(hidden_size, return_sequences=True))(text_avg)

        title_bilstm = Dropout(DROP)(title_bilstm)
        text_bilstm = Dropout(DROP)(text_bilstm)

        title_avg = GlobalAveragePooling1D()(title_bilstm)
        text_avg = GlobalAveragePooling1D()(text_bilstm)

        concatenated = Concatenate(axis=-1)([title_avg, text_avg])

        # Concatenate title and text pooling layers
        # average_pooling = Average()([title_bilstm, text_bilstm])
        dense = Dense(128, activation='relu')(concatenated)
        dense = Dense(64, activation='relu')(dense)
        dense = Dense(32, activation='relu')(dense)

        # Dense layer for final prediction
        output_layer = Dense(3, activation='softmax')(dense)

        return output_layer

    def buildModel(self):
        # Build the model
        model_BIGRU = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        opt = Adam(learning_rate=0.00001)
        model_BIGRU.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model_BIGRU.summary()

        plot_model(model_BIGRU, to_file=PATH + MODEL_IMAGE + BIGRU_IMAGE, show_shapes=True, show_layer_names=True)

        return model_BIGRU

    def trainModel(self):
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=STOP_PATIENCE, verbose=1, mode='max')
        checkpoint = ModelCheckpoint(PATH + MODEL + BIGRU_MODEL, save_best_only=True, monitor='val_accuracy', mode='max')
        history = self.model.fit(
            [np.array(self.train_title), np.array(self.train_text)],
            self.train_rating,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=([np.array(self.val_title), np.array(self.val_text)], self.val_rating),
            callbacks=[early_stopping, checkpoint]
        )

        # self.model.save(PATH + BIGRU_MODEL)

        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('BIGRU Model')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(PATH + CHART + BIGRU_CHART)
        plt.close()

        return self.model
    
    def testModel(self, x_test, y_test):
        self.model = load_model(PATH + MODEL + BIGRU_MODEL)
        y_pred = self.model.predict(x_test)
        pred = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, utils.to_categorical(pred, num_classes=3))
        acc = accuracy_score(y_test, utils.to_categorical(pred, num_classes=3))
        acc_line = f'Accuracy: {acc}\n'
        report += acc_line
        print(report)

        with open(PATH + REPORT + BIGRU_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + REPORT + BIGRU_REPORT}..................")
