from keras.layers import Input, Embedding, Conv2D, MaxPool2D, Flatten, Dense, Concatenate, Average, Bidirectional, LSTM, Reshape, Dropout
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
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
            embedding_matrix
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
        self.embedding_matrix = embedding_matrix
        self.output = self.getOutput() 
        self.model = self.buildModel()

    def getOutput(self):
        # Define input layers for the title and text inputs
        hidden_size = 256
        DROP = 0.3
        num_filters = 256
        filter_sizes = [3, 4, 5]

        self.title_input = Input(shape=(self.train_title.shape[1],))
        self.text_input = Input(shape=(self.train_text.shape[1],))
        title_embedding = Embedding(input_dim=self.vocab_size, output_dim=EMBEDDING_DIM, weights=[self.embedding_matrix], trainable=TRAINABLE)(self.title_input)
        text_embedding = Embedding(input_dim=self.vocab_size, output_dim=EMBEDDING_DIM, weights=[self.embedding_matrix], trainable=TRAINABLE)(self.text_input)
        title_bilstm = Bidirectional(LSTM(hidden_size, return_sequences=True))(title_embedding)
        text_bilstm = Bidirectional(LSTM(hidden_size, return_sequences=True))(text_embedding)
        title_reshape = Reshape((self.train_title.shape[1], hidden_size * 2, 1))(title_bilstm)
        text_reshape = Reshape((self.train_text.shape[1], hidden_size * 2, 1))(text_bilstm)

        title_conv_blocks = []
        for filter_size in filter_sizes:
            title_conv = Conv2D(num_filters, kernel_size=(filter_size, EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(title_reshape)
            title_pool = MaxPool2D(pool_size=(self.train_title.shape[1] - filter_size + 1, 1), strides=(1,1), padding='valid')(title_conv)
            # title_pool = GlobalMaxPooling1D()(title_conv)
            title_conv_blocks.append(title_pool)
        title_concat = Concatenate(axis=1)(title_conv_blocks)
        title_flat = Flatten()(title_concat)
        title_drop = Dropout(DROP)(title_flat)
        
        text_conv_blocks = []
        for filter_size in filter_sizes:
            text_conv = Conv2D(num_filters, kernel_size=(filter_size, EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(text_reshape)
            text_pool = MaxPool2D(pool_size=(self.train_text.shape[1] - filter_size + 1, 1), strides=(1,1), padding='valid')(text_conv)
            # text_pool = GlobalMaxPooling1D()(text_conv)
            text_conv_blocks.append(text_pool)
        text_concat = Concatenate(axis=1)(text_conv_blocks)
        text_flat = Flatten()(text_concat)
        text_drop = Dropout(DROP)(text_flat)

        average = Average()([title_drop, text_drop])

        dense1 = Dense(256, activation='relu')(average)

        return Dense(3, activation='softmax')(dense1)

    def buildModel(self):
        # Build the model
        cnn_bilstm_model = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        opt = Adam(learning_rate=0.00001)
        cnn_bilstm_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        cnn_bilstm_model.summary()

        plot_model(cnn_bilstm_model, to_file=PATH + MODEL_IMAGE + FUSION_CNN_BILSTM_IMAGE, show_shapes=True, show_layer_names=True)

        return cnn_bilstm_model

    def trainModel(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=STOP_PATIENCE, verbose=0, mode='min')
        checkpoint = ModelCheckpoint(PATH + MODEL + FUSION_CNN_BILSTM_MODEL, save_best_only=True, monitor='val_accuracy', mode='max')
        history = self.model.fit(
            [np.array(self.train_title), np.array(self.train_text)],
            self.train_rating,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=([np.array(self.val_title), np.array(self.val_text)], self.val_rating),
            callbacks=[early_stopping, checkpoint]
        )

        # self.model.save(PATH + MODEL + FUSION_CNN_BILSTM_MODEL)

        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('Fusion CNN + BiLSTM Model')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(PATH + CHART + FUSION_CNN_BILSTM_CHART)
        plt.close()
    
    def testModel(self, x_test, y_test):
        self.model = load_model(PATH + MODEL + FUSION_CNN_BILSTM_MODEL)
        y_pred = self.model.predict(x_test)
        pred = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, utils.to_categorical(pred, num_classes=3))
        acc = accuracy_score(y_test, utils.to_categorical(pred, num_classes=3))
        acc_line = f'Accuracy: {acc}\n'
        report += acc_line
        print(report)

        with open(PATH + REPORT + FUSION_CNN_BILSTM_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + REPORT + FUSION_CNN_BILSTM_REPORT}..................")
