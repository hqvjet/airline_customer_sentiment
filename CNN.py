from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Flatten, Dense, concatenate, Dropout, Average
from keras.models import Model
import numpy as np
from sklearn.metrics import classification_report
from constants import *
from tensorflow.keras import utils
from keras.utils import plot_model

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
        # Input for title
        num_filters = 128
        filter_sizes = [3, 4, 5]
        DROP = 0.3
        
        self.title_input = Input(shape=(self.train_title.shape[1],))
        title_embedding = Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.train_title.shape[1], trainable=TRAINABLE)(self.title_input)
        title_conv_blocks = []
        for filter_size in filter_sizes:
            title_conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(title_embedding)
            title_pool = GlobalMaxPooling1D(pool_size=self.train_title.shape[1] - filter_size + 1)(title_conv)
            title_conv_blocks.append(title_pool)
        title_concat = concatenate(title_conv_blocks, axis=-1)
        title_flat = Flatten()(title_concat)
        title_drop = Dropout(DROP)(title_flat)

        # Input for text
        self.text_input = Input(shape=(self.train_text.shape[1],))
        text_embedding = Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.train_text.shape[1], trainable=TRAINABLE)(self.text_input)
        text_conv_blocks = []
        for filter_size in filter_sizes:
            text_conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(text_embedding)
            text_pool = GlobalMaxPooling1D(pool_size=self.train_text.shape[1] - filter_size + 1)(text_conv)
            text_conv_blocks.append(text_pool)
        text_concat = concatenate(text_conv_blocks, axis=-1)
        text_flat = Flatten()(text_concat)
        text_drop = Dropout(DROP)(text_flat)

        # Average two inputs
        average = Average()([title_drop, text_drop])

        # Additional layers of the model
        dense1 = Dense(512, activation='relu')(average)

        return Dense(3, activation='softmax')(dense1)

    def buildModel(self):
        # Build the model
        model_CNN = Model(inputs=[self.title_input, self.text_input], outputs=self.output)

        model_CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_CNN.summary()
        
        plot_model(model_CNN, to_file='CNN.png', show_shapes=True, show_layer_names=True)

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
