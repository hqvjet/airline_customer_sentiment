# from tensorflow import ops
import tensorflow as tf
from keras.layers import Input, GlobalAveragePooling1D, Dense, Dropout, LayerNormalization, MultiHeadAttention, Layer, Concatenate, Embedding
from keras.models import Model, load_model
import keras
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from constants import *
from tensorflow.keras import utils
from keras.utils import plot_model
import matplotlib.pyplot as plt


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class Transformer():
    def __init__(self,
            train_title,
            train_text,
            train_rating,
            val_title,
            val_text,
            val_rating,
            vocab_size,
            emb_mat
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
        self.word_emb = emb_mat
        self.output = self.getOutput() 
        self.model = self.buildModel()

    def getOutput(self):
        DROP = 0.1

        self.title_input = Input(shape=(self.train_title.shape[1], ))
        title_embedding = Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.train_title.shape[1], weights=[self.word_emb], trainable=TRAINABLE)(self.title_input)
        transformer_block_title = TransformerBlock(EMBEDDING_DIM, 8, EMBEDDING_DIM)
        x_title = transformer_block_title(title_embedding)
        x_title = GlobalAveragePooling1D()(x_title)
        x_title = Dropout(DROP)(x_title)

        self.text_input = Input(shape=(self.train_text.shape[1], ))
        text_embedding = Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.train_text.shape[1], weights=[self.word_emb], trainable=TRAINABLE)(self.text_input)
        transformer_block_text = TransformerBlock(EMBEDDING_DIM, 8, EMBEDDING_DIM)
        x_text = transformer_block_text(text_embedding)
        x_text = GlobalAveragePooling1D()(x_text)
        x_text = Dropout(DROP)(x_text)
        
        concat = Concatenate(axis=-1)([x_title, x_text])

        # x = Dense(768, activation='relu')(concat)
        # x = Dense(512, activation="relu")(x)
        x = Dense(128, activation='relu')(concat)
        x = Dense(64, activation="relu")(x)

        return Dense(3, activation="softmax")(x)

    def buildModel(self):
        # Build the model
        model_transformer = Model(inputs=[self.title_input, self.text_input], outputs=self.output)
        opt = Adam(learning_rate=0.00001)
        model_transformer.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model_transformer.summary()
        
        plot_model(model_transformer, to_file=PATH + MODEL_IMAGE + TRANSFORMER_IMAGE, show_shapes=True, show_layer_names=True)

        return model_transformer

    def trainModel(self):
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=STOP_PATIENCE, verbose=0, mode='max')
        checkpoint = ModelCheckpoint(PATH + MODEL + TRANSFORMER_MODEL, save_best_only=True, monitor='val_accuracy', mode='max')
        history = self.model.fit(
            [np.array(self.train_title), np.array(self.train_text)],
            self.train_rating,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=([np.array(self.val_title), np.array(self.val_text)], self.val_rating),
            callbacks=[early_stopping, checkpoint]
        )

        # self.model.save(PATH + MODEL + CNN_MODEL)

        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('Transformer Model')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(PATH + CHART + TRANSFORMER_CHART)
        plt.close()

        return self.model
    
    def testModel(self, x_test, y_test):
        self.model = load_model(PATH + MODEL + TRANSFORMER_MODEL, custom_objects={'TransformerBlock': TransformerBlock})
        y_pred = self.model.predict(x_test)
        pred = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, utils.to_categorical(pred, num_classes=3))
        acc = accuracy_score(y_test, utils.to_categorical(pred, num_classes=3))
        acc_line = f'Accuracy: {acc}\n'
        report += acc_line
        print(report)

        with open(PATH + REPORT + TRANSFORMER_REPORT, 'w') as file:
            print(report, file=file)

        print(f"Classification report saved to {PATH + REPORT + TRANSFORMER_REPORT}..................")

