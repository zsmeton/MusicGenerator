from keras import Sequential, optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, LSTM,  RepeatVector, TimeDistributed, BatchNormalization
from sklearn.model_selection import train_test_split

from prep_batch_loading import read_size_of_data

def create_note_generator(X_shape, out_length) -> Sequential:
    lstm_model = Sequential()
    lstm_model.add(LSTM(
        input_dim=X_shape[1],
        return_sequences=False,
        output_dim=256,
        activation='tanh',
        dropout=0.1,
        recurrent_dropout=0.1
    ))
    lstm_model.add(BatchNormalization(momentum=0.7))
    lstm_model.add(RepeatVector(out_length))
    lstm_model.add(LSTM(units=256,
                        return_sequences=True,
                        activation='tanh',
                        dropout=0.1,
                        recurrent_dropout=0.1))
    lstm_model.add(BatchNormalization(momentum=0.7))
    lstm_model.add(TimeDistributed(Dense(256, activation='relu')))
    lstm_model.add(TimeDistributed(Dense(128, activation='sigmoid')))
    return lstm_model


if __name__ == "__main__":
    note_generator = create_note_generator(read_size_of_data(), 50)
    note_generator.summary()
