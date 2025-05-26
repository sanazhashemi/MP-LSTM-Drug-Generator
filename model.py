from keras.models import Model
from keras.layers import Input, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, concatenate

def build_mp_lstm(input_shape, output_dim):
    # Input layer
    input_layer = Input(shape=input_shape)

    # Path 1: Direct LSTM
    path1 = LSTM(64, activation='relu', return_sequences=True)(input_layer)
    path1 = GlobalMaxPooling1D()(path1)

    # Path 2: Conv1D + LSTM
    path2 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)
    path2 = MaxPooling1D(pool_size=2)(path2)
    path2 = LSTM(64, activation='relu', return_sequences=True)(path2)
    path2 = GlobalMaxPooling1D()(path2)

    # Path 3: Deep Conv1D + LSTM
    path3 = Conv1D(32, kernel_size=5, activation='relu', padding='same')(input_layer)
    path3 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(path3)
    path3 = MaxPooling1D(pool_size=2)(path3)
    path3 = LSTM(64, activation='relu', return_sequences=True)(path3)
    path3 = GlobalMaxPooling1D()(path3)

    # Merge paths
    merged = concatenate([path1, path2, path3], axis=-1)

    # Dense layers
    dense = Dense(128, activation='relu')(merged)
    dropout = Dropout(0.3)(dense)
    output_layer = Dense(output_dim, activation='softmax')(dropout)

    # Build model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model
