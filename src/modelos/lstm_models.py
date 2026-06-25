import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Input
from tensorflow.keras import Model

def build_lstm_model(model_type, input_shape, lstm_u1=64, lstm_u2=32, dropout_rate=0.2, dense_u=32):
    """Construye el modelo LSTM"""
    inputs = Input(shape=input_shape)

    if model_type == 'stacked':
        x = LSTM(lstm_u1, return_sequences=True)(inputs)
        x = Dropout(dropout_rate)(x)
        x = LSTM(lstm_u2)(x)
        x = Dropout(dropout_rate)(x)

    elif model_type == 'bidirectional':
        x = Bidirectional(LSTM(lstm_u1, return_sequences=True))(inputs)
        x = Dropout(dropout_rate)(x)
        x = Bidirectional(LSTM(lstm_u2))(inputs)
        x = Dropout(dropout_rate)(x)

    elif model_type == 'attention':
        lstm_out = LSTM(lstm_u1, return_sequences=True)(inputs)
        query = Dense(lstm_u1)(lstm_out)
        value = Dense(lstm_u1)(lstm_out)
        attention_layer = Attention()([query, value])
        
        x = tf.keras.layers.GlobalAveragePooling1D()(attention_layer)
        x = Dense(dense_u)(x)

    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


