import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Input
from tensorflow.keras import Model

def build_lstm_model(model_type, input_shape):
    """
    Construye y compila el modelo Deep Learning
    Tipos: 'stacked', 'bidirectional', 'attention'
    """
    inputs = Input(shape=input_shape)

    if model_type == 'stacked':
        x = LSTM(64, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(32)(x)
        x = Dropout(0.2)(x)

    elif model_type == 'bidirectional':
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(32))(inputs)
        x = Dropout(0.2)(x)

    elif model_type == 'attention':
        lstm_out = LSTM(64, return_sequences=True)(inputs)
        query = Dense(64)(lstm_out)
        value = Dense(64)(lstm_out)
        attention_layer = Attention()([query, value])
        
        x = tf.keras.layers.GlobalAveragePooling1D()(attention_layer)
        x = Dense(32)(x)

    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


