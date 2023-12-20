from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf 
import numpy as np

def create_model():
    # Definindo arquitetura do modelo
    model = Sequential()

    # Camada convolucional, 32 filtros, kernel 3x3, função de ativação relu
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    # Camada de normalização
    model.add(BatchNormalization())
    # Camada convolucional, 64 filtros, kernel 3x3, função de ativação relu
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # Camada de normalização
    model.add(BatchNormalization())
    # Camada de Max Pooling tamanho 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Camada de descarte 20%
    model.add(Dropout(0.2))

    # Camada convolucional, 128 filtros, kernel 3x3, função de ativação relu
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    # Camada de normalização
    model.add(BatchNormalization())
    # Camada convolucional, 128 filtros, kernel 3x3, função de ativação relu
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    # Camada de normalização
    model.add(BatchNormalization())
    # Camada de Max Pooling tamanho 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Camada de descarte 20%
    model.add(Dropout(0.2))

    # Camada convolucional, 256 filtros, kernel 3x3, função de ativação relu
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    # Camada de normalização
    model.add(BatchNormalization())
    # Camada convolucional, 256 filtros, kernel 3x3, função de ativação relu
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    # Camada de normalização
    model.add(BatchNormalization())
    # Camada de Max Pooling tamanho 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Camada de descarte 20%
    model.add(Dropout(0.2))

    # Achatação após parte convolucional
    model.add(Flatten())
    # Camada densa com 256 neuronios, função de ativação relu
    model.add(Dense(256, activation='relu'))
    # Camada de normalização
    model.add(BatchNormalization())
    # Camada de descarte 50%
    model.add(Dropout(0.5))
    # Camada densa de saida, 7 neuronios seguindo o numero de classes finais, função de ativação softmax
    model.add(Dense(7, activation='softmax'))

    # Compilação do modelo com função perda "categorical_crossentropy", otimizador "adam", e metrica de acurácia
    model.compile(loss="categorical_crossentropy", optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), metrics=['accuracy'])

    return model

class Cnn_emotion_predictor:

    def __init__(self) -> None:
        self.model = create_model()
        self.model.load_weights(r'emotion_detection\model_weights.h5')

    def predict(self, image):
        predictions = self.model.predict(image)

        predicted_class_index = np.argmax(predictions)
        class_labels = ['Raiva', 'Nojo', 'Medo', 'Felicidade', 'Neutro', 'Tristeza', 'Surpresa']
        predicted_class_label = class_labels[predicted_class_index]

        print(f'Predicted Class: {predicted_class_label}')