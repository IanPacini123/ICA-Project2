from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf 
import numpy as np

def create_model():
    # Define the model architecture
    model = Sequential()

    # Add a convolutional layer with 32 filters, 3x3 kernel size, and relu activation function
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    # Add a batch normalization layer
    model.add(BatchNormalization())
    # Add a second convolutional layer with 64 filters, 3x3 kernel size, and relu activation function
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # Add a second batch normalization layer
    model.add(BatchNormalization())
    # Add a max pooling layer with 2x2 pool size
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add a dropout layer with 0.25 dropout rate
    model.add(Dropout(0.25))

    # Add a third convolutional layer with 128 filters, 3x3 kernel size, and relu activation function
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    # Add a third batch normalization layer
    model.add(BatchNormalization())
    # Add a fourth convolutional layer with 128 filters, 3x3 kernel size, and relu activation function
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    # Add a fourth batch normalization layer
    model.add(BatchNormalization())
    # Add a max pooling layer with 2x2 pool size
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add a dropout layer with 0.25 dropout rate
    model.add(Dropout(0.25))

    # Add a fifth convolutional layer with 256 filters, 3x3 kernel size, and relu activation function
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    # Add a fifth batch normalization layer
    model.add(BatchNormalization())
    # Add a sixth convolutional layer with 256 filters, 3x3 kernel size, and relu activation function
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    # Add a sixth batch normalization layer
    model.add(BatchNormalization())
    # Add a max pooling layer with 2x2 pool size
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add a dropout layer with 0.25 dropout rate
    model.add(Dropout(0.25))

    # Flatten the output of the convolutional layers
    model.add(Flatten())
    # Add a dense layer with 256 neurons and relu activation function
    model.add(Dense(256, activation='relu'))
    # Add a seventh batch normalization layer
    model.add(BatchNormalization())
    # Add a dropout layer with 0.5 dropout rate
    model.add(Dropout(0.5))
    # Add a dense layer with 7 neurons (one for each class) and softmax activation function
    model.add(Dense(7, activation='softmax'))

    # Compile the model with categorical cross-entropy loss, adam optimizer, and accuracy metric
    model.compile(loss="categorical_crossentropy", optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), metrics=['accuracy'])

    return model

class Cnn_emotion_predictor:

    def __init__(self) -> None:
        self.model = create_model()
        self.model.load_weights(r'C:\Users\Ianpa\Desktop\VSCode\ICA-Project2\emotion_detection\model_weights.h5')

    def predict(self, image):
        predictions = self.model.predict(image)

        predicted_class_index = np.argmax(predictions)
        class_labels = ['Raiva', 'Nojo', 'Medo', 'Felicidade', 'Neutro', 'Tristeza', 'Surpresa']
        predicted_class_label = class_labels[predicted_class_index]

        print(f'Predicted Class: {predicted_class_label}')