import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


class FlowerCNN:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.model = []

    
    def create_model(self, optimizer='adam', dropout_rate=0.5, learning_rate=0.001):

        def bilinear_pooling(x):
            # x shape: [batch, height, width, channels]
            shape = tf.shape(x)
            batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
            # Reshape to [batch, height*width, channels]
            x = tf.reshape(x, [batch_size, height * width, channels])
            # Compute the bilinear pooling (outer product) per sample
            phi_I = tf.matmul(x, x, transpose_a=True) / tf.cast(height * width, tf.float32)
            # Flatten the bilinear feature matrix into a vector
            phi_I = tf.reshape(phi_I, [batch_size, channels * channels])
            # Apply signed square-root
            y = tf.sign(phi_I) * tf.sqrt(tf.abs(phi_I) + 1e-12)
            # L2 normalization
            y = tf.nn.l2_normalize(y, axis=-1)
            return y
        

        if optimizer == 'adam':
            optimizer_instance = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optimizer_instance = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optimizer_instance = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'adamax':
            optimizer_instance = keras.optimizers.Adamax(learning_rate=learning_rate)
        elif optimizer == 'nadam':
            optimizer_instance = keras.optimizers.Nadam(learning_rate=learning_rate)
        elif optimizer == 'adamw':
            optimizer_instance = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.1*learning_rate)
        else:
            raise ValueError(f"Optimizer '{optimizer}' is not supported")

        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2)
        ])

        # data_augmentation = keras.Sequential([
        #     layers.RandomFlip("horizontal"),
        #     layers.RandomFlip("vertical"),
        #     layers.RandomRotation(0.4),
        #     layers.RandomZoom(0.2),
        #     layers.RandomContrast(0.2),
        #     layers.RandomTranslation(0.1, 0.1),
        # ])

        # model = keras.Sequential([
        #     keras.Input(shape=self.input_shape),
        #     layers.Rescaling(1./255),
        #     data_augmentation,
        #     layers.Conv2D(32, (3, 3), activation='relu'),
        #     layers.MaxPooling2D((2, 2)),
        #     layers.BatchNormalization(),
        #     layers.Conv2D(64, (3, 3), activation='relu'),
        #     layers.MaxPooling2D((2, 2)),
        #     layers.BatchNormalization(),
        #     layers.Conv2D(128, (3, 3), activation='relu'),
        #     layers.MaxPooling2D((2, 2)),
        #     layers.BatchNormalization(),
        #     layers.Flatten(),
        #     layers.Dense(128, 
        #               activation='relu',
        #               kernel_regularizer=keras.regularizers.l2(0.01),  
        #               bias_regularizer=keras.regularizers.l2(0.01)),
        #     layers.BatchNormalization(),
        #     layers.Dropout(dropout_rate),
        #     layers.Dense(102, activation='softmax')
        # ])

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.trainable = False  # Freeze the base model layers

        model = models.Sequential([
            keras.Input(shape=self.input_shape),
            layers.Rescaling(1./255),
            data_augmentation,
            base_model, 
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            # layers.Lambda(bilinear_pooling), 
            layers.Dense(128, 
                      activation='relu',
                      kernel_regularizer=keras.regularizers.l2(0.01),  
                      bias_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(102, activation='softmax')  # Output layer with 102 units for classification
        ])

        model.compile(optimizer=optimizer_instance, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    

    def set_training_data(self, train_images_memmap, train_labels):
        self.train_images = np.memmap(train_images_memmap, dtype='float32', mode='r', shape=(len(train_labels),) + self.input_shape)
        self.train_labels = np.array(train_labels)
        self.train_labels = self.train_labels - 1

    
    def set_test_data(self, test_images_memmap, test_labels):
        self.test_images = np.memmap(test_images_memmap, dtype='float32', mode='r', shape=(len(test_labels),) + self.input_shape)
        self.test_labels = np.array(test_labels)
        self.test_labels = self.test_labels - 1
    
    
    def train(self, epochs=10, batch_size=32):
        self.model.fit(self.train_images, self.train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    
    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.test_images, self.test_labels)
        print(f"Test Accuracy: {accuracy * 100}%")
        return accuracy