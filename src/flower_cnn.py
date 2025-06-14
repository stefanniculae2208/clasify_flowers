import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


class FlowerCNN:
    def __init__(self, input_shape=(128, 128, 3), num_classes=102):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train_images = []
        self.train_labels = []
        self.validation_images = []
        self.validation_labels = []
        self.test_images = []
        self.test_labels = []
        self.model = []

    
    def create_model(self, optimizer='adam', dropout_rate=0.5, learning_rate=0.001, use_pre_trained_model=True):
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

        base_model = None
        if use_pre_trained_model:
            base_model = keras.Sequential([
                MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape),
                layers.GlobalAveragePooling2D()
            ])
            base_model.layers[0].trainable = False
        else:
            base_model = keras.Sequential([
                layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                
                layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                
                layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                
                layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                
                layers.Flatten()
            ])

        model = models.Sequential([
            keras.Input(shape=self.input_shape),
            layers.Rescaling(1./255),
            data_augmentation,
            base_model,
            layers.BatchNormalization(),
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

    def set_validation_data(self, validation_images_memmap, validation_labels):
        self.validation_images = np.memmap(validation_images_memmap, dtype='float32', mode='r', shape=(len(validation_labels),) + self.input_shape)
        self.validation_labels = np.array(validation_labels)
        self.validation_labels = self.validation_labels - 1
    
    def train(self, epochs=10, batch_size=32):
        self.model.fit(self.train_images, self.train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    
    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.test_images, self.test_labels)
        print(f"Test Accuracy: {accuracy * 100}%")
        
        # Calculate confusion matrix
        cm = self.calculate_confusion_matrix()
        
        return accuracy, cm
        
        
    def calculate_classification_metrics(self, use_validation=False):
        """
        Manually calculate precision and recall for the classification task
        
        Args:
            use_validation (bool): Whether to use validation data instead of test data
            
        Returns:
            dict: Dictionary containing precision and recall metrics
        """
        if use_validation:
            if self.validation_images is None or self.validation_labels is None or self.model is None:
                print("Validation data or model not available")
                return None
            images = self.validation_images
            true_classes = self.validation_labels
        else:
            if self.test_images is None or self.test_labels is None or self.model is None:
                print("Test data or model not available")
                return None
            images = self.test_images
            true_classes = self.test_labels
            
        predictions = self.model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        true_positives = np.zeros(self.num_classes)
        false_positives = np.zeros(self.num_classes)
        false_negatives = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            true_positives[i] = np.sum((predicted_classes == i) & (true_classes == i))
            false_positives[i] = np.sum((predicted_classes == i) & (true_classes != i))
            false_negatives[i] = np.sum((predicted_classes != i) & (true_classes == i))

        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            if true_positives[i] + false_positives[i] > 0:
                precision[i] = true_positives[i] / (true_positives[i] + false_positives[i])
            else:
                precision[i] = 0
            if true_positives[i] + false_negatives[i] > 0:
                recall[i] = true_positives[i] / (true_positives[i] + false_negatives[i])
            else:
                recall[i] = 0
        
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        total_tp = np.sum(true_positives)
        total_fp = np.sum(false_positives)
        total_fn = np.sum(false_negatives)
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        dataset_type = "Validation" if use_validation else "Test"
        print(f"\nManual {dataset_type} Classification Metrics:")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Micro Precision: {micro_precision:.4f}")
        print(f"Micro Recall: {micro_recall:.4f}")
        print(f"Micro F1: {micro_f1:.4f}")
        
        metrics = {
            'classification_macro_precision': macro_precision,
            'classification_macro_recall': macro_recall,
            'classification_macro_f1': macro_f1,
            'classification_micro_precision': micro_precision,
            'classification_micro_recall': micro_recall,
            'classification_micro_f1': micro_f1
        }
        
        return metrics

    def calculate_confusion_matrix(self, use_validation=False):
        """
        Calculate the confusion matrix for the classification task
        
        Args:
            use_validation (bool): Whether to use validation data instead of test data
            
        Returns:
            numpy.ndarray: Confusion matrix where rows are true labels and columns are predicted labels
        """
        if use_validation:
            if self.validation_images is None or self.validation_labels is None or self.model is None:
                print("Validation data or model not available")
                return None
            images = self.validation_images
            true_classes = self.validation_labels
        else:
            if self.test_images is None or self.test_labels is None or self.model is None:
                print("Test data or model not available")
                return None
            images = self.test_images
            true_classes = self.test_labels
            
        predictions = self.model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        cm = np.zeros((102, 102), dtype=int)  # 102 flower classes
        for i in range(len(true_classes)):
            cm[true_classes[i]][predicted_classes[i]] += 1
        
        return cm