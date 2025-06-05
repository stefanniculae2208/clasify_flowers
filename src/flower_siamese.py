import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np

class FlowerSiamese:
    def __init__(self, input_shape=(128, 128, 3), num_classes=102):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
        self.train_images = None
        self.train_labels = None
        self.train_masks = None
        self.validation_images = None
        self.validation_labels = None
        self.validation_masks = None
        self.test_images = None
        self.test_labels = None
        self.test_masks = None
    
    def create_model(self, optimizer='adam', dropout_rate=0.5, learning_rate=0.001, use_pre_trained_model=True):
        """
        Create a siamese model with shared encoder for both classification and segmentation
        """
        if optimizer == 'adam':
            optimizer_instance = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optimizer_instance = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optimizer_instance = keras.optimizers.RMSprop(learning_rate=learning_rate)
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
        encoder_outputs = None
        
        if use_pre_trained_model:
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
            base_model.trainable = False  
            
            encoder_outputs = [
                base_model.get_layer('block_1_expand_relu').output,    # 64x64
                base_model.get_layer('block_3_expand_relu').output,    # 32x32
                base_model.get_layer('block_6_expand_relu').output,    # 16x16
                base_model.get_layer('block_13_expand_relu').output,   # 8x8
                base_model.output                                      # 4x4
            ]
        else:
            inputs = keras.Input(shape=self.input_shape)
            x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
            skip1 = x
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            skip2 = x
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
            skip3 = x
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
            skip4 = x
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            bottleneck = x
            
            base_model = keras.Model(inputs=inputs, outputs=bottleneck)
            encoder_outputs = [skip1, skip2, skip3, skip4, bottleneck]
        
        encoder_model = keras.Model(inputs=base_model.input, outputs=encoder_outputs)
        
        classification_branch = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(128, 
                      activation='relu',
                      kernel_regularizer=keras.regularizers.l2(0.01),  
                      bias_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ], name='classification_output')
        
        # Create segmentation branch (U-Net decoder)
        def create_segmentation_branch(bottleneck, skip1, skip2, skip3, skip4, use_pre_trained=True):
            x = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(bottleneck)
            x = layers.Concatenate()([x, skip4])
            x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            
            x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
            x = layers.Concatenate()([x, skip3])
            x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
            x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
            
            x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
            x = layers.Concatenate()([x, skip2])
            x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
            x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
            
            x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
            x = layers.Concatenate()([x, skip1])
            x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            
            # We need an extra upsampling step for the pre-trained model
            if use_pre_trained:
                x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
                x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
                x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
            
            # Output layer for segmentation - binary mask
            x = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='segmentation_output')(x)
            return x
        
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Rescaling(1./255)(inputs)
        x = data_augmentation(x)
        encoder_features = encoder_model(x)
        bottleneck = encoder_features[4]
        skip1, skip2, skip3, skip4 = encoder_features[0], encoder_features[1], encoder_features[2], encoder_features[3]
        classification_features = classification_branch(bottleneck)
        segmentation_output = create_segmentation_branch(bottleneck, skip1, skip2, skip3, skip4, use_pre_trained_model)
        model = keras.Model(
            inputs=inputs, 
            outputs={
                'classification_output': classification_features,
                'segmentation_output': segmentation_output
            }
        )
        
        model.compile(
            optimizer=optimizer_instance,
            loss={
                'classification_output': 'sparse_categorical_crossentropy',
                'segmentation_output': 'binary_crossentropy'
            },
            metrics={
                'classification_output': ['accuracy'],
                'segmentation_output': [
                    'accuracy',
                    tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='iou')
                ]
            },
            loss_weights={
                'classification_output': 1.0,
                'segmentation_output': 1.0
            }
        )
        self.model = model
        return model
    
    def set_training_data(self, train_images_memmap, train_labels, train_masks_memmap):
        """Load training data from memory-mapped files"""
        self.train_images = np.memmap(train_images_memmap, dtype='float32', mode='r', 
                                     shape=(len(train_labels),) + self.input_shape)
        self.train_labels = np.array(train_labels) - 1  # Adjust for 0-based indexing
        self.train_masks = np.memmap(train_masks_memmap, dtype='float32', mode='r', 
                                    shape=(len(train_labels), self.input_shape[0], self.input_shape[1], 1))
    
    def set_validation_data(self, validation_images_memmap, validation_labels, validation_masks_memmap):
        """Load validation data from memory-mapped files"""
        self.validation_images = np.memmap(validation_images_memmap, dtype='float32', mode='r', 
                                         shape=(len(validation_labels),) + self.input_shape)
        self.validation_labels = np.array(validation_labels) - 1  # Adjust for 0-based indexing
        self.validation_masks = np.memmap(validation_masks_memmap, dtype='float32', mode='r', 
                                       shape=(len(validation_labels), self.input_shape[0], self.input_shape[1], 1))
    
    def set_test_data(self, test_images_memmap, test_labels, test_masks_memmap):
        """Load test data from memory-mapped files"""
        self.test_images = np.memmap(test_images_memmap, dtype='float32', mode='r', 
                                    shape=(len(test_labels),) + self.input_shape)
        self.test_labels = np.array(test_labels) - 1  # Adjust for 0-based indexing
        self.test_masks = np.memmap(test_masks_memmap, dtype='float32', mode='r', 
                                  shape=(len(test_labels), self.input_shape[0], self.input_shape[1], 1))
    
    def train(self, epochs=10, batch_size=32):
        """Train the model with both classification and segmentation tasks"""
        history = self.model.fit(
            self.train_images, 
            {
                'classification_output': self.train_labels,
                'segmentation_output': self.train_masks
            },
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                self.validation_images, 
                {
                    'classification_output': self.validation_labels,
                    'segmentation_output': self.validation_masks
                }
            ) if self.validation_images is not None else None
        )
        
        return history
    
    def evaluate(self):
        """Evaluate the model on test data"""
        results = self.model.evaluate(
            self.test_images, 
            {
                'classification_output': self.test_labels,
                'segmentation_output': self.test_masks
            },
            verbose=1,
            return_dict=True 
        )
        print("\nEvaluation Results:")
        for name, value in results.items():
            print(f"{name}: {value}")     
        return results
    
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
        else:
            if self.test_images is None or self.test_labels is None or self.model is None:
                print("Test data or model not available")
                return None
        
        images = self.validation_images if use_validation else self.test_images
        true_classes = self.validation_labels if use_validation else self.test_labels

        predictions = self.model.predict(images)
        classification_predictions = predictions['classification_output']
        predicted_classes = np.argmax(classification_predictions, axis=1)
        num_classes = self.num_classes
        true_positives = np.zeros(num_classes)
        false_positives = np.zeros(num_classes)
        false_negatives = np.zeros(num_classes)
        
        for i in range(num_classes):
            true_positives[i] = np.sum((predicted_classes == i) & (true_classes == i))
            false_positives[i] = np.sum((predicted_classes == i) & (true_classes != i))
            false_negatives[i] = np.sum((predicted_classes != i) & (true_classes == i))

        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        
        for i in range(num_classes):
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
    
    def calculate_segmentation_metrics(self, use_validation=False):
        """
        Manually calculate precision and recall for the segmentation task
        
        Args:
            use_validation (bool): Whether to use validation data instead of test data
        
        Returns:
            dict: Dictionary containing segmentation precision and recall metrics
        """
        if use_validation:
            if self.validation_images is None or self.validation_masks is None or self.model is None:
                print("Validation data or model not available")
                return None
        else:
            if self.test_images is None or self.test_masks is None or self.model is None:
                print("Test data or model not available")
                return None
            
        images = self.validation_images if use_validation else self.test_images
        masks = self.validation_masks if use_validation else self.test_masks

        predictions = self.model.predict(images)
        segmentation_predictions = predictions['segmentation_output']
        threshold = 0.5
        predicted_masks = (segmentation_predictions > threshold).astype(np.float32)
        true_masks = masks
        total_true_positive = 0
        total_false_positive = 0
        total_false_negative = 0
        
        for i in range(len(predicted_masks)):
            pred_mask = predicted_masks[i]
            true_mask = true_masks[i]
            true_positive = np.sum((pred_mask > 0) & (true_mask > 0))
            false_positive = np.sum((pred_mask > 0) & (true_mask == 0))
            false_negative = np.sum((pred_mask == 0) & (true_mask > 0))
            total_true_positive += true_positive
            total_false_positive += false_positive
            total_false_negative += false_negative
        
        precision = total_true_positive / (total_true_positive + total_false_positive) if (total_true_positive + total_false_positive) > 0 else 0
        recall = total_true_positive / (total_true_positive + total_false_negative) if (total_true_positive + total_false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        dataset_type = "Validation" if use_validation else "Test"
        print(f"\n{dataset_type} Segmentation Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        metrics = {
            'segmentation_precision': precision,
            'segmentation_recall': recall,
            'segmentation_f1': f1,
        }
        
        return metrics
    
    def predict(self, image):
        """Make predictions for a single image"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        classification, segmentation = self.model.predict(image)
        predicted_class = np.argmax(classification[0])
        segmentation_mask = segmentation[0]
        
        return predicted_class, segmentation_mask
