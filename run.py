import os # Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
import itertools
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from PIL import Image

from src.flower_dataset import FlowerDataset
from src.flower_cnn import FlowerCNN
from src.flower_gui import FlowerClassifierGUI
from src.flower_siamese import FlowerSiamese


# Oxford 102 Flower Dataset class names
FLOWER_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold", 
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon", 
    "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower", 
    "peruvian lily", "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", 
    "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian", 
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist", "mexican aster", 
    "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip", 
    "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue", 
    "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy", 
    "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia", 
    "pink-yellow dahlia?", "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush", 
    "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", 
    "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", 
    "lotus", "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", 
    "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", 
    "ball moss", "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", 
    "blanket flower", "trumpet creeper", "blackberry lily"
]


def find_best_model():
    input_shape=(224, 224, 3)
    segmentation_dir="/home/stefan/Documents/clasify_flowers/utils/segmim/"
    dataset = FlowerDataset("./data/102flowers/jpg", "./utils/imagelabels.mat", "./utils/setid.mat", input_shape, segmentation_dir)
    dataset.load_data()
    dataset.summary()

    flower_cnn = FlowerCNN(input_shape)
    flower_cnn.set_training_data(dataset.train_images_memmap, dataset.train_labels)
    flower_cnn.set_validation_data(dataset.validation_images_memmap, dataset.validation_labels)
    flower_cnn.set_test_data(dataset.test_images_memmap, dataset.test_labels)

    # optimizers = ['adam', 'sgd', 'rmsprop', 'adamax', 'nadam']
    # learning_rates = [0.0001, 0.001, 0.01]
    # dropout_rates = [0.3, 0.5, 0.7]
    # epochs_list = [5, 10, 20]
    # batch_sizes = [16, 32, 64]

    optimizers = ['adamw']
    learning_rates = [0.001]
    dropout_rates = [0.5]
    epochs_list = [50]
    batch_sizes = [8]
    param_combinations = itertools.product(optimizers, learning_rates, dropout_rates, epochs_list, batch_sizes)

    early_stopping = EarlyStopping(
        monitor='accuracy',
        patience=5,       
        restore_best_weights=True
    )

    best_accuracy = 0
    best_params = {}

    # Iterate through all combinations of chosen parameters
    for optimizer, learning_rate, dropout_rate, epochs, batch_size in param_combinations:
        print(f"Testing combination: optimizer={optimizer}, learning_rate={learning_rate}, "
            f"dropout_rate={dropout_rate}, epochs={epochs}, batch_size={batch_size}")

        model = flower_cnn.create_model(optimizer=optimizer, dropout_rate=dropout_rate, learning_rate=learning_rate)

        model.fit(
            flower_cnn.train_images, 
            flower_cnn.train_labels, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(flower_cnn.validation_images, flower_cnn.validation_labels),
            callbacks=[early_stopping], 
            verbose=1
        )
        
        # Get validation accuracy for model selection
        val_predictions = model.predict(flower_cnn.validation_images, batch_size=batch_size, verbose=0)
        val_predictions = np.argmax(val_predictions, axis=1)
        avg_accuracy = accuracy_score(flower_cnn.validation_labels, val_predictions)
        print(f"Accuracy on validation data: {avg_accuracy}")

        # Update the best accuracy and parameters
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_params = {
                'optimizer': optimizer,
                'learning_rate': learning_rate,
                'dropout_rate': dropout_rate,
                'epochs': epochs,
                'batch_size': batch_size
            }
            flower_cnn.model = model

    flower_cnn.evaluate()

    flower_cnn.model.save(f"data/models/best_model_{best_accuracy*100}%.keras")
    print(f"Model saved as 'data/models/best_model_{best_accuracy*100}%.keras'")
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy}")


def classify_image(model, image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)
        img = tf.keras.preprocessing.image.img_to_array(
            tf.image.resize(img, (224, 224))
        )
        img = np.expand_dims(img, axis=0) 
        
        predictions = model.predict(img)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx] * 100
        flower_name = FLOWER_NAMES[class_idx]
        
        return f"{flower_name} (Confidence: {confidence:.2f}%)"
        
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def find_best_model_siamese():
    """
    Find the best siamese model for both classification and segmentation
    """
    print("Finding best siamese model for classification and segmentation...")
    input_shape = (224, 224, 3)
    segmentation_dir = "/home/stefan/Documents/clasify_flowers/utils/segmim/"
    dataset = FlowerDataset("./data/102flowers/jpg", "./utils/imagelabels.mat", "./utils/setid.mat", input_shape, segmentation_dir)
    dataset.load_data()
    dataset.summary()
    
    optimizers = ['adam']
    learning_rates = [0.001]
    dropout_rates = [0.5]
    epochs_list = [1]
    batch_sizes = [16]
    param_combinations = itertools.product(optimizers, learning_rates, dropout_rates, epochs_list, batch_sizes)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    best_combined_metric = 0
    best_params = {}
    best_model = None
    
    for optimizer, learning_rate, dropout_rate, epochs, batch_size in param_combinations:
        print(f"\nTraining with: optimizer={optimizer}, learning_rate={learning_rate}, "
              f"dropout_rate={dropout_rate}, epochs={epochs}, batch_size={batch_size}")
        
        siamese_model = FlowerSiamese(input_shape=input_shape)
        model = siamese_model.create_model(optimizer=optimizer, dropout_rate=dropout_rate, learning_rate=learning_rate)
        model.summary()
        siamese_model.set_training_data(dataset.train_images_memmap, dataset.train_labels, dataset.train_masks_memmap)
        siamese_model.set_validation_data(dataset.validation_images_memmap, dataset.validation_labels, dataset.validation_masks_memmap)
        siamese_model.set_test_data(dataset.test_images_memmap, dataset.test_labels, dataset.test_masks_memmap)
        
        try:
            train_outputs = {
                'classification_output': siamese_model.train_labels,
                'segmentation_output': siamese_model.train_masks
            }
            validation_outputs = {
                'classification_output': siamese_model.validation_labels,
                'segmentation_output': siamese_model.validation_masks
            }
            history = model.fit(
                siamese_model.train_images,
                train_outputs,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(
                    siamese_model.validation_images,
                    validation_outputs
                ) if siamese_model.validation_images is not None else None,
                callbacks=[early_stopping]
            )
            val_classification_acc = max(history.history['val_classification_output_accuracy']) if 'val_classification_output_accuracy' in history.history else 0
            val_segmentation_iou = max(history.history['val_segmentation_output_iou']) if 'val_segmentation_output_iou' in history.history else 0
            combined_metric = (0.5 * val_classification_acc + 0.5 * val_segmentation_iou)
            
            print(f"Validation classification accuracy: {val_classification_acc:.4f}")
            print(f"Validation segmentation IoU: {val_segmentation_iou:.4f}")
            print(f"Combined metric: {combined_metric:.4f}")
            if combined_metric > best_combined_metric:
                best_combined_metric = combined_metric
                best_params = {
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'dropout_rate': dropout_rate,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'val_classification_accuracy': val_classification_acc,
                    'val_segmentation_iou': val_segmentation_iou
                }
                best_model = siamese_model
        
        except Exception as e:
            print(f"Error during training: {str(e)}")
            continue
    
    if best_model is None:
        print("No successful model training. Please check the errors above.")
        return None
    
    print("\n\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    try:
        print("\nEvaluating the best model on test data...")
        metrics = best_model.evaluate()      
        os.makedirs("./data/models", exist_ok=True)
        best_model.model.save("./data/models/best_siamese_model.keras")
        print("Best siamese model saved to ./data/models/best_siamese_model.keras")  
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
    
    return best_model


if __name__ == "__main__":
    # app = FlowerClassifierGUI(find_best_model, classify_image)
    # app.run()
    find_best_model_siamese()