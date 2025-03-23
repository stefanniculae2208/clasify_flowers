import numpy as np
# from sklearn.model_selection import GridSearchCV
# from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
import itertools
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

from src.flower_dataset import FlowerDataset
from src.flower_cnn import FlowerCNN


def find_best_model():
    input_shape=(224, 224, 3)
    dataset = FlowerDataset("./data/102flowers/jpg", "./utils/imagelabels.mat", "./utils/setid.mat", input_shape)
    dataset.load_data()
    dataset.summary()

    flower_cnn = FlowerCNN(input_shape)
    flower_cnn.set_training_data(dataset.train_images_memmap, dataset.train_labels)
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
    batch_sizes = [16]
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

        # Use StratifiedKFold for cross-validation
        # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        # fold_accuracies = []
        # for train_idx, val_idx in cv.split(flower_cnn.train_images, flower_cnn.train_labels):
        #     X_train, X_val = flower_cnn.train_images[train_idx], flower_cnn.train_images[val_idx]
        #     y_train, y_val = flower_cnn.train_labels[train_idx], flower_cnn.train_labels[val_idx]
        #     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)
        #     predictions = model.predict(X_val, batch_size=batch_size, verbose=0)
        #     predictions = np.argmax(predictions, axis=1)
        #     fold_accuracy = accuracy_score(y_val, predictions)
        #     fold_accuracies.append(fold_accuracy)
        # avg_accuracy = np.mean(fold_accuracies)
        # print(f"Average accuracy for this combination: {avg_accuracy}")

        model.fit(flower_cnn.train_images, flower_cnn.train_labels, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)
        predictions = model.predict(flower_cnn.test_images, batch_size=batch_size, verbose=0)
        predictions = np.argmax(predictions, axis=1)
        avg_accuracy = accuracy_score(flower_cnn.test_labels, predictions)
        print(f"Accuracy on test data: {avg_accuracy:.4f}")

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

    flower_cnn.model.save(f".data/models/best_model_{best_accuracy*100}%.keras")
    print(f"Model saved as '.data/models/best_model_{best_accuracy*100}%.keras'")
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy}")


if __name__ == "__main__":
    find_best_model()