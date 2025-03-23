import os
import numpy as np
import scipy.io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array


class FlowerDataset:
    def __init__(self, image_dir, labels_path, setid_path, input_shape=(128, 128, 3)):
        self.image_dir = image_dir
        self.labels_path = labels_path
        self.setid_path = setid_path
        self.input_shape = input_shape
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.train_images_memmap = './utils/train_images.dat'
        self.test_images_memmap = './utils/test_images.dat'


    def _preprocess_image(self, img):
        return img_to_array(tf.image.resize(img, (self.input_shape[0], self.input_shape[1])))


    def load_data(self):
        # Labels come in the '.mat file' and are of 3 types test, train and validation.
        labels = scipy.io.loadmat(self.labels_path)["labels"].flatten()
        setid = scipy.io.loadmat(self.setid_path)
        train_indices = setid["trnid"].flatten() - 1  # MATLAB index starts from 1
        test_indices = setid["tstid"].flatten() - 1
        val_indices = setid["valid"].flatten() - 1

        image_files = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]

        # I don't have enough RAM to load the files directly into memory so I have to uses a memmap.
        self.train_images = np.memmap(self.train_images_memmap, dtype='float32', mode='w+', shape=(len(train_indices),) + self.input_shape)
        self.test_images = np.memmap(self.test_images_memmap, dtype='float32', mode='w+', shape=(len(test_indices),) + self.input_shape)
        
        # Pre-allocate label lists with correct size
        self.train_labels = [0] * len(train_indices)
        self.test_labels = [0] * len(test_indices)
        
        train_nr = 0
        test_nr = 0
        validation_nr = 0
        
        for img_file in image_files:
            idx = int(img_file[6:11]) - 1  # Extract numeric index from filename
            img_path = os.path.join(self.image_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            img = self._preprocess_image(np.array(img))
            
            if idx in train_indices:
                print(f"Train at {train_nr}")
                self.train_images[train_nr] = np.array(img)
                self.train_labels[train_nr] = labels[idx]
                train_nr += 1
            elif idx in test_indices:
                print(f"Test at {test_nr}")
                self.test_images[test_nr] = np.array(img)
                self.test_labels[test_nr] = labels[idx]
                test_nr += 1
            elif idx in val_indices:
                validation_nr += 1

        # Convert labels to numpy arrays
        # self.train_labels = np.array(self.train_labels)
        # self.test_labels = np.array(self.test_labels)
        
        # Verify data integrity
        assert len(self.train_images) == len(self.train_labels), "Training images and labels count mismatch"
        assert len(self.test_images) == len(self.test_labels), "Test images and labels count mismatch"
        assert train_nr == len(train_indices), f"Expected {len(train_indices)} training samples, got {train_nr}"
        assert test_nr == len(test_indices), f"Expected {len(test_indices)} test samples, got {test_nr}"
        
        print(f"Train: {train_nr}\nTest: {test_nr}\nValidation: {validation_nr}\nTotal: {len(image_files)} vs {test_nr+train_nr+validation_nr}")
        
    def summary(self):
        print(f"Total Training Images: {len(self.train_images)}")
        print(f"Total Testing Images: {len(self.test_images)}")