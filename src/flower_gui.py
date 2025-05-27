import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tensorflow import keras
from PIL import Image, ImageTk
import numpy as np

class FlowerClassifierGUI:
    def __init__(self, find_best_model_fn, classify_image_fn, classify_image_siamese_fn=None, find_best_siamese_model_fn=None):
        self.root = tk.Tk()
        self.root.title("Flower Classification")
        self.root.geometry("1600x1600")
        self.model = None
        self.siamese_model = None
        self.find_best_model_fn = find_best_model_fn
        self.classify_image_fn = classify_image_fn
        self.classify_image_siamese_fn = classify_image_siamese_fn
        self.find_best_siamese_model_fn = find_best_siamese_model_fn
        self.current_image_path = None
        self.photo_image = None
        self.segmentation_photo = None  
        self.similar_photos = []  
        self.setup_gui()

    def setup_gui(self):
        self.frame = ttk.Frame(self.root, padding="20")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        find_model_btn = ttk.Button(self.frame, text="Find Best Model", command=self.find_best_model_fn)
        find_model_btn.grid(row=0, column=0, pady=10, padx=10, sticky=tk.EW)
        
        find_siamese_btn = ttk.Button(self.frame, text="Find Best Siamese Model", command=self.find_best_siamese_model_fn)
        find_siamese_btn.grid(row=1, column=0, pady=10, padx=10, sticky=tk.EW)
        
        load_model_btn = ttk.Button(self.frame, text="Load Classification Model", command=self.load_model)
        load_model_btn.grid(row=2, column=0, pady=10, padx=10, sticky=tk.EW)
        
        load_siamese_btn = ttk.Button(self.frame, text="Load Siamese Model", command=self.load_siamese_model)
        load_siamese_btn.grid(row=3, column=0, pady=10, padx=10, sticky=tk.EW)
        
        self.classify_btn = ttk.Button(self.frame, text="Classify Image", command=self.classify_image, state="disabled")
        self.classify_btn.grid(row=4, column=0, pady=10, padx=10, sticky=tk.EW)
        
        self.model_label = ttk.Label(self.frame, text="No model loaded", wraplength=350)
        self.model_label.grid(row=5, column=0, pady=10, padx=10)
        
        image_container = ttk.Frame(self.frame)
        image_container.grid(row=6, column=0, pady=10, padx=10, sticky=tk.NSEW)
        image_container.columnconfigure(0, weight=1)
        image_container.columnconfigure(1, weight=1)
        
        self.image_frame = ttk.Frame(image_container, borderwidth=2, relief="groove")
        self.image_frame.grid(row=0, column=0, pady=10, padx=10, sticky=tk.NSEW)
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        
        self.segmentation_frame = ttk.Frame(image_container, borderwidth=2, relief="groove")
        self.segmentation_frame.grid(row=0, column=1, pady=10, padx=10, sticky=tk.NSEW)
        self.segmentation_label = ttk.Label(self.segmentation_frame)
        self.segmentation_label.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        
        similar_container = ttk.Frame(self.frame)
        similar_container.grid(row=7, column=0, pady=10, padx=10, sticky=tk.EW)
        
        similar_title = ttk.Label(similar_container, text="Similar Flowers (Same Class):", font=("Arial", 10, "bold"))
        similar_title.pack(pady=(5, 10), anchor=tk.W)
        
        similar_images_frame = ttk.Frame(similar_container)
        similar_images_frame.pack(fill=tk.BOTH, expand=True)
        
        self.similar_labels = []
        for i in range(3):
            frame = ttk.Frame(similar_images_frame, borderwidth=1, relief="groove", width=150, height=150)
            frame.pack(side=tk.LEFT, padx=10, pady=5)
            frame.pack_propagate(False)
            label = tk.Label(frame)
            label.pack(expand=True, fill=tk.BOTH)
            self.similar_labels.append(label)
        
        self.result_label = ttk.Label(self.frame, text="", wraplength=350)
        self.result_label.grid(row=8, column=0, pady=10, padx=10)
        
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(6, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def load_model(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("Keras Model", "*.keras"), ("All Files", "*.*")],
                initialdir="data/models"
            )
            
            if file_path:
                self.model = keras.models.load_model(file_path)
                self.siamese_model = None
                model_name = os.path.basename(file_path)
                self.model_label.config(text=f"Loaded classification model: {model_name}")
                self.classify_btn.config(state="normal")
                self.result_label.config(text="")
                self.clear_images()
                messagebox.showinfo("Success", "Classification model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model = None
            self.model_label.config(text="No model loaded")
            self.classify_btn.config(state="disabled")
            self.result_label.config(text="")
            
    def load_siamese_model(self):
        try:
            if self.classify_image_siamese_fn is None:
                messagebox.showerror("Error", "Siamese model functionality is not available")
                return
                
            file_path = filedialog.askopenfilename(
                title="Select Siamese Model File",
                filetypes=[("Keras Model", "*.keras"), ("All Files", "*.*")],
                initialdir="data/models"
            )
            
            if file_path:
                self.siamese_model = keras.models.load_model(file_path)
                self.model = None  # Reset classification model when loading siamese model
                model_name = os.path.basename(file_path)
                self.model_label.config(text=f"Loaded siamese model: {model_name}")
                self.classify_btn.config(state="normal")
                self.result_label.config(text="")
                self.clear_images()
                messagebox.showinfo("Success", "Siamese model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load siamese model: {str(e)}")
            self.siamese_model = None
            self.model_label.config(text="No model loaded")
            self.classify_btn.config(state="disabled")
            self.result_label.config(text="")

    def clear_images(self):
        """Clear all image displays"""
        self.image_label.config(image=None)
        self.segmentation_label.config(image=None)
        self.photo_image = None
        self.segmentation_photo = None
        self.current_image_path = None
        
        for label in self.similar_labels:
            label.config(image=None)
        self.similar_photos = []

    def display_image(self, image_path):
        try:
            img = Image.open(image_path)
            
            max_width = 400
            max_height = 300
            width, height = img.size
            
            if width > height:
                new_width = max_width
                new_height = int(height * (max_width / width))
            else:
                new_height = max_height
                new_width = int(width * (max_height / height))
                
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            self.photo_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo_image)
            self.current_image_path = image_path
            self.segmentation_label.config(image=None)
            self.segmentation_photo = None
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
            self.image_label.config(image=None)
            self.segmentation_label.config(image=None)
            self.current_image_path = None

    def display_segmentation_overlay(self, original_image, segmentation_mask):
        """Display segmentation mask overlay on the right panel"""
        try:
            if isinstance(original_image, Image.Image):
                original_np = np.array(original_image)
            else:
                original_np = original_image
                
            if len(segmentation_mask.shape) == 3 and segmentation_mask.shape[2] == 1:
                seg_mask = np.squeeze(segmentation_mask)
            else:
                seg_mask = segmentation_mask
                
            height, width = seg_mask.shape
            overlay = np.zeros((height, width, 3), dtype=np.uint8)
            overlay[seg_mask > 0.5] = [255, 0, 0]  # Red for foreground
            original_resized = Image.fromarray(original_np).resize((width, height), Image.Resampling.LANCZOS)
            original_resized_np = np.array(original_resized)
            alpha = 0.7
            blended = np.uint8(original_resized_np * alpha + overlay * (1 - alpha))
            overlay_img = Image.fromarray(blended)
            max_width = 400
            max_height = 300
            width, height = overlay_img.size
            if width > height:
                new_width = max_width
                new_height = int(height * (max_width / width))
            else:
                new_height = max_height
                new_width = int(width * (max_height / height))
                
            overlay_img = overlay_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.segmentation_photo = ImageTk.PhotoImage(overlay_img)
            self.segmentation_label.config(image=self.segmentation_photo)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display segmentation overlay: {str(e)}")
            self.segmentation_label.config(image=None)

    def display_similar_flowers(self, similar_paths):
        """Display similar flower images in the bottom row
        
        Args:
            similar_paths: List of paths to similar flower images
        """
        for label in self.similar_labels:
            label.config(image=None)
        self.similar_photos = []
        for i, img_path in enumerate(similar_paths[:3]):
            if i >= len(self.similar_labels):
                break
            try:
                img = Image.open(img_path)
                img = img.resize((140, 140), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.similar_photos.append(photo)
                self.similar_labels[i].config(image=photo)
            except Exception as e:
                print(f"Error displaying similar image {i}: {str(e)}")

    def classify_image(self):
        if not self.model and not self.siamese_model:
            messagebox.showerror("Error", "No model loaded!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image to Classify",
            filetypes=[("JPEG files", "*.jpg"), ("All Files", "*.*")],
            initialdir="data/102flowers/jpg"
        )
        
        if file_path:
            self.display_image(file_path)
            
            try:
                if self.model:
                    result, similar_paths = self.classify_image_fn(self.model, file_path)
                    self.result_label.config(text=result)
                    self.segmentation_label.config(image=None)
                    self.segmentation_photo = None
                    if similar_paths and len(similar_paths) > 0:
                        self.display_similar_flowers(similar_paths)
                    
                elif self.siamese_model:
                    result, segmentation_mask, similar_paths = self.classify_image_siamese_fn(self.siamese_model, file_path)
                    self.result_label.config(text=result)
                    orig_img = Image.open(file_path)
                    self.display_segmentation_overlay(orig_img, segmentation_mask)
                    if similar_paths and len(similar_paths) > 0:
                        self.display_similar_flowers(similar_paths)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to classify image: {str(e)}")
                self.result_label.config(text="Classification failed")
                print(f"Error details: {str(e)}")  # Print more detailed error information

    def run(self):
        self.root.mainloop()
