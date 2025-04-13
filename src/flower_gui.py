import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tensorflow import keras

class FlowerClassifierGUI:
    def __init__(self, find_best_model_fn, classify_image_fn):
        self.root = tk.Tk()
        self.root.title("Flower Classification")
        self.root.geometry("800x600")
        self.model = None
        self.find_best_model_fn = find_best_model_fn
        self.classify_image_fn = classify_image_fn
        self.setup_gui()

    def setup_gui(self):
        self.frame = ttk.Frame(self.root, padding="20")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        find_model_btn = ttk.Button(self.frame, text="Find Best Model", command=self.find_best_model_fn)
        find_model_btn.grid(row=0, column=0, pady=10, padx=10, sticky=tk.EW)
        load_model_btn = ttk.Button(self.frame, text="Load Model", command=self.load_model)
        load_model_btn.grid(row=1, column=0, pady=10, padx=10, sticky=tk.EW)
        self.classify_btn = ttk.Button(self.frame, text="Classify Image", command=self.classify_image, state="disabled")
        self.classify_btn.grid(row=2, column=0, pady=10, padx=10, sticky=tk.EW)
        
        self.model_label = ttk.Label(self.frame, text="No model loaded", wraplength=350)
        self.model_label.grid(row=3, column=0, pady=10, padx=10)
        self.result_label = ttk.Label(self.frame, text="", wraplength=350)
        self.result_label.grid(row=4, column=0, pady=10, padx=10)
        
        self.frame.columnconfigure(0, weight=1)

    def load_model(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("Keras Model", "*.keras"), ("All Files", "*.*")],
                initialdir="data/models"
            )
            
            if file_path:
                self.model = keras.models.load_model(file_path)
                model_name = os.path.basename(file_path)
                self.model_label.config(text=f"Loaded model: {model_name}")
                self.classify_btn.config(state="normal")
                self.result_label.config(text="")
                messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model = None
            self.model_label.config(text="No model loaded")
            self.classify_btn.config(state="disabled")
            self.result_label.config(text="")

    def classify_image(self):
        if not self.model:
            messagebox.showerror("Error", "No model loaded!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image to Classify",
            filetypes=[("JPEG files", "*.jpg"), ("All Files", "*.*")],
            initialdir="data/102flowers/jpg"
        )
        
        if file_path:
            try:
                result = self.classify_image_fn(self.model, file_path)
                self.result_label.config(text=f"Classification Result: {result}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to classify image: {str(e)}")
                self.result_label.config(text="Classification failed")

    def run(self):
        self.root.mainloop()
