import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from scipy.stats import spearmanr
import torch
from thingsvision.model_class import Model
from thingsvision.utils import data
from thingsvision.utils.data import transform as tr

# Define the main application class
class RSAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RSA Analysis GUI")
        self.geometry("800x600")

        # Initialize variables
        self.image_dirs = [None, None]
        self.features = [None, None]
        self.rdms = [None, None]

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        # Image directory selection
        self.dir_buttons = [None, None]
        self.dir_labels = [None, None]
        for i in range(2):
            self.dir_labels[i] = tk.Label(self, text=f"Image Directory {i+1}: Not selected")
            self.dir_labels[i].pack()
            self.dir_buttons[i] = tk.Button(self, text=f"Select Image Directory {i+1}",
                                            command=lambda idx=i: self.select_image_directory(idx))
            self.dir_buttons[i].pack()

        # Model selection
        self.model_var = tk.StringVar(self)
        self.model_var.set("resnet50")  # Default model
        self.model_label = tk.Label(self, text="Select Model for Feature Extraction:")
        self.model_label.pack()
        self.model_option = ttk.Combobox(self, textvariable=self.model_var,
                                         values=["resnet50", "alexnet", "vgg16"])
        self.model_option.pack()

        # Compare RDMs button
        self.compare_button = tk.Button(self, text="Compare RDMs", command=self.compare_rdms)
        self.compare_button.pack()

        # Results label
        self.results_label = tk.Label(self, text="")
        self.results_label.pack()

    def select_image_directory(self, idx):
        dir_name = filedialog.askdirectory()
        if dir_name:
            self.image_dirs[idx] = dir_name
            self.dir_labels[idx].config(text=f"Image Directory {idx+1}: {dir_name}")
            self.extract_features(idx)

    def extract_features(self, idx):
        # Extract features using selected model
        model_name = self.model_var.get()
        image_dir = self.image_dirs[idx]

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        model = Model(model_name, pretrained=True, device=device)
        
        # Create dataset and dataloader
        dataset = data.create_dataset(root=image_dir, transform=tr.default_transform())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        # Extract features
        features = []
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().detach().numpy())
        
        self.features[idx] = np.concatenate(features, axis=0)
        self.calculate_rdm(idx)

    def calculate_rdm(self, idx):
        # Calculate RDM
        features = self.features[idx]
        n_samples = features.shape[0]
        rdm = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                corr, _ = spearmanr(features[i], features[j])
                rdm[i, j] = 1 - corr
        
        self.rdms[idx] = rdm

    def compare_rdms(self):
        if self.rdms[0] is not None and self.rdms[1] is not None:
            rdm1_flat = self.rdms[0].flatten()
            rdm2_flat = self.rdms[1].flatten()
            corr, _ = spearmanr(rdm1_flat, rdm2_flat)
            self.results_label.config(text=f"Spearman Correlation: {corr}")
        else:
            self.results_label.config(text="Please select two image directories and wait for feature extraction.")

