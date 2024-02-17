import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def calculate_rdm(matrix, method='spearman'):
    # TODO: Correlation Coefficient type should be selectable
    n_samples = matrix.shape[0]
    rdm = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            if method == 'spearman':
                corr, _ = spearmanr(matrix[i], matrix[j])
                rdm[i, j] = 1 - corr
    return rdm

def compare_rdms(rdm1, rdm2):
    # TODO: Correlation Coefficient type should be selectable
    rdm1_flat = rdm1.flatten()
    rdm2_flat = rdm2.flatten()
    corr, _ = spearmanr(rdm1_flat, rdm2_flat)
    return corr

class RSAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RSA GUI")
        self.geometry("400x200")

        # Buttons
        self.load_matrix1_button = tk.Button(self, text="Load Matrix 1", command=self.load_matrix1)
        self.load_matrix1_button.pack()

        self.load_matrix2_button = tk.Button(self, text="Load Matrix 2", command=self.load_matrix2)
        self.load_matrix2_button.pack()

        self.compare_button = tk.Button(self, text="Compare RDMs", command=self.compare_rdms)
        self.compare_button.pack()

        # Results Label
        self.results_label = tk.Label(self, text="")
        self.results_label.pack()

        # Data
        self.matrix1 = None
        self.matrix2 = None

    def load_matrix(self):
        # TODO: Files should be .py or .mat
        filename = filedialog.askopenfilename(title="Select file", filetypes=(("CSV files", "*.csv"),))
        if filename:
            return pd.read_csv(filename).values
        return None

    def load_matrix1(self):
        # TODO: Input should use ThingsVision to flatten input
        self.matrix1 = self.load_matrix()

    def load_matrix2(self):
        # TODO: Input should use ThingsVision to flatten input
        self.matrix2 = self.load_matrix()

    def compare_rdms(self):
        if self.matrix1 is not None and self.matrix2 is not None:
            rdm1 = calculate_rdm(self.matrix1)
            rdm2 = calculate_rdm(self.matrix2)
            correlation = compare_rdms(rdm1, rdm2)
            self.results_label.config(text=f"Spearman Correlation: {correlation}")
        else:
            self.results_label.config(text="Please load both matrices.")


if __name__ == "__main__":
    app = RSAApp()
    app.mainloop()
