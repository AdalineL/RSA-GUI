""""""

# Import necessary modules
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from inspect import getmembers, isfunction
import torchvision.models as torchvision_models
import keras.applications as keras_models
import timm.models as timm_models
from clip import available_models


def _get_function_names(module):
    """Get the names of all the functions in a module."""
    # Get all the functions in the module
    functions = getmembers(module, isfunction)
    return [function[0] for function in functions]


def compile_models_dct():
    """Compile a list of models that thingsvision supports."""
    # Initialize the dictionary of models
    models_dct = {}

    # Get all the functions in the torchvision models module
    models_dct["torchvision"] = _get_function_names(torchvision_models)
    models_dct["keras"] = _get_function_names(keras_models)
    models_dct["timm"] = _get_function_names(timm_models)
    models_dct["clip"] = available_models()
    return models_dct


def get_image_dir(root):
    """Get the image directory from the user."""
    # Initialize the path to the image directory
    user_home = os.path.expanduser("~")
    image_dir = tk.StringVar(root)
    image_dir.set(user_home)

    # Make labels and entries for the GUI
    image_dir_label = tk.Label(root, text="Image directory")
    image_dir_label.grid(row=0, column=0)
    image_dir_entry = tk.Entry(root, textvariable=image_dir)
    image_dir_entry.grid(row=0, column=1)

    # Make a button to browse for the image directory
    image_dir_button = ttk.Button(
        root,
        text="Browse",
        command=lambda: image_dir.set(
            filedialog.askdirectory(initialdir=user_home)
        ),
    )
    image_dir_button.grid(row=0, column=2)
    return image_dir


def get_model(root):
    """Get the model to extract features from."""
    # Initialize the source and model variables
    source = tk.StringVar(root)
    model = tk.StringVar(root)

    # Get the list of models
    models_dct = compile_models_dct()

    # Update the model options when the source changes
    def update_model_options(*args):
        """Update the model options when the source changes."""
        model.set(models_dct[source.get()][0])
        model_menu = tk.OptionMenu(root, model, *models_dct[source.get()])
        model_menu.grid(row=2, column=1)
        return

    source.trace_add("write", update_model_options)

    # Make labels and entries for the model source
    source_label = tk.Label(root, text="Model source")
    source_label.grid(row=1, column=0)
    source_menu = tk.OptionMenu(root, source, *models_dct.keys())
    source_menu.grid(row=1, column=1)

    # Make labels and entries for the model
    model_label = tk.Label(root, text="Model architecture")
    model_label.grid(row=2, column=0)
    model_menu = tk.OptionMenu(root, model, "")
    model_menu.grid(row=2, column=1)

    # Set the default source
    source.set("torchvision")
    return model


def make_gui():
    """Make the GUI for the RSA analysis."""
    # Create a new GUI
    root = tk.Tk()
    root.title("Representational Similarity Analysis (RSA) GUI")
    root.geometry("800x600")

    # Get the image directory
    image_dir = get_image_dir(root)

    # Determine which model to extract features from
    model = get_model(root)

    # Run the GUI
    root.mainloop()
    return


if __name__ == "__main__":
    make_gui()
