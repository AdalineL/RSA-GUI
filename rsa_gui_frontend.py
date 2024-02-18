""""""

# Standard library imports
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from inspect import getmembers, isfunction
import torch

# Libraries with pre-trained models
import torchvision.models as torchvision_models
import keras.applications as keras_models
import timm.models as timm_models
from clip import available_models

# ThingsVision modules
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader


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


def user_select_dir(root, start_dir, title, row):
    """Get the image directory from the user."""
    # Initialize the path to the image directory
    selected_dir = tk.StringVar(root)
    selected_dir.set(start_dir)

    # Make labels and entries for the GUI
    selected_dir_label = tk.Label(root, text=title)
    selected_dir_label.grid(row=row, column=0)
    selected_entry = tk.Entry(root, textvariable=selected_dir)
    selected_entry.grid(row=row, column=1)

    # Make a button to browse for the image directory
    selected_dir_button = ttk.Button(
        root,
        text="Browse",
        command=lambda: selected_dir.set(
            filedialog.askdirectory(initialdir=start_dir)
        ),
    )
    selected_dir_button.grid(row=row, column=2)
    return selected_dir


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
    return source, model


def compute_feature_maps(image_dir, source, model, feature_map_head_dir):
    """Get the layers to extract features from."""
    # Set up extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = get_extractor(
        model_name=model.get(),
        source=source.get(),
        device=device,
        pretrained=True,
    )

    # Make feature map directory (if necessary)
    feature_map_dir = f"{feature_map_head_dir.get()}/feature_maps"
    os.makedirs(feature_map_dir, exist_ok=True)

    # Set up dataset and dataloader
    dataset = ImageDataset(
        root=image_dir.get(),
        out_path=feature_map_dir,
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(resize_dim=256, crop_dim=224),
    )
    batches = DataLoader(
        dataset=dataset, batch_size=32, backend=extractor.get_backend()
    )

    # Extract features
    modules = extractor.get_module_names()
    for i, module_name in enumerate(modules):
        # Make directory for module-specific feature maps
        module_dir = (
            f"{feature_map_dir}/{source.get()}/{model.get()}/{module_name}"
        )
        os.makedirs(module_dir, exist_ok=True)

        # Skip if the module has already been processed
        if os.path.exists(f"{module_dir}/features.npy"):
            continue

        # Extract features
        print(
            f"Extracting features from {module_name} "
            f"module ({i+1}/{len(modules)})"
        )
        features = extractor.extract_features(
            batches=batches,
            module_name=module_name,
            flatten_acts=True,
            output_type="ndarray",
        )

        # Save features
        save_features(
            features=features, out_path=module_dir, file_format="npy"
        )
    return


def make_gui():
    """Make the GUI for the RSA analysis."""
    # Create a new GUI
    root = tk.Tk()
    root.title("Representational Similarity Analysis (RSA) GUI")
    root.geometry("800x600")

    # Get the image directory
    user_home = os.path.expanduser("~")
    image_dir = user_select_dir(root, user_home, "Image directory", 0)

    # Determine which model to extract features from
    source, model = get_model(root)

    # Get feature map directory
    feature_map_head_dir = user_select_dir(
        root, os.getcwd(), "Feature map directory", 3
    )

    # Make a button to generate the feature maps
    feature_maps_button = ttk.Button(
        root,
        text="Compute feature maps",
        command=lambda: compute_feature_maps(
            image_dir, source, model, feature_map_head_dir
        ),
    )
    feature_maps_button.grid(row=4, column=1)

    # Run the GUI
    root.mainloop()
    return


if __name__ == "__main__":
    make_gui()
