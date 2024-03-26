"""Frontend for the Representational Similarity Analysis (RSA) GUI. Uses 
thingsvision to extract features from images."""

# Standard library imports
import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from inspect import getmembers, isfunction
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.spatial.distance import correlation as distance_correlation
from sklearn.metrics import mutual_info_score
from sklearn.cross_decomposition import CCA
import torch
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure



# Libraries with pre-trained models
import torchvision.models as torchvision_models
import keras.applications as keras_models
import timm.models as timm_models
from clip import available_models
from open_clip import list_pretrained

# ThingsVision modules
from thingsvision import get_extractor
import thingsvision.vision as vision
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader



# Models that cannot be programmatically determined
SSL_MODELS = [
    "simclr-rn50",
    "mocvo2-rn50",
    "jigsaw-rn50",
    "rotnet-rn50",
    "swav-rn50",
    "pirl-rn50",
    "barlowtwins-rn50",
    "vicreg-rn50",
    "dino-rn50",
    "dino-vit-small-p8",
    "dino-vit-small-p16",
    "dino-vit-base-p8",
    "dino-vit-base-p16",
    "dino-xcit-small-12-p16",
    "dino-xcit-small-12-p8",
    "dino-xcit-medium-24-p16",
    "dino-xcit-medium-24-p8",
]
CUSTOM_MODELS = [
    "cornet-s",
    "cornet-r",
    "cornet-rt",
    "cornet-z",
    "Alexnet_ecoset",
    "Resnet50_ecoset",
    "VGG16_ecoset",
    "Inception_ecoset",
]
# Other global variables
CORRELATION_METHODS = {
    'Correlation': "correlation",
    'Cosine': "cosine",
    'Euclidean': "euclidean",
    'Gaussian': "gaussian",
}
#TODO: create a getter method to get layer_activations_dct
LAYER_ACTIVATIONS = {}
# Global variable to track current RDM index
CURR_RDM_IDX = 0


def _get_function_names(module):
    """Get the names of all the functions in a module."""
    # Get all the functions in the module
    functions = getmembers(module, isfunction)
    return [function[0] for function in functions]


def compile_models_dct(ssl_models=SSL_MODELS, custom_models=CUSTOM_MODELS):
    """Compile a list of models that thingsvision supports."""
    # Initialize the dictionary of models
    models_dct = {}

    # Get all the functions in the torchvision models module
    models_dct["torchvision"] = _get_function_names(torchvision_models)
    models_dct["keras"] = _get_function_names(keras_models)
    models_dct["timm"] = _get_function_names(timm_models)
    models_dct["clip"] = available_models()
    models_dct["open_clip"] = list_pretrained()
    models_dct["ssl"] = ssl_models
    models_dct["custom"] = custom_models
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


def compute_layer_activations(image_dir, source, model, layer_activation_head_dir):
    """Get the layers to extract features from."""
    # Set up extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = get_extractor(
        model_name=model.get(),
        source=source.get(),
        device=device,
        pretrained=True,
    )

    # Make layer activation directory (if necessary)
    layer_activation_dir = f"{layer_activation_head_dir.get()}/layer_activations"
    os.makedirs(layer_activation_dir, exist_ok=True)

    # Set up dataset and dataloader
    dataset = ImageDataset(
        root=image_dir.get(),
        out_path=layer_activation_dir,
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(resize_dim=256, crop_dim=224),
    )
    batches = DataLoader(
        dataset=dataset, batch_size=32, backend=extractor.get_backend()
    )

    # Extract features
    modules = extractor.get_module_names()
    for i, module_name in enumerate(modules):
        # Make directory for module-specific layer activations
        module_dir = (
            f"{layer_activation_dir}/{source.get()}/{model.get()}/{module_name}"
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
        extractor.extract_features(
            batches=batches,
            module_name=module_name,
            flatten_acts=True,
            output_type="ndarray",
            output_dir=module_dir,
        )
    return


def load_layer_activations(layer_activation_head_dir, source, model):
    """Load the layer activation that have been computed. WATCH OUT FOR MEMORY!"""
    global LAYER_ACTIVATIONS
    
    # Determine module directories
    layer_activation_dir = f"{layer_activation_head_dir.get()}/layer_activations"
    modules_head_dir = f"{layer_activation_dir}/{source.get()}/{model.get()}"
    module_dirs = [
        f"{modules_head_dir}/{module}"
        for module in os.listdir(modules_head_dir)
        if os.path.exists(f"{modules_head_dir}/{module}/features_0-1.npy")
    ]

    # Load the layer activations
    layer_activations_dct = {}
    for module_dir in module_dirs:
        # Determine the module
        module = module_dir.split("/")[-1]

        # Load the layer activation for the module
        layers = np.load(f"{module_dir}/features_0-1.npy")

        # Add the layer activation to the dictionary
        layer_activations_dct[module] = layers

        # Print the shape of the layer activation
        print(
            f"Loaded layer activation for {module} module"
            f" with shape {layers.shape}"
        )
        
    LAYER_ACTIVATIONS = layer_activations_dct
    print("Layer activations have been loaded.")
    
    return layer_activations_dct


def get_correlation_method(root):
    """Create a dropdown menu for selecting the correlation method."""
    correlation_method = tk.StringVar(root)
    
    options = list(CORRELATION_METHODS.keys())
    correlation_method.set(options[0])  # default value to the first option

    method_label = tk.Label(root, text="Correlation Method for RDMs")
    method_label.grid(row=6, column=0)
    method_menu = tk.OptionMenu(root, correlation_method, *options)
    method_menu.grid(row=6, column=1)

    return correlation_method


def compute_rdm(source, model, rdm_head_dir, method_name):
    global LAYER_ACTIVATIONS
    if not LAYER_ACTIVATIONS:
        messagebox.showinfo("Error", "Layer activations dictionary is empty.")
        return
    
    # Get the correlation method
    method_name = CORRELATION_METHODS.get(method_name, "correlation")
    if method_name not in ['correlation', 'cosine', 'euclidean', 'gaussian']:
        raise ValueError("Unsupported correlation method")
    
    # Make RDM directories (if necessary)
    rdm_dir = f"{rdm_head_dir.get()}/rdms"
    os.makedirs(rdm_dir, exist_ok=True)

    # Initialize a dictionary to store RDMs for each layer
    rdms = {}
    
    # Calculate RDM for each layer
    for layer, activations in LAYER_ACTIVATIONS.items():
        # Compute the RDM
        rdm = vision.compute_rdm(activations, method=method_name)
        rdms[layer] = rdm
        
        # Make directory for module-specific RDMs
        module_dir = (
            f"{rdm_dir}/{source.get()}/{model.get()}/{layer}"
        )
        os.makedirs(module_dir, exist_ok=True)

        # Save the RDM to a file within this layer's directory
        file_path = os.path.join(module_dir, "rdm.npy")
        np.save(file_path, rdm)

    print("RDM computation using " + method_name + " is complete.")
    return rdms



def display_rdms(rdm_display_tab, rdm_head_dir, source, model):
    global CURR_RDM_IDX
    
    # Determine RDM directories
    rdm_dir = f"{rdm_head_dir.get()}/rdms"
    modules_head_dir = f"{rdm_dir}/{source.get()}/{model.get()}"
    module_dirs = [
        f"{modules_head_dir}/{module}"
        for module in os.listdir(modules_head_dir)
        if os.path.exists(f"{modules_head_dir}/{module}/rdm.npy")
    ]

    if module_dirs: 
        # Get the RDM based on current index
        selected_module_dir = module_dirs[CURR_RDM_IDX]
        module = selected_module_dir.split("/")[-1]
        rdm = np.load(f"{selected_module_dir}/rdm.npy")

        # Create matplotlib figure
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        cax = ax.matshow(rdm, cmap='viridis')
        fig.colorbar(cax)
        ax.set_title(f"RDM: {module}")

        # Convert to a format Tkinter can use
        canvas_agg = FigureCanvasAgg(fig)
        canvas_agg.draw()
        tk_img = ImageTk.PhotoImage(image=Image.frombytes('RGB', canvas_agg.get_width_height(), canvas_agg.tostring_rgb()))

        # Display on Tkinter canvas
        image_label = tk.Label(rdm_display_tab, image=tk_img)
        image_label.image = tk_img  # Keep reference
        image_label.grid(row=0, column=0, columnspan=3)

    else:
        print("No RDMs found to display.")
        
        
# Helper function to navigate to the next RDM    
def next_rdm(rdm_display_tab, rdm_head_dir, source, model):
    global CURR_RDM_IDX
    CURR_RDM_IDX += 1  
    
    rdm_dir = f"{rdm_head_dir.get()}/rdms"
    modules_head_dir = f"{rdm_dir}/{source.get()}/{model.get()}"
    module_dirs = [
        f"{modules_head_dir}/{module}"
        for module in os.listdir(modules_head_dir)
        if os.path.exists(f"{modules_head_dir}/{module}/rdm.npy")
    ]

    # Restart index count if it exceeds the number of RDMs
    if CURR_RDM_IDX > len(module_dirs):
        CURR_RDM_IDX = 0
        
    display_rdms(rdm_display_tab, rdm_head_dir, source, model)  
        
        
def compare_rdms(rdm_comparison_tab, target_rdm_dir, comparison_rdm_dir, method_name):
    
    # Get the correlation method
    method_name = CORRELATION_METHODS.get(method_name, "correlation")
    if method_name not in ['correlation', 'cosine', 'euclidean', 'gaussian']:
        raise ValueError("Unsupported correlation method")
    
    # Load the target RDM
    target = np.load(f"{target_rdm_dir}/rdm.npy")
    
    # Load the comparison RDM
    comparison = np.load(f"{comparison_rdm_dir}/rdm.npy")
    
    # Compare the two RDMs
    rdm_correlation = vision.correlate_rdms(target, comparison, correlation=method_name)
    rdm = np.load(rdm_correlation)

    # Create matplotlib figure
    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    cax = ax.matshow(rdm, cmap='viridis')
    fig.colorbar(cax)
    ax.set_title(f"RDM Comparison")

    # Convert to a format Tkinter can use
    canvas_agg = FigureCanvasAgg(fig)
    canvas_agg.draw()
    tk_img = ImageTk.PhotoImage(image=Image.frombytes('RGB', canvas_agg.get_width_height(), canvas_agg.tostring_rgb()))

    # Display on Tkinter canvas
    image_label = tk.Label(rdm_comparison_tab, image=tk_img)
    image_label.image = tk_img  # Keep reference
    image_label.grid(row=0, column=0, columnspan=3)

    print("RDM computation using " + method_name + " is complete.")
    return rdm_correlation





def make_gui():
    """Make the GUI for the RSA analysis."""
    # Create a new GUI
    root = tk.Tk()
    root.title("Representational Similarity Analysis (RSA) GUI")
    root.geometry("800x600")
    
    # Initialize ttk.Notebook for tabbed interface
    tab_control = ttk.Notebook(root)
    
    # Create tabs
    main_tab = ttk.Frame(tab_control)
    rdm_display_tab = ttk.Frame(tab_control)
    rdm_comparison_tab = ttk.Frame(tab_control)
    
    # Add tabs to the notebook
    tab_control.add(main_tab, text='Main')
    tab_control.add(rdm_display_tab, text='RDM Display')
    tab_control.add(rdm_comparison_tab, text='RDM Comparison Display')
    
    tab_control.pack(expand=1, fill="both")

    # Get the image directory
    user_home = os.path.expanduser("~")
    image_dir = user_select_dir(main_tab, os.path.join(os.getcwd(), 'images'), "Image directory", 0)

    # Determine which model to extract layer from
    source, model = get_model(main_tab)

    # Get layer activation directory
    layer_activation_head_dir = user_select_dir(
        main_tab, os.getcwd(), "Layer activation directory", 3
    )

    # Make a button to generate the layer activations
    compute_layer_activations_button = ttk.Button(
        main_tab,
        text="Compute layer activations",
        command=lambda: compute_layer_activations(
            image_dir, source, model, layer_activation_head_dir
        ),
    )
    compute_layer_activations_button.grid(row=4, column=1)

    # Make a button to load the layer activations
    load_layer_activations_button = ttk.Button(
        main_tab,
        text="Load layer activations",
        command=lambda: load_layer_activations(layer_activation_head_dir, source, model),
    )
    load_layer_activations_button.grid(row=4, column=2)
    
    # Get RDMs directory
    rdm_head_dir = user_select_dir(
        main_tab, os.getcwd(), "RDMs directory", 5
    )
    
    # Add correlation method selection
    correlation_method_button = get_correlation_method(main_tab)  

    # Make a button to compute the correlation coefficient
    compute_rdm_button = ttk.Button(
        main_tab, 
        text="Compute RDMs",
        command=lambda: compute_rdm(source, model, rdm_head_dir, method_name=correlation_method_button.get())
    )
    compute_rdm_button.grid(row=7, column=1)
    
    # Make a button to display the RDMs
    display_rdms_button = ttk.Button(
        main_tab,
        text="Display RDMs",
        command=lambda: display_rdms(rdm_display_tab, rdm_head_dir, source, model)
    )
    display_rdms_button.grid(row=8, column=1)
    
    # Button to navigate to the next RDM
    next_rdm_button = ttk.Button(
        rdm_display_tab,
        text="Next RDM",
        command=lambda: next_rdm(rdm_display_tab, rdm_head_dir, source, model)
    )
    next_rdm_button.grid(row=9, column=1)
    
    # Get target RDM directory
    target_rdm_head_dir = user_select_dir(
        main_tab, os.getcwd(), "Target RDM directory", 9
    )
    
    # Get comparison RDM directory
    comparison_rdm_head_dir = user_select_dir(
        main_tab, os.getcwd(), "Comparison RDM directory", 10
    )
    
    # Make a button to compare the RDMs
    compare_rdms_button = ttk.Button(
        main_tab,
        text="Compare RDMs",
        command=lambda: compare_rdms(rdm_comparison_tab, target_rdm_head_dir, comparison_rdm_head_dir, method_name=correlation_method_button.get())
    )
    compare_rdms_button.grid(row=11, column=1)

    # Run the GUI
    root.mainloop()
    return


if __name__ == "__main__":
    make_gui()
