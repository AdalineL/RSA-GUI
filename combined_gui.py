"""Frontend for the Representational Similarity Analysis (RSA) GUI. Uses 
thingsvision to extract features from images."""

# Standard library imports
import os
import numpy as np
import rsatoolbox
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
from bayes_opt import BayesianOptimization

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
from thingsvision.core.rsa import compute_rdm, correlate_rdms




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
# Spearman, squared euclidean
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
        os.path.join(modules_head_dir, module)
        for module in os.listdir(modules_head_dir)
        if os.path.isdir(os.path.join(modules_head_dir, module))
    ]

    # Load the layer activations
    layer_activations_dct = {}
    for module_dir in module_dirs:
        # Determine the module
        module = os.path.basename(module_dir)
        module_activations = []

        # Iterate through all .npy files in the directory
        for file in os.listdir(module_dir):
            if file.endswith(".npy"):
                layer_activation = np.load(os.path.join(module_dir, file))
                module_activations.append(layer_activation)

        layer_activations_dct[module] = module_activations
        
        # Print information about loaded activations
        print(
            f"Loaded {len(module_activations)} layer activations for {module} module."
            f" with shape {module_activations[0].shape}"
        )

    LAYER_ACTIVATIONS = layer_activations_dct
    # print(LAYER_ACTIVATIONS)
    print("Layer activations have been loaded.")
    
    return layer_activations_dct


def get_correlation_method_for_rdm(root):
    """Create a dropdown menu for selecting the correlation method."""
    correlation_method = tk.StringVar(root)
    
    options = list(CORRELATION_METHODS.keys())
    correlation_method.set(options[0])  # default value to the first option

    method_label = tk.Label(root, text="Correlation Method for RDMs")
    method_label.grid(row=6, column=0)
    method_menu = tk.OptionMenu(root, correlation_method, *options)
    method_menu.grid(row=6, column=1)

    return correlation_method


def get_correlation_method_for_comparison(root):
    """Create a dropdown menu for selecting the correlation method."""
    correlation_method = tk.StringVar(root)
    
    options = list(CORRELATION_METHODS.keys())
    correlation_method.set(options[0])  # default value to the first option

    method_label = tk.Label(root, text="Correlation Method for RDMs")
    method_label.grid(row=11, column=0)
    method_menu = tk.OptionMenu(root, correlation_method, *options)
    method_menu.grid(row=11, column=1)

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
    for layer, activations_arr in LAYER_ACTIVATIONS.items():
        
        if all(isinstance(act, np.ndarray) for act in activations_arr):
            # Concatenate along the first axis (adjust axis if necessary)
            activations = np.concatenate(activations_arr, axis=0)
        else:
            print(f"Layer {layer} has non-array activations.")
            continue
        
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
        buf = canvas_agg.buffer_rgba()
        tk_img = ImageTk.PhotoImage(image=Image.frombuffer('RGBA', canvas_agg.get_width_height(), buf, 'raw', 'RGBA', 0, 1))


        # Display on Tkinter canvas
        image_label = tk.Label(rdm_display_tab, image=tk_img)
        image_label.image = tk_img  # Keep reference
        image_label.grid(row=0, column=0, columnspan=3)
        
        print("RDMs are Displayed.")


    else:
        print("No RDMs found to display.")
        
# Helper function to navigate to the previous RDM    
def prev_rdm(rdm_display_tab, rdm_head_dir, source, model):
    global CURR_RDM_IDX 
    
    rdm_dir = f"{rdm_head_dir.get()}/rdms"
    modules_head_dir = f"{rdm_dir}/{source.get()}/{model.get()}"
    module_dirs = [
        f"{modules_head_dir}/{module}"
        for module in os.listdir(modules_head_dir)
        if os.path.exists(f"{modules_head_dir}/{module}/rdm.npy")
    ]

    # Restart index count if it exceeds the number of RDMs
    CURR_RDM_IDX = (CURR_RDM_IDX - 1) % len(module_dirs) if module_dirs else 0

    display_rdms(rdm_display_tab, rdm_head_dir, source, model)  
        
        
# Helper function to navigate to the next RDM    
def next_rdm(rdm_display_tab, rdm_head_dir, source, model):
    global CURR_RDM_IDX 
    
    rdm_dir = f"{rdm_head_dir.get()}/rdms"
    modules_head_dir = f"{rdm_dir}/{source.get()}/{model.get()}"
    module_dirs = [
        f"{modules_head_dir}/{module}"
        for module in os.listdir(modules_head_dir)
        if os.path.exists(f"{modules_head_dir}/{module}/rdm.npy")
    ]

    # Restart index count if it exceeds the number of RDMs
    CURR_RDM_IDX = (CURR_RDM_IDX + 1) % len(module_dirs) if module_dirs else 0

    display_rdms(rdm_display_tab, rdm_head_dir, source, model)  
        
        
        
# Have spearman as an option
def compare_rdms(rdm_comparison_tab, target_rdm_dir, comparison_rdm_dir, method_name):
    
    # Get the correlation method
    method_name = CORRELATION_METHODS.get(method_name, "correlation")
    if method_name not in ['correlation', 'cosine', 'euclidean', 'gaussian']:
        raise ValueError("Unsupported correlation method")
    # For vision.compute_rdm, must specify correlation as "pearson"
    if method_name == "correlation":
        method_name = "pearson"
    
    # Load the target RDM
    target = np.load(f"{target_rdm_dir.get()}/rdm.npy")
    
    # Load the comparison RDM
    comparison = np.load(f"{comparison_rdm_dir.get()}/rdm.npy")
    
    # Compare the two RDMs
    # "sigma_k": covariance matrix of the pattern estimates. Used only for methods ‘corr_cov’ and ‘cosine_cov’.
    # rdm_comparison = rsatoolbox.rdm.compare(target, comparison, method=method_name, sigma_k=None)
    rdm_coef = correlate_rdms(target, comparison, correlation=method_name)
    
    # Create matplotlib figure
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    
    # Display target RDM
    ax1 = fig.add_subplot(gs[0, 0])
    cax1 = ax1.matshow(target, cmap='viridis')
    fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title("Target RDM")

    # Display comparison RDM
    ax2 = fig.add_subplot(gs[0, 1])
    cax2 = ax2.matshow(comparison, cmap='viridis')
    fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title("Comparison RDM")

    # Display correlation coefficient
    ax3 = fig.add_subplot(gs[1, :])
    ax3.text(0.5, 0.5, f"RDM Coefficient: {rdm_coef:.2f}", fontsize=14, ha='center', va='center')
    ax3.set_title(f"Comparison using {method_name.capitalize()}")
    ax3.axis('off')  # Hide axes
    
    # Convert to a format Tkinter can use
    canvas_agg = FigureCanvasAgg(fig)
    canvas_agg.draw()
    tk_img = ImageTk.PhotoImage(image=Image.frombytes('RGB', canvas_agg.get_width_height(), canvas_agg.tostring_rgb()))

    # Display on Tkinter canvas
    image_label = tk.Label(rdm_comparison_tab, image=tk_img)
    image_label.image = tk_img  # Keep reference
    image_label.grid(row=0, column=0, columnspan=3)

    print("RDM computation using " + method_name + " is complete.")
    return rdm_coef



def model_performance(weights, rdms, target_rdm):
    # Combine RDMs based on the weights and calculate performance as negative Euclidean distance
    combined_rdm = sum(weight * rdms[model] for model, weight in weights.items())
    performance = -np.linalg.norm(target_rdm - combined_rdm)
    return performance



# todo: fix this
def run_bayesian_optimization(rdm_directories, target_rdm_dir):
    # Load the target RDM
    target_rdm = np.load(f"{target_rdm_dir.get()}/rdm.npy")

    # Load comparison RDMs and map them by model name
    # TODO - fix (!!!!!!!!!!!)
    rdms = {extract_model_name(dir): load_rdm(dir) for dir in rdm_directories}

    # Define bounds for Bayesian Optimization (weights between 0 and 1 for each model)
    pbounds = {model: (0, 1) for model in rdms.keys()}

    # Define the optimization function wrapping model_performance
    def optimization_function(**weights):
        return model_performance(weights, rdms, target_rdm)

    # Initialize optimizer with the defined bounds and function
    optimizer = BayesianOptimization(
        f=optimization_function,
        pbounds=pbounds,
        random_state=1,
    )

    # Perform optimization
    optimizer.maximize(
        init_points=2,
        n_iter=10,
    )

    # Output the best combination found
    best_weights = optimizer.max['params']
    print("Best weights found:", best_weights)
    return best_weights



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
    correlation_method_for_rdm_button = get_correlation_method_for_rdm(main_tab)  

    # Make a button to compute the correlation coefficient
    compute_rdm_button = ttk.Button(
        main_tab, 
        text="Compute RDMs",
        command=lambda: compute_rdm(source, model, rdm_head_dir, method_name=correlation_method_for_rdm_button.get())
    )
    compute_rdm_button.grid(row=7, column=1)
    
    # Make a button to display the RDMs
    display_rdms_button = ttk.Button(
        main_tab,
        text="Display RDMs",
        command=lambda: display_rdms(rdm_display_tab, rdm_head_dir, source, model)
    )
    display_rdms_button.grid(row=8, column=1)
    
    # Button to navigate to the previous RDM
    next_rdm_button = ttk.Button(
        rdm_display_tab,
        text="Previous RDM",
        command=lambda: prev_rdm(rdm_display_tab, rdm_head_dir, source, model)
    )
    next_rdm_button.grid(row=9, column=1)
    
    # Button to navigate to the next RDM
    next_rdm_button = ttk.Button(
        rdm_display_tab,
        text="Next RDM",
        command=lambda: next_rdm(rdm_display_tab, rdm_head_dir, source, model)
    )
    next_rdm_button.grid(row=9, column=2)
    
    # Get target RDM directory
    target_rdm_head_dir = user_select_dir(
        main_tab, os.getcwd(), "Target RDM directory", 9
    )
    
    # Get comparison RDM directory
    comparison_rdm_head_dir = user_select_dir(
        main_tab, os.getcwd(), "Comparison RDM directory", 10
    )
    
    # Drop down menu for correlation type
    correlation_method_for_comparison_button = get_correlation_method_for_comparison(main_tab)  
    
    # Make a button to compare the RDMs
    compare_rdms_button = ttk.Button(
        main_tab,
        text="Compare RDMs",
        command=lambda: compare_rdms(rdm_comparison_tab, target_rdm_head_dir, comparison_rdm_head_dir, method_name=correlation_method_for_comparison_button.get())
    )
    compare_rdms_button.grid(row=12, column=1)
    
    # Bayesian optimization button
    # optimize_button = ttk.Button(
    #     main_tab,
    #     text="Optimize Model Weights",
    #     command=run_bayesian_optimization(target_rdm_head_dir, comparison_rdm_head_dir)
    # )
    # optimize_button.grid(row=13, column=1)
    
    # Run the GUI
    root.mainloop()
    return


if __name__ == "__main__":
    make_gui()




# Button 
# Optimize all the models to the target
# or all the targets to the model

# optimization is what combniation of weighted coefficients of models or targets optimizes the target or model (for explainablitiy)
# ultiate goal is a weighted linear equation of either models or targets, inversion can sometimes be useful to narrow down on one speciific set of save_features# models and weights make up either the target or vice versa

# variational RSA
    
    
# Comparison RDM: need to be able to compare a stack of RDMs (Iterate through them)
# (!!) Scroll through and see what each layer's contribution is to the coefficient correlation 
# (!!) Next would be weighted using bayesion optimization (refer to 1:1 notes)

# todo: Include more Alignment Methods (provide proof that they are needed and necessary)