""""""

# Import necessary modules
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog


def get_image_dir(gui):
    """Get the image directory from the user."""
    # Initialize the path to the image directory
    user_home = os.path.expanduser("~")
    image_dir = tk.StringVar()
    image_dir.set(user_home)

    # Make labels and entries for the GUI
    image_dir_label = tk.Label(gui, text="Image directory")
    image_dir_label.grid(row=0, column=0)
    image_dir_entry = tk.Entry(gui, textvariable=image_dir)
    image_dir_entry.grid(row=0, column=1)

    # Make a button to browse for the image directory
    image_dir_button = ttk.Button(
        gui,
        text="Browse",
        command=lambda: image_dir.set(
            filedialog.askdirectory(initialdir=user_home)
        ),
    )
    image_dir_button.grid(row=0, column=2)
    return image_dir


def make_gui():
    """Make the GUI for the RSA analysis."""
    # Create a new GUI
    gui = tk.Tk()
    gui.title("Representational Similarity Analysis (RSA) GUI")
    gui.geometry("800x600")

    # Get the image directory
    image_dir = get_image_dir(gui)

    # Run the GUI
    gui.mainloop()
    return


if __name__ == "__main__":
    make_gui()
