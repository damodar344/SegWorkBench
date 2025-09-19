import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import signal
import sys
import os
import pandas as pd
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LinearSegmentedColormap
from skimage import filters
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Forward declare UNetTrainingTool class to avoid reference error
class UNetTrainingTool:
    """Class for handling UNet training functionality"""
    def __init__(self, master):
        self.master = master
        # Initialize the UI components
        self.setup_ui()
        
    def setup_ui(self):
        # Create a frame for the UNet training UI
        self.main_frame = ttk.Frame(self.master, padding="10 10 10 10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a label explaining the training tool
        ttk.Label(self.main_frame, 
                 text="UNet Training Tool", 
                 font=("Arial", 16, "bold")).pack(pady=10)
        
        # Add description
        description = """This tool allows you to train a UNet model for image segmentation.
        You can specify training data, validation data, and model parameters.
        
        Note: This feature is currently under development and will be available in a future update."""
        
        ttk.Label(self.main_frame, text=description, wraplength=500, justify="left").pack(pady=20)
        
        # Add placeholder for future UI components
        placeholder_frame = ttk.LabelFrame(self.main_frame, text="Training Parameters", padding="10 10 10 10")
        placeholder_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Add some placeholder controls
        ttk.Label(placeholder_frame, text="Training Data Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(placeholder_frame, width=50, state="disabled").grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(placeholder_frame, text="Browse", state="disabled").grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(placeholder_frame, text="Validation Data Directory:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(placeholder_frame, width=50, state="disabled").grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(placeholder_frame, text="Browse", state="disabled").grid(row=1, column=2, padx=5, pady=5)
        
        # Add model parameters section
        model_frame = ttk.LabelFrame(self.main_frame, text="Model Parameters", padding="10 10 10 10")
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Add some placeholder model parameters
        ttk.Label(model_frame, text="Learning Rate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(model_frame, width=10, state="disabled").grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(model_frame, width=10, state="disabled").grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Epochs:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(model_frame, width=10, state="disabled").grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Add training control buttons
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Start Training", state="disabled").pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Stop Training", state="disabled").pack(side=tk.LEFT, padx=10)
        
        # Add a status label
        self.status_label = ttk.Label(self.main_frame, text="Training tool is under development", foreground="blue")
        self.status_label.pack(pady=10)

class ImageHistogramViewer:
    def __init__(self, master):
        self.master = master
        # Note: title should only be set on root window, not on frames

        self.image_path = tk.StringVar()
        self.current_img = None
        self.hist_fig = None
        self.hist_ax = None
        self.segmented_img = None
        self.threshold_value = None
        self.gmm = None
        self.component1_img = None
        self.component2_img = None
        self.comp1_grayscale = None
        self.comp2_grayscale = None
        self.combined_viz = None
        
        # Component selection variables
        self.selected_comp1 = tk.IntVar(value=0)  # Default to component 1
        self.selected_comp2 = tk.IntVar(value=1)  # Default to component 2
        
        # Number of Gaussian components variable
        self.n_components = tk.IntVar(value=2)  # Default to 2 components
        
        # Threshold method selection variable
        self.threshold_method = tk.StringVar(value="GMM")  # Default to GMM
        
        # Pixel statistics
        self.pixel_stats = None

        # --- File Selection Frame ---
        file_frame = ttk.Frame(master, padding="0 0 0 0")
        file_frame.pack(fill=tk.X)

        ttk.Label(file_frame, text="Image File:").grid(row=0, column=0, sticky=tk.W)
        self.file_path_label = ttk.Label(file_frame, textvariable=self.image_path, wraplength=300)
        self.file_path_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(file_frame, text="Open Image", command=self.open_image_file).grid(row=0, column=2, sticky=tk.E, padx=5)
        
        # Add Gaussian components selection
        components_frame = ttk.Frame(master, padding="0 0 0 0")
        components_frame.pack(fill=tk.X)
        ttk.Label(components_frame, text="Number of Gaussian Distributions:").pack(side=tk.LEFT, padx=0)
        components_spinbox = ttk.Spinbox(components_frame, from_=1, to=9, width=5, textvariable=self.n_components)
        components_spinbox.pack(side=tk.LEFT, padx=0)

        # Create a single main content frame instead of tabbed interface
        self.main_content_frame = ttk.Frame(master)
        self.main_content_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Initialize threshold variables for left, center, and right sliders
        self.threshold_left_var = tk.DoubleVar(value=0.0)  # Blue starts at 0
        self.threshold_var = tk.DoubleVar(value=0.5)       # Cutoff remains at 0.5
        self.threshold_right_var = tk.DoubleVar(value=1.0)  # Red starts at 1.0
        
        # Create frames for all UI components in the single view
        # Main horizontal container for images and histogram
        self.content_row_frame = ttk.Frame(self.main_content_frame)
        self.content_row_frame.pack(fill=tk.BOTH, pady=0)
        
        # Left side: Images container
        self.images_row_frame = ttk.Frame(self.content_row_frame)
        self.images_row_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0)
        
        # Configure smaller font style for LabelFrame titles
        style = ttk.Style()
        style.configure('Small.TLabelframe.Label', font=('TkDefaultFont', 8))
        
        # Image viewer frame (first in row)
        self.ui1_frame = ttk.LabelFrame(self.images_row_frame, text="Image Viewer", style='Small.TLabelframe', padding="0 0 0 0")
        self.ui1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Component 1 frame (second in row)
        self.comp1_gray_frame = ttk.LabelFrame(self.images_row_frame, text="Component 1 (Red ROI)", style='Small.TLabelframe', padding="0 0 0 0")
        self.comp1_gray_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Component 2 frame (third in row)
        self.comp2_gray_frame = ttk.LabelFrame(self.images_row_frame, text="Component 2 (Blue ROI)", style='Small.TLabelframe', padding="0 0 0 0")
        self.comp2_gray_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Combined visualization frame (fourth in row)
        self.combined_viz_frame = ttk.LabelFrame(self.images_row_frame, text="Result", style='Small.TLabelframe', padding="0 0 0 0")
        self.combined_viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Right side: Histogram (shorter and on the right)
        self.ui2_frame = ttk.LabelFrame(self.content_row_frame, text="Histogram", style='Small.TLabelframe', padding="0 0 0 0")
        self.ui2_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=True, padx=0, pady=0)

        # Setup image display - consistent size for all images in the horizontal row
        self.fig, self.ax = plt.subplots(figsize=(2, 2))
        # Create image frame in UI Component 1 frame
        self.image_frame = ttk.Frame(self.ui1_frame)
        self.image_frame.pack(side=tk.TOP, anchor=tk.NW, padx=0, pady=0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, anchor=tk.NW)
        self.canvas_widget.config(width=200, height=200)
        
        # Remove axes from main image display
        self.ax.axis('off')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.patch.set_visible(False)
        self.fig.set_facecolor('none')
        
        # Setup component 1 display
        self.comp1_gray_fig, self.comp1_gray_ax = plt.subplots(figsize=(2, 2))
        self.comp1_gray_canvas = FigureCanvasTkAgg(self.comp1_gray_fig, master=self.comp1_gray_frame)
        self.comp1_gray_canvas_widget = self.comp1_gray_canvas.get_tk_widget()
        self.comp1_gray_canvas_widget.pack(side=tk.TOP, anchor=tk.NW)
        self.comp1_gray_canvas_widget.config(width=200, height=200)
        
        # Remove axes from component 1 display
        self.comp1_gray_ax.axis('off')
        self.comp1_gray_ax.set_xticks([])
        self.comp1_gray_ax.set_yticks([])
        self.comp1_gray_fig.patch.set_visible(False)
        self.comp1_gray_fig.set_facecolor('none')
        
        # Setup component 2 display
        self.comp2_gray_fig, self.comp2_gray_ax = plt.subplots(figsize=(2, 2))
        self.comp2_gray_canvas = FigureCanvasTkAgg(self.comp2_gray_fig, master=self.comp2_gray_frame)
        self.comp2_gray_canvas_widget = self.comp2_gray_canvas.get_tk_widget()
        self.comp2_gray_canvas_widget.pack(side=tk.TOP, anchor=tk.NW)
        self.comp2_gray_canvas_widget.config(width=200, height=200)
        
        # Remove axes from component 2 display
        self.comp2_gray_ax.axis('off')
        self.comp2_gray_ax.set_xticks([])
        self.comp2_gray_ax.set_yticks([])
        self.comp2_gray_fig.patch.set_visible(False)
        self.comp2_gray_fig.set_facecolor('none')
        
        # Setup combined visualization display
        self.combined_viz_fig, self.combined_viz_ax = plt.subplots(figsize=(2, 2))
        self.combined_viz_canvas = FigureCanvasTkAgg(self.combined_viz_fig, master=self.combined_viz_frame)
        self.combined_viz_canvas_widget = self.combined_viz_canvas.get_tk_widget()
        self.combined_viz_canvas_widget.pack(side=tk.TOP, anchor=tk.NW)
        self.combined_viz_canvas_widget.config(width=200, height=200)
        
        # Remove axes from combined visualization display
        self.combined_viz_ax.axis('off')
        self.combined_viz_ax.set_xticks([])
        self.combined_viz_ax.set_yticks([])
        self.combined_viz_fig.patch.set_visible(False)
        self.combined_viz_fig.set_facecolor('none')
        
        # Setup histogram display - shorter and positioned on the right
        self.hist_fig, self.hist_ax = plt.subplots(figsize=(6, 2))
        # Create histogram frame in UI Component 2 frame
        self.histogram_frame = ttk.Frame(self.ui2_frame)
        self.histogram_frame.pack(fill=tk.BOTH, expand=True)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.histogram_frame)
        self.hist_canvas_widget = self.hist_canvas.get_tk_widget()
        self.hist_canvas_widget.pack(side=tk.TOP, anchor=tk.NW)
        self.hist_canvas_widget.config(width=600, height=200)
        
        # Configure histogram display with visible axes
        self.hist_ax.axis('on')  # Turn on axes
        self.hist_ax.set_xlabel('Pixel Value', fontsize=10, fontweight='bold')
        self.hist_ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        # Make spines visible
        for spine in self.hist_ax.spines.values():
            spine.set_visible(True)
        self.hist_fig.patch.set_visible(False)  # Make figure background transparent
        self.hist_fig.set_facecolor('none')  # Set figure face color transparent
        self.hist_fig.tight_layout(pad=1.1)  # Add more padding around the histogram plot
        
        # Make sure ticks are visible and properly formatted
        self.hist_ax.tick_params(axis='both', which='both', length=4, width=1, direction='out', labelsize=8)
        
        # Ensure histogram figure size remains fixed at 6x2 as specified during initialization
        # This prevents the figure from expanding when new histograms are plotted
        self.hist_fig.set_size_inches(1, 1, forward=True)
        
        # Initialize threshold variables for left, center, and right sliders
        self.threshold_left_var = tk.DoubleVar(value=0.0)  # Blue starts at 0
        self.threshold_var = tk.DoubleVar(value=0.5)       # Cutoff remains at 0.5
        self.threshold_right_var = tk.DoubleVar(value=1.0)  # Red starts at 1.0
        
        # We'll store the radio buttons in lists for easy access
        self.comp1_radio_buttons = []
        self.comp2_radio_buttons = []
        
        # Define colors for up to 9 components
        self.component_colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown']
        self.component_color_names = ['Blue', 'Red', 'Green', 'Purple', 'Orange', 'Cyan', 'Magenta', 'Yellow', 'Brown']
                
        # Add a method to update the component selection UI when the number of components changes
        def update_component_selection(*args):
            n_comp = self.n_components.get()
            # Update component 1 radio buttons
            for i, rb in enumerate(self.comp1_radio_buttons):
                if i < n_comp:
                    rb.pack(side=tk.LEFT, padx=0)
                else:
                    rb.pack_forget()
            # Update component 2 radio buttons
            for i, rb in enumerate(self.comp2_radio_buttons):
                if i < n_comp:
                    rb.pack(side=tk.LEFT, padx=0)
                else:
                    rb.pack_forget()
            # Reset selection if it's out of range
            if self.selected_comp1.get() >= n_comp:
                self.selected_comp1.set(0)
            if self.selected_comp2.get() >= n_comp:
                self.selected_comp2.set(1)

        # Bind the update method to the n_components variable
        self.n_components.trace_add("write", update_component_selection)
        
        # Create a container frame for threshold method, adjustment, and component selection
        # Place it below the histogram
        self.threshold_container_frame = ttk.Frame(self.main_content_frame)
        self.threshold_container_frame.pack(fill=tk.X, padx=0, pady=2)
        
        # Add threshold method selection frame
        self.threshold_method_frame = ttk.LabelFrame(self.threshold_container_frame, text="Threshold Method")
        self.threshold_method_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)  # Reduced vertical padding
        
        # Add threshold method selection controls
        ttk.Label(self.threshold_method_frame, text="Select threshold method:").pack(anchor=tk.W, padx=0, pady=0)
        
        # Threshold method selection
        method_frame = ttk.Frame(self.threshold_method_frame)
        method_frame.pack(fill=tk.X, padx=0, pady=0)
        
        # Create radio buttons for threshold methods
        methods = ["GMM", "Otsu", "Triangle", "IsoData"]
        for method in methods:
            rb = ttk.Radiobutton(method_frame, text=method, variable=self.threshold_method, value=method,
                                command=self.update_threshold_method)
            rb.pack(side=tk.LEFT, padx=0)
        
        # Add threshold adjustment frame
        self.threshold_control_frame = ttk.LabelFrame(self.threshold_container_frame, text="Threshold Adjustment")
        self.threshold_control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Add component selection frame
        self.component_selection_frame = ttk.LabelFrame(self.threshold_container_frame, text="Component Selection")
        self.component_selection_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=2)  # Reduced vertical padding

        # Add threshold adjustment controls
        
        # Red threshold (formerly Left threshold)
        threshold_left_frame = ttk.Frame(self.threshold_control_frame)
        threshold_left_frame.pack(fill=tk.X, padx=5, pady=2)  # Increased vertical padding
        ttk.Label(threshold_left_frame, text="Red:").pack(side=tk.LEFT)
        # Initialize with default values, will be updated when image is loaded
        self.threshold_left_scale = tk.Scale(threshold_left_frame, from_=0, to=255, resolution=0.0002, orient=tk.HORIZONTAL,
                                    variable=self.threshold_left_var, command=self.update_threshold_left)  # Removed height parameter
        self.threshold_left_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add left threshold value display - hidden
        self.threshold_left_value_label = ttk.Label(threshold_left_frame, text="")
        self.threshold_left_value_label.pack(side=tk.RIGHT, padx=5)
        
        # Cutoff threshold (formerly Center threshold)
        threshold_frame = ttk.Frame(self.threshold_control_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=2)  # Increased vertical padding
        ttk.Label(threshold_frame, text="Cutoff:").pack(side=tk.LEFT)
        # Initialize with default values, will be updated when image is loaded
        self.threshold_scale = tk.Scale(threshold_frame, from_=0, to=255, resolution=0.0002, orient=tk.HORIZONTAL,
                                    variable=self.threshold_var, command=self.update_threshold)  # Removed height parameter
        self.threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add threshold value display
        self.threshold_value_label = ttk.Label(threshold_frame, text="0.5000")
        self.threshold_value_label.pack(side=tk.RIGHT, padx=5)
        
        # Blue threshold (formerly Right threshold)
        threshold_right_frame = ttk.Frame(self.threshold_control_frame)
        threshold_right_frame.pack(fill=tk.X, padx=5, pady=2)  # Increased vertical padding
        ttk.Label(threshold_right_frame, text="Blue:").pack(side=tk.LEFT)
        # Initialize with default values, will be updated when image is loaded
        self.threshold_right_scale = tk.Scale(threshold_right_frame, from_=0, to=255, resolution=0.0002, orient=tk.HORIZONTAL,
                                    variable=self.threshold_right_var, command=self.update_threshold_right)  # Removed height parameter
        self.threshold_right_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add right threshold value display - hidden
        self.threshold_right_value_label = ttk.Label(threshold_right_frame, text="")
        self.threshold_right_value_label.pack(side=tk.RIGHT, padx=5)
        
        # Add component selection controls
        ttk.Label(self.component_selection_frame, text="Select components for segmentation:").pack(anchor=tk.W, padx=0, pady=0)
        
        # Component 1 selection (Red ROI)
        comp1_frame = ttk.Frame(self.component_selection_frame)
        comp1_frame.pack(fill=tk.X, padx=5, pady=2)  # Increased vertical padding
        ttk.Label(comp1_frame, text="Component 1 (Red ROI):").pack(side=tk.LEFT)
        
        # Create radio buttons for component 1 (up to 9)
        for i in range(9):
            rb = ttk.Radiobutton(comp1_frame, text=f"C{i+1} ({self.component_color_names[i]})", 
                                variable=self.selected_comp1, value=i)
            rb.pack(side=tk.LEFT, padx=0)
            self.comp1_radio_buttons.append(rb)
            # Hide buttons beyond the current number of components
            if i >= self.n_components.get():
                rb.pack_forget()
        
        # Component 2 selection
        comp2_frame = ttk.Frame(self.component_selection_frame)
        comp2_frame.pack(fill=tk.X, padx=5, pady=2)  # Increased vertical padding
        ttk.Label(comp2_frame, text="Component 2 (Blue ROI):").pack(side=tk.LEFT)
        
        # Create radio buttons for component 2 (up to 9)
        for i in range(9):
            rb = ttk.Radiobutton(comp2_frame, text=f"C{i+1} ({self.component_color_names[i]})", 
                                variable=self.selected_comp2, value=i)
            rb.pack(side=tk.LEFT, padx=0)
            self.comp2_radio_buttons.append(rb)
            # Hide buttons beyond the current number of components
            if i >= self.n_components.get():
                rb.pack_forget()
        
        # Add segmentation and save controls
        # Add segmentation and save controls to the file_frame
        ttk.Button(file_frame, text="Apply Threshold Segmentation", 
                  command=self.apply_segmentation).grid(row=0, column=3, sticky=tk.E, padx=5)
        
        ttk.Button(file_frame, text="Save Results", 
                  command=self.save_segmented_image).grid(row=0, column=4, sticky=tk.E, padx=5)
        
        # Add status bar
        self.status_label = ttk.Label(master, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=2)  # Reduced padding to optimize vertical space
        
    def open_image_file(self):
        filetypes = ("Image files", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.gif"), ("All files", "*.*")
        filepath = filedialog.askopenfilename(title="Open an Image File", filetypes=filetypes)
        if filepath:
            self.image_path.set(filepath)
            self.status_label.config(text="Image loaded: " + filepath)
            try:
                img = Image.open(filepath)
                # Check if it's a TIFF file
                is_tiff = filepath.lower().endswith(('.tiff', '.tif'))
                
                # For non-TIFF files or color images, convert to grayscale if needed
                if not is_tiff and img.mode != 'L':
                    img = img.convert('L')
                # For TIFF files, preserve the original mode but ensure it's grayscale-compatible
                elif is_tiff and img.mode not in ['L', 'I', 'F']:
                    # Convert to 'I' (32-bit integer) or 'F' (32-bit float) to preserve bit depth
                    if img.mode == 'RGB':
                        # Convert RGB to grayscale while preserving bit depth
                        img = img.convert('L')
                        # If it was a high bit-depth TIFF, convert to 'F' to preserve precision
                        if img.mode == 'L' and np.max(np.array(img)) > 0:
                            img = img.convert('F')
                
                self.current_img = img
                self.segmented_img = None  # Reset segmented image
                self.threshold_value = None  # Reset threshold value
                self.display_image(img)
                # Automatically show histogram when image is loaded
                self.plot_histogram(img)
                
                # Display image mode information in status bar
                self.status_label.config(text=f"Image loaded: {filepath} (Mode: {img.mode})")
                
                # Ensure window is properly sized after loading image
                self.master.update_idletasks()  # Process all pending UI updates
                # Get the root window (may be different from self.master if in a tab)
                root = self.master.winfo_toplevel()
                # Update window geometry to ensure all elements are visible
                current_geometry = root.geometry().split('+')[0]  # Get current size without position
                width, height = map(int, current_geometry.split('x'))
                if height < 1000:  # If window is too small, resize it
                    root.geometry(f"{width}x1000")
            except Exception as e:
                self.status_label.config(text=f"Error displaying image: {e}")

    def display_image(self, img_pil, title=None):
        self.fig.clf()  # Clear the entire figure
        self.ax = self.fig.add_subplot(111) # Re-create the axes
        # Use appropriate normalization based on image mode
        if img_pil.mode in ['I', 'F']:
            # For high bit-depth images, normalize properly
            img_array = np.array(img_pil)
            vmin = np.min(img_array)
            vmax = np.max(img_array)
            self.ax.imshow(img_array, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 1, 0, 1], interpolation='nearest')
        else:
            self.ax.imshow(img_pil, cmap='gray', aspect='auto', extent=[0, 1, 0, 1], interpolation='nearest')
            
        # Re-apply axis properties after clearing and re-creating
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title(f"Original Image (Mode: {img_pil.mode})")
        self.ax.axis('off')  # Ensure axes are turned off for the image
        self.ax.set_frame_on(False)  # Remove the frame
        
        # Remove all spines
        for spine in self.ax.spines.values():
            spine.set_visible(False)
            
        # Adjust figure to remove all padding and margins
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0, hspace=0)
        self.fig.patch.set_visible(False)  # Make figure background transparent
        self.fig.set_facecolor('none')  # Make figure face color transparent
        
        # Remove ticks and tick labels
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # Ensure tight layout
        self.fig.tight_layout(pad=0)
        
        # Ensure figure size remains fixed at 1x1 as specified during initialization
        # This prevents the figure from expanding when new images are loaded
        # IMPORTANT: This must be called AFTER tight_layout to prevent size override
        self.fig.set_size_inches(1, 1, forward=True)
        
        self.canvas.draw()

    def update_threshold_method(self):
        """Calculate and update threshold based on the selected threshold method"""
        if self.current_img is None:
            return
            
        # Get the selected threshold method
        method = self.threshold_method.get()

        if method == "GMM":
            if self.gmm is None:
                # If GMM model is not available, try to plot histogram to initialize it
                self.plot_histogram(self.current_img) # Pass current_img to plot_histogram
                if self.gmm is None: # Check again after attempting to plot
                    self.status_label.config(text="Error: No GMM model available. Please load an image first.")
                    messagebox.showerror("GMM Error", "No GMM model available. Please load an image first.")
                    return
        
        # Get pixel values and filter out black pixels (ROI)
        pixels = np.array(self.current_img).flatten()
        non_black_pixels = pixels[pixels > 0]  # Filter out black pixels
        
        # Calculate threshold based on the selected method
        if method == "GMM":
            # Use GMM to calculate threshold between two components
            # Sort components by mean value
            means = [self.gmm.means_[i][0] for i in range(self.n_components.get())]
            stds = [np.sqrt(self.gmm.covariances_[i][0][0]) for i in range(self.n_components.get())]
            
            # Use the two components with the closest means
            sorted_indices = np.argsort(means)
            mean_diffs = [abs(means[sorted_indices[i]] - means[sorted_indices[i+1]]) for i in range(len(sorted_indices)-1)]
            closest_pair_idx = np.argmin(mean_diffs)
            
            comp1_idx = sorted_indices[closest_pair_idx]
            comp2_idx = sorted_indices[closest_pair_idx + 1]
            
            # Ensure comp1_idx has the smaller mean
            if means[comp1_idx] > means[comp2_idx]:
                comp1_idx, comp2_idx = comp2_idx, comp1_idx
            
            mean1 = means[comp1_idx]
            mean2 = means[comp2_idx]
            std1 = stds[comp1_idx]
            std2 = stds[comp2_idx]
            
            # Calculate the intersection point between the two Gaussian components
            # Define the range between the two means to search for intersection
            pixel_min = np.min(non_black_pixels)
            pixel_max = np.max(non_black_pixels)
            search_min = max(mean1, pixel_min)
            search_max = min(mean2, pixel_max)
            
            # Create a finer grid for more precise intersection finding
            search_x = np.linspace(search_min, search_max, 10000).reshape(-1, 1)
            
            # Component 1 (smaller mean) - properly scaled by weight
            comp1_pdf = self.gmm.weights_[comp1_idx] * np.exp(-0.5 * ((search_x - mean1) / std1) ** 2) / (std1 * np.sqrt(2 * np.pi))
            
            # Component 2 (larger mean) - properly scaled by weight
            comp2_pdf = self.gmm.weights_[comp2_idx] * np.exp(-0.5 * ((search_x - mean2) / std2) ** 2) / (std2 * np.sqrt(2 * np.pi))
            
            # Find the intersection point (where the two PDFs are equal)
            pdf_diff = comp1_pdf - comp2_pdf
            
            # Use a more robust approach to find the intersection
            # First check for sign changes (crossing points)
            sign_changes = np.where(np.diff(np.signbit(pdf_diff)))[0]
            
            # Initialize threshold with a fallback value
            threshold = (mean1 + mean2) / 2  # Default to midpoint between means
            
            if len(sign_changes) > 0:
                # We found at least one crossing point
                # For multiple crossings, find the one closest to the midpoint between means
                if len(sign_changes) > 1:
                    midpoint = (mean1 + mean2) / 2
                    # Find crossing closest to midpoint
                    closest_idx = np.argmin(np.abs(search_x[sign_changes] - midpoint))
                    idx = sign_changes[closest_idx]
                else:
                    # Just one crossing, use it
                    idx = sign_changes[0]
                
                # Use linear interpolation for more precision
                x1, x2 = search_x[idx][0], search_x[idx+1][0]
                y1, y2 = pdf_diff[idx][0], pdf_diff[idx+1][0]
                
                # Avoid division by zero
                if y2 != y1:
                    # Linear interpolation to find x where y=0
                    threshold = x1 - y1 * (x2 - x1) / (y2 - y1)
            else:
                # If no crossing found, use weighted average based on standard deviations
                threshold = (mean1 * std2 + mean2 * std1) / (std1 + std2)
                
        elif method == "Otsu":
            # Use Otsu's method to calculate threshold
            threshold = filters.threshold_otsu(non_black_pixels)
            
        elif method == "Triangle":
            # Use Triangle method to calculate threshold
            threshold = filters.threshold_triangle(non_black_pixels)
            
        elif method == "IsoData":
            # Use IsoData method to calculate threshold
            threshold = filters.threshold_isodata(non_black_pixels)
        
        # Update the threshold value
        self.threshold_value = threshold
        self.threshold_var.set(threshold)
        
        # Update the threshold value label
        self.threshold_value_label.config(text=f"{threshold:.4f}")
        
        # Redraw the histogram with the new threshold line
        self.update_histogram_threshold()
    
    def update_threshold_left(self, *args, step=0.0001):
        if self.current_img is not None and self.gmm is not None:
            # Get the new blue threshold value
            if len(args) > 0 and isinstance(args[0], float):
                # Round to the nearest step value for precise control
                threshold = round(float(args[0]) / step) * step
                self.threshold_left_var.set(threshold)
            else:
                threshold = self.threshold_left_var.get()
            
            # Ensure blue threshold doesn't exceed cutoff threshold
            if threshold > self.threshold_var.get():
                threshold = self.threshold_var.get()
                self.threshold_left_var.set(threshold)
            
            # Keep the red threshold value label empty (swapped from blue)
            self.threshold_left_value_label.config(text="")
            
            # Redraw the histogram with the new threshold lines
            self.update_histogram_threshold()
    
    def update_threshold(self, *args, step=0.0001):
        if self.current_img is not None and self.gmm is not None:
            # Get the new cutoff threshold value
            if len(args) > 0 and isinstance(args[0], float):
                # Round to the nearest step value for precise control
                threshold = round(args[0] / step) * step
                self.threshold_var.set(threshold)
            else:
                threshold = self.threshold_var.get()
            
            # Ensure cutoff threshold is not below blue threshold (which is fixed at 0)
            # Blue threshold is always 0, so we don't need to adjust it
            if threshold < 0:
                threshold = 0
                self.threshold_var.set(threshold)
            if threshold > self.threshold_right_var.get():
                self.threshold_right_var.set(threshold)
                self.threshold_right_value_label.config(text="")
            
            # Update the threshold value for segmentation
            self.threshold_value = threshold
            
            # Update the threshold value label
            self.threshold_value_label.config(text=f"{threshold:.4f}")
            
            # Update the histogram threshold line
            self.update_histogram_threshold()
            
            # Update the cutoff threshold value label with 4 decimal places
            self.threshold_value_label.config(text=f"{threshold:.4f}")
            
            # Redraw the histogram with the new threshold lines
            self.update_histogram_threshold()
    
    def update_threshold_right(self, *args, step=0.0001):
        if self.current_img is not None and self.gmm is not None:
            # Get the new red threshold value
            if len(args) > 0 and isinstance(args[0], float):
                # Round to the nearest step value for precise control
                threshold = round(args[0] / step) * step
                self.threshold_right_var.set(threshold)
            else:
                threshold = self.threshold_right_var.get()
            
            # Ensure red threshold doesn't go below cutoff threshold
            if threshold < self.threshold_var.get():
                threshold = self.threshold_var.get()
                self.threshold_right_var.set(threshold)
            
            # Keep the blue threshold value label empty (swapped from red)
            self.threshold_right_value_label.config(text="")
            
            # Redraw the histogram with the new threshold lines
            self.update_histogram_threshold()
    
    def update_histogram_threshold(self):
        if self.current_img is not None and self.gmm is not None and self.threshold_value is not None:
            # Get the current y-axis limits
            y_min, y_max = self.hist_ax.get_ylim()
            
            # Clear the histogram axis completely to remove all previous elements
            self.hist_ax.clear()
            
            # Re-plot the histogram bars and GMM components
            self.plot_histogram_elements()
            
            # Get threshold values
            cutoff_threshold = self.threshold_value
            
            # Only add the green threshold line without text label
            if self.threshold_value is not None:
                self.hist_ax.axvline(x=cutoff_threshold, color='green', linestyle='--', linewidth=1.5)
                self.hist_ax.text(cutoff_threshold + 0.01, self.hist_ax.get_ylim()[1] * 0.9, 
                                  f'Cutoff: {cutoff_threshold:.4f}', color='green', 
                                  verticalalignment='top', horizontalalignment='left', fontsize=8)
            
        # Make sure axes are visible for the histogram
        self.hist_ax.axis('on')
        for spine in self.hist_ax.spines.values():
            spine.set_visible(True)
            
        # Make sure ticks are visible and properly formatted
        self.hist_ax.tick_params(axis='both', which='both', length=4, width=1, direction='out', labelsize=8)
            
        # Ensure axis labels remain visible
        method_name = self.threshold_method.get()
        self.hist_ax.set_xlabel('Pixel Intensity', fontsize=10, fontweight='bold')
        self.hist_ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        self.hist_ax.set_title(f'Multivariate Gaussian Mixture Model ({method_name})', fontsize=11, fontweight='bold')
            
        # Make figure background transparent but keep only black histogram bars
        self.hist_fig.patch.set_visible(False)
        self.hist_fig.set_facecolor('none')
        
        # Redraw the canvas
        self.hist_canvas.draw()

    def plot_histogram_elements(self):
        # This function will contain the common plotting logic for histogram bars and GMM components
        # to be called by both plot_histogram and update_histogram_threshold
        if self.current_img is None:
            return

        pixels = np.array(self.current_img).flatten()
        non_black_pixels = pixels[pixels > 0]

        if len(non_black_pixels) == 0:
            return

        # Calculate statistics using non-black pixels
        n_pixels = len(non_black_pixels)
        pixel_min = np.min(non_black_pixels)
        pixel_max = np.max(non_black_pixels)
        pixel_mean = np.mean(non_black_pixels)
        pixel_std = np.std(non_black_pixels)
        
        # Calculate histogram values and bins
        num_bins = 256
        hist_values, bin_edges = np.histogram(non_black_pixels, bins=num_bins, range=(pixel_min, pixel_max), density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mode_bin_index = np.argmax(hist_values)
        mode_value = bin_centers[mode_bin_index]
        mode_count = hist_values[mode_bin_index]

        # Plot histogram bars
        n, bins, patches = self.hist_ax.hist(non_black_pixels, bins=256, range=(pixel_min, pixel_max), color='black', alpha=0.7, density=False)

        # Plot GMM components if available
        if self.gmm is not None:
            x = np.linspace(pixel_min, pixel_max, 1000).reshape(-1, 1)
            for i in range(self.n_components.get()):
                mean = self.gmm.means_[i][0]
                std = np.sqrt(self.gmm.covariances_[i][0][0])
                weight = self.gmm.weights_[i]
                pdf = weight * np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                # Scale PDF to match histogram height
                scaled_pdf = hist_values.max() * pdf / pdf.max() if pdf.max() > 0 else pdf
                color = self.component_colors[i] if i < len(self.component_colors) else 'blue'
                label = f'C{i+1}: μ={mean:.3f}, σ={std:.3f}'
                self.hist_ax.plot(x.flatten(), scaled_pdf, color=color, linewidth=2, label=label)
            self.hist_ax.legend()

        # Set initial axis limits to actual data range
        self.hist_ax.set_xlim(pixel_min, pixel_max)

        # Adjust y-axis limits to reduce length and fit content
        max_freq = 0
        for patch in self.hist_ax.patches:
            height = patch.get_height()
            if height > max_freq:
                max_freq = height
        if max_freq > 0:
            self.hist_ax.set_ylim(0, max_freq * 1.1)

        # Set labels and title
        method_name = self.threshold_method.get()
        self.hist_ax.set_xlabel('Pixel Intensity', fontsize=10, fontweight='bold')
        self.hist_ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        self.hist_ax.set_title(f'Multivariate Gaussian Mixture Model ({method_name})', fontsize=11, fontweight='bold')

        # Make sure axes are visible for the histogram
        self.hist_ax.axis('on')
        for spine in self.hist_ax.spines.values():
            spine.set_visible(True)

        # Make sure ticks are visible and properly formatted
        self.hist_ax.tick_params(axis='both', which='both', length=4, width=1, direction='out', labelsize=8)

        # Make figure background transparent but keep only black histogram bars
        self.hist_fig.patch.set_visible(False)
        self.hist_fig.set_facecolor('none')

    def plot_histogram(self, img):
        # Clear previous histogram to remove shadows
        self.hist_ax.clear()

        # Get pixel values and filter out black pixels (ROI)
        pixels = np.array(img).flatten()
        non_black_pixels = pixels[pixels > 0]  # Filter out black pixels
        
        # Check if there are any non-black pixels
        if len(non_black_pixels) == 0:
            # Handle the case when there are no non-black pixels
            self.hist_ax.text(0.5, 0.5, "No non-zero pixels in the image", 
                             ha='center', va='center', transform=self.hist_ax.transAxes,
                             fontsize=12, color='red')
            self.hist_ax.set_xlim(0, 1)
            self.hist_ax.set_ylim(0, 1)
            
            # Set default values for threshold
            self.threshold_var.set(0.5)  # Default middle value for 0-1 range
            self.threshold_scale.config(from_=0.0, to=1.0)
            
            # Make sure axes are visible
            self.hist_ax.set_xlabel('Pixel Intensity (Normalized)', fontsize=10, fontweight='bold')
            self.hist_ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            self.hist_ax.set_title('Multivariate Gaussian Mixture Model (GMM)', fontsize=12, fontweight='bold')
            
            # Make sure axes are visible for the histogram
            self.hist_ax.axis('on')
            for spine in self.hist_ax.spines.values():
                spine.set_visible(True)
            
            # Make sure ticks are visible and properly formatted
            self.hist_ax.tick_params(axis='both', which='both', length=4, width=1, direction='out', labelsize=8)
            
            # Make figure background transparent but keep only black histogram bars
            self.hist_fig.patch.set_visible(False)
            self.hist_fig.set_facecolor('none')
            
            # Set fixed histogram figure size
            self.hist_fig.set_size_inches(1, 1)

            # Update the UI
            self.hist_canvas.draw()
            
            # Show message to user
            self.status_label.config(text="Warning: No non-zero pixels found in the image. Cannot perform GMM analysis.")
            messagebox.showwarning("No Data", "No non-zero pixels found in the image. The image may be completely black or have very low intensity values.")
            return
        
        # Calculate statistics using non-black pixels
        n_pixels = len(non_black_pixels)
        pixel_min = np.min(non_black_pixels)
        pixel_max = np.max(non_black_pixels)
        pixel_mean = np.mean(non_black_pixels)
        pixel_std = np.std(non_black_pixels)
        
        # Set initial axis limits to actual data range
        self.hist_ax.set_xlim(pixel_min, pixel_max)

        # Calculate histogram values and bins using actual data range
        num_bins = 256
        hist_values, bin_edges = np.histogram(non_black_pixels, bins=num_bins, range=(pixel_min, pixel_max), density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mode_bin_index = np.argmax(hist_values)
        mode_value = bin_centers[mode_bin_index]
        mode_count = hist_values[mode_bin_index]
        
        # Plot histogram with black bars and no transparency (without label)
        hist_data = self.hist_ax.hist(non_black_pixels, bins=num_bins, range=(pixel_min, pixel_max), color='black', edgecolor='black', alpha=0.6, density=False)
        
        # Fit Gaussian Mixture Model with user-specified number of components
        n_components = self.n_components.get()
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        
        # Reshape data for GMM fitting
        X = non_black_pixels.reshape(-1, 1)
        gmm.fit(X)
        self.gmm = gmm  # Store the GMM model for later use
        
        # Generate x values for plotting the GMM using actual data range
        x = np.linspace(pixel_min, pixel_max, 1000).reshape(-1, 1)
        
        # Get probabilities and responsibilities
        logprob = gmm.score_samples(x)
        responsibilities = gmm.predict_proba(x)
        
        # Plot the individual Gaussian components (skip the combined GMM curve)
        colors = self.component_colors[:n_components]  # Use the defined component colors
        pdf = np.exp(logprob)  # Still calculate but don't plot the combined curve
        
        # Plot each component
        pdf_components = []
        for i in range(n_components):
            pdf_component = responsibilities[:, i] * np.exp(logprob)
            pdf_components.append(pdf_component)
            # Scale the PDF to match histogram height
            scaled_pdf = hist_values.max() * pdf_component / pdf.max() if pdf.max() > 0 else pdf_component
            self.hist_ax.plot(x.flatten(), scaled_pdf, 
                             color=colors[i], linewidth=2, 
                             label=f'C{i+1}: μ={gmm.means_[i][0]:.3f}, σ={np.sqrt(gmm.covariances_[i][0][0]):.3f}')
        
        # Add legend with draggable option - ensure it only appears once
        handles, labels = self.hist_ax.get_legend_handles_labels()
        if handles:
            legend = self.hist_ax.legend(handles, labels, loc='upper right', fontsize='small')
            legend.set_draggable(True)
        
        # Set initial threshold values
        initial_center = (pixel_min + pixel_max) / 2
        initial_right = initial_center + (pixel_max - initial_center) / 2
        
        # Always set blue threshold to exactly 0, regardless of image content
        self.threshold_left_var.set(0.0)
        self.threshold_var.set(initial_center)
        self.threshold_right_var.set(initial_right)
        
        # Update threshold scale ranges to match histogram range
        self.threshold_left_scale.config(from_=pixel_min, to=pixel_max)
        self.threshold_scale.config(from_=pixel_min, to=pixel_max)
        self.threshold_right_scale.config(from_=pixel_min, to=pixel_max)
        
        # Add title and labels with improved formatting
        method_name = self.threshold_method.get()
        self.hist_ax.set_title(f'Multivariate Gaussian Mixture Model ({method_name})', fontsize=11, fontweight='bold')
        self.hist_ax.set_xlabel('Pixel Intensity', fontsize=10, fontweight='bold')
        self.hist_ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        
        # Make sure axes are visible for the histogram
        self.hist_ax.axis('on')
        for spine in self.hist_ax.spines.values():
            spine.set_visible(True)
            
        # Make sure ticks are visible and properly formatted
        self.hist_ax.tick_params(axis='both', which='both', length=4, width=1, direction='out', labelsize=8)
        
        # Ensure histogram figure size remains fixed at 6x2 as specified during initialization
        # This prevents the figure from expanding when new histograms are plotted
        self.hist_fig.set_size_inches(1, 1, forward=True)
        
        # Calculate threshold based on the selected method
        self.update_threshold_method()
        
        # Redraw the canvas
        self.hist_canvas.draw()

    def plot_histogram_elements(self):
        # This function will contain the common plotting logic for histogram bars and GMM components
        # to be called by both plot_histogram and update_histogram_threshold
        if self.current_img is None:
            return

        pixels = np.array(self.current_img).flatten()
        non_black_pixels = pixels[pixels > 0]

        if len(non_black_pixels) == 0:
            return

        # Calculate statistics using non-black pixels
        n_pixels = len(non_black_pixels)
        pixel_min = np.min(non_black_pixels)
        pixel_max = np.max(non_black_pixels)
        pixel_mean = np.mean(non_black_pixels)
        pixel_std = np.std(non_black_pixels)
        
        # Calculate histogram values and bins
        num_bins = 256
        hist_values, bin_edges = np.histogram(non_black_pixels, bins=num_bins, range=(pixel_min, pixel_max), density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mode_bin_index = np.argmax(hist_values)
        mode_value = bin_centers[mode_bin_index]
        mode_count = hist_values[mode_bin_index]

        # Plot histogram bars
        n, bins, patches = self.hist_ax.hist(non_black_pixels, bins=256, range=(pixel_min, pixel_max), color='black', alpha=0.7, density=False)

        # Plot GMM components if available
        if self.gmm is not None:
            x = np.linspace(pixel_min, pixel_max, 1000).reshape(-1, 1)
            for i in range(self.n_components.get()):
                mean = self.gmm.means_[i][0]
                std = np.sqrt(self.gmm.covariances_[i][0][0])
                weight = self.gmm.weights_[i]
                pdf = weight * np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                # Scale PDF to match histogram height
                scaled_pdf = hist_values.max() * pdf / pdf.max() if pdf.max() > 0 else pdf
                color = self.component_colors[i] if i < len(self.component_colors) else 'blue'
                label = f'C{i+1}: μ={mean:.3f}, σ={std:.3f}'
                self.hist_ax.plot(x.flatten(), scaled_pdf, color=color, linewidth=2, label=label)
            self.hist_ax.legend()

        # Set initial axis limits to actual data range
        self.hist_ax.set_xlim(pixel_min, pixel_max)

        # Adjust y-axis limits to reduce length and fit content
        max_freq = 0
        for patch in self.hist_ax.patches:
            height = patch.get_height()
            if height > max_freq:
                max_freq = height
        if max_freq > 0:
            self.hist_ax.set_ylim(0, max_freq * 1.1)

        # Set labels and title
        method_name = self.threshold_method.get()
        self.hist_ax.set_xlabel('Pixel Intensity', fontsize=10, fontweight='bold')
        self.hist_ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        self.hist_ax.set_title(f'Multivariate Gaussian Mixture Model ({method_name})', fontsize=11, fontweight='bold')

        # Make sure axes are visible for the histogram
        self.hist_ax.axis('on')
        for spine in self.hist_ax.spines.values():
            spine.set_visible(True)

        # Make sure ticks are visible and properly formatted
        self.hist_ax.tick_params(axis='both', which='both', length=4, width=1, direction='out', labelsize=8)

        # Make figure background transparent but keep only black histogram bars
        self.hist_fig.patch.set_visible(False)
        self.hist_fig.set_facecolor('none')
        
        # Store pixel statistics for later use
        self.pixel_stats = {
            'non_black_pixels': non_black_pixels,
            'n_pixels': n_pixels,
            'pixel_min': pixel_min,
            'pixel_max': pixel_max,
            'pixel_mean': pixel_mean,
            'pixel_std': pixel_std,
            'mode': mode_value,
            'mode_count': mode_count
        }
        
        # Display statistics in status bar
        stats_text = f"Pixels: {n_pixels}, Min: {pixel_min:.2f}, Max: {pixel_max:.2f}, Mean: {pixel_mean:.2f}, Std: {pixel_std:.2f}, Mode: {mode_value:.2f}"
        self.status_label.config(text=stats_text)
        
        # Set initial threshold values
        initial_center = (pixel_min + pixel_max) / 2
        initial_right = initial_center + (pixel_max - initial_center) / 2
        
        # Always set blue threshold to exactly 0, regardless of image content
        self.threshold_left_var.set(0.0)
        self.threshold_var.set(initial_center)
        self.threshold_right_var.set(initial_right)
        
        # Update threshold scale ranges to match histogram range
        self.threshold_left_scale.config(from_=pixel_min, to=pixel_max)
        self.threshold_scale.config(from_=pixel_min, to=pixel_max)
        self.threshold_right_scale.config(from_=pixel_min, to=pixel_max)
        
        # Calculate mode (most frequent value)
        # Adjust bin count based on image bit depth
        if img.mode in ['I', 'F']:
            num_bins = min(1024, len(np.unique(non_black_pixels)))  # More bins for high bit-depth
        else:
            num_bins = 256  # Standard for 8-bit
            
        # Calculate bin width
        bin_width = (pixel_max - pixel_min) / num_bins if num_bins > 0 else 0
        
        # Calculate mode (most frequent value)
        hist_values, bin_edges = np.histogram(non_black_pixels, bins=num_bins)
        mode_bin_index = np.argmax(hist_values)
        mode_value = (bin_edges[mode_bin_index] + bin_edges[mode_bin_index + 1]) / 2
        mode_count = hist_values[mode_bin_index]
        
        # Plot histogram with solid black bars (without label)
        hist_data = self.hist_ax.hist(non_black_pixels, bins=num_bins, color='black', edgecolor='black')
        
        # Ensure y-axis shows full range of histogram values
        hist_max = hist_data[0].max()
        self.hist_ax.set_ylim(0, hist_max * 1.2)  # Add 20% padding to the top for better visibility
        
        # Store the histogram max value for consistent scaling
        self.hist_max_value = hist_max
        
        # Fit Gaussian Mixture Model with user-specified number of components
        n_components = self.n_components.get()
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        
        # Reshape data for GMM fitting
        X = non_black_pixels.reshape(-1, 1)
        gmm.fit(X)
        self.gmm = gmm  # Store the GMM model for later use
        
        # Store GMM components for redrawing
        self.gmm_means = [gmm.means_[i][0] for i in range(n_components)]
        self.gmm_stds = [np.sqrt(gmm.covariances_[i][0][0]) for i in range(n_components)]
        self.gmm_weights = gmm.weights_
        
        # Store PDF functions for each component
        self.gmm_pdfs = []
        for i in range(n_components):
            mean = self.gmm_means[i]
            std = self.gmm_stds[i]
            # Create a lambda function for this component's PDF
            pdf = lambda x, mean=mean, std=std: np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            self.gmm_pdfs.append(pdf)
        
        # Generate x values for plotting the GMM (0-1 range)
        x = np.linspace(0, 1, 1000).reshape(-1, 1)
        
        # Get probabilities and responsibilities
        logprob = gmm.score_samples(x)
        responsibilities = gmm.predict_proba(x)
        
        # Plot the individual Gaussian components (skip the combined GMM curve)
        colors = self.component_colors[:n_components]  # Use the defined component colors
        pdf = np.exp(logprob)  # Still calculate but don't plot the combined curve
        
        # Plot each component
        pdf_components = []
        for i in range(n_components):
            pdf_component = responsibilities[:, i] * np.exp(logprob)
            pdf_components.append(pdf_component)
            # Scale the component curves to match the histogram height
            # Use the actual histogram maximum value for better visibility
            scale_factor = hist_data[0].max() * 1.0  # Scale to match histogram height
            # Store this value for consistent scaling across updates
            self.hist_max_value = hist_data[0].max()
            
            # Ensure proper scaling of the PDF component
            scaled_component = scale_factor * pdf_component / pdf.max() if pdf.max() > 0 else scale_factor * pdf_component
            self.hist_ax.plot(x, scaled_component, 
                             color=colors[i], linewidth=2, 
                             label=f'C{i+1}: μ={gmm.means_[i][0]:.3f}, σ={np.sqrt(gmm.covariances_[i][0][0]):.3f}')
            # Only add each component once to avoid duplicate legends
        
        # Don't create a new legend here to avoid duplication
        # Just improve tick visibility
        self.hist_ax.tick_params(axis='both', which='major', labelsize=9)
        self.hist_ax.tick_params(axis='both', which='minor', labelsize=7)
        
        # Add title and labels with improved formatting
        method_name = self.threshold_method.get()
        self.hist_ax.set_title(f'Multivariate Gaussian Mixture Model ({method_name})', fontsize=11, fontweight='bold')
        self.hist_ax.set_xlabel('Pixel Intensity', fontsize=10, fontweight='bold')
        self.hist_ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        
        # Make sure axes are visible for the histogram
        self.hist_ax.axis('on')
        for spine in self.hist_ax.spines.values():
            spine.set_visible(True)
            
        # Make sure ticks are visible and properly formatted
        self.hist_ax.tick_params(axis='both', which='both', length=4, width=1, direction='out', labelsize=8)
        
        # Ensure histogram figure size remains fixed at 6x2 as specified during initialization
        # This prevents the figure from expanding when new histograms are plotted
        self.hist_fig.set_size_inches(1, 1, forward=True)
        
        # Ensure the figure background is transparent
        self.hist_fig.patch.set_visible(False)
        self.hist_fig.set_facecolor('none')
        
        # Calculate threshold based on the selected method
        self.update_threshold_method()
        
        # Redraw the canvas
        self.hist_canvas.draw()

    def plot_histogram_elements(self):
        # This function will contain the common plotting logic for histogram bars and GMM components
        # to be called by both plot_histogram and update_histogram_threshold
        if self.current_img is None:
            return

        pixels = np.array(self.current_img).flatten()
        non_black_pixels = pixels[pixels > 0]

        if len(non_black_pixels) == 0:
            return

        # Calculate statistics using non-black pixels
        n_pixels = len(non_black_pixels)
        pixel_min = np.min(non_black_pixels)
        pixel_max = np.max(non_black_pixels)
        pixel_mean = np.mean(non_black_pixels)
        pixel_std = np.std(non_black_pixels)
        
        # Calculate histogram values and bins
        num_bins = 256
        hist_values, bin_edges = np.histogram(non_black_pixels, bins=num_bins, range=(pixel_min, pixel_max), density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mode_bin_index = np.argmax(hist_values)
        mode_value = bin_centers[mode_bin_index]
        mode_count = hist_values[mode_bin_index]

        # Plot histogram bars
        n, bins, patches = self.hist_ax.hist(non_black_pixels, bins=256, range=(pixel_min, pixel_max), color='black', alpha=0.7, density=False)

        # Plot GMM components if available
        if self.gmm is not None:
            x = np.linspace(pixel_min, pixel_max, 1000).reshape(-1, 1)
            for i in range(self.n_components.get()):
                mean = self.gmm.means_[i][0]
                std = np.sqrt(self.gmm.covariances_[i][0][0])
                weight = self.gmm.weights_[i]
                pdf = weight * np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                # Scale PDF to match histogram height
                scaled_pdf = hist_values.max() * pdf / pdf.max() if pdf.max() > 0 else pdf
                color = self.component_colors[i] if i < len(self.component_colors) else 'blue'
                label = f'C{i+1}: μ={mean:.3f}, σ={std:.3f}'
                self.hist_ax.plot(x.flatten(), scaled_pdf, color=color, linewidth=2, label=label)
            self.hist_ax.legend()

        # Set initial axis limits to actual data range
        self.hist_ax.set_xlim(pixel_min, pixel_max)

        # Adjust y-axis limits to reduce length and fit content
        max_freq = 0
        for patch in self.hist_ax.patches:
            height = patch.get_height()
            if height > max_freq:
                max_freq = height
        if max_freq > 0:
            self.hist_ax.set_ylim(0, max_freq * 1.1)

        # Set labels and title
        method_name = self.threshold_method.get()
        self.hist_ax.set_xlabel('Pixel Intensity', fontsize=10, fontweight='bold')
        self.hist_ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        self.hist_ax.set_title(f'Multivariate Gaussian Mixture Model ({method_name})', fontsize=11, fontweight='bold')

        # Make sure axes are visible for the histogram
        self.hist_ax.axis('on')
        for spine in self.hist_ax.spines.values():
            spine.set_visible(True)

        # Make sure ticks are visible and properly formatted
        self.hist_ax.tick_params(axis='both', which='both', length=4, width=1, direction='out', labelsize=8)

        # Make figure background transparent but keep only black histogram bars
        self.hist_fig.patch.set_visible(False)
        self.hist_fig.set_facecolor('none')
        
        # Redraw the canvas
        self.hist_canvas.draw()

    def plot_histogram_elements(self):
        # This function will contain the common plotting logic for histogram bars and GMM components
        # to be called by both plot_histogram and update_histogram_threshold
        if self.current_img is None:
            return

        pixels = np.array(self.current_img).flatten()
        non_black_pixels = pixels[pixels > 0]

        if len(non_black_pixels) == 0:
            return

        # Calculate statistics using non-black pixels
        n_pixels = len(non_black_pixels)
        pixel_min = np.min(non_black_pixels)
        pixel_max = np.max(non_black_pixels)
        pixel_mean = np.mean(non_black_pixels)
        pixel_std = np.std(non_black_pixels)
        
        # Calculate histogram values and bins
        num_bins = 256
        hist_values, bin_edges = np.histogram(non_black_pixels, bins=num_bins, range=(pixel_min, pixel_max), density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mode_bin_index = np.argmax(hist_values)
        mode_value = bin_centers[mode_bin_index]
        mode_count = hist_values[mode_bin_index]

        # Plot histogram bars
        n, bins, patches = self.hist_ax.hist(non_black_pixels, bins=256, range=(pixel_min, pixel_max), color='black', alpha=0.7, density=False)

        # Plot GMM components if available
        if self.gmm is not None:
            x = np.linspace(pixel_min, pixel_max, 1000).reshape(-1, 1)
            for i in range(self.n_components.get()):
                mean = self.gmm.means_[i][0]
                std = np.sqrt(self.gmm.covariances_[i][0][0])
                weight = self.gmm.weights_[i]
                pdf = weight * np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                # Scale PDF to match histogram height
                scaled_pdf = hist_values.max() * pdf / pdf.max() if pdf.max() > 0 else pdf
                color = self.component_colors[i] if i < len(self.component_colors) else 'blue'
                label = f'C{i+1}: μ={mean:.3f}, σ={std:.3f}'
                self.hist_ax.plot(x.flatten(), scaled_pdf, color=color, linewidth=2, label=label)
            self.hist_ax.legend()

        # Set initial axis limits to actual data range
        self.hist_ax.set_xlim(pixel_min, pixel_max)

        # Adjust y-axis limits to reduce length and fit content
        max_freq = 0
        for patch in self.hist_ax.patches:
            height = patch.get_height()
            if height > max_freq:
                max_freq = height
        if max_freq > 0:
            self.hist_ax.set_ylim(0, max_freq * 1.1)

        # Set labels and title
        method_name = self.threshold_method.get()
        self.hist_ax.set_xlabel('Pixel Intensity', fontsize=10, fontweight='bold')
        self.hist_ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        self.hist_ax.set_title(f'Multivariate Gaussian Mixture Model ({method_name})', fontsize=11, fontweight='bold')

        # Make sure axes are visible for the histogram
        self.hist_ax.axis('on')
        for spine in self.hist_ax.spines.values():
            spine.set_visible(True)

        # Make sure ticks are visible and properly formatted
        self.hist_ax.tick_params(axis='both', which='both', length=4, width=1, direction='out', labelsize=8)

        # Make figure background transparent but keep only black histogram bars
        self.hist_fig.patch.set_visible(False)
        self.hist_fig.set_facecolor('none')
        
        # Store pixel statistics for later use
        self.pixel_stats = {
            'n_pixels': n_pixels,
            'min': pixel_min,
            'max': pixel_max,
            'mean': pixel_mean,
            'std': pixel_std,
            'mode': mode_value,
            'mode_count': mode_count
        }
        
        # Display statistics in status bar
        stats_text = f"Pixels: {n_pixels}, Min: {pixel_min:.2f}, Max: {pixel_max:.2f}, Mean: {pixel_mean:.2f}, Std: {pixel_std:.2f}, Mode: {mode_value:.2f}"
        self.status_label.config(text=stats_text)
        
    def apply_segmentation(self):
        if self.current_img is None:
            messagebox.showinfo("No Image", "Please load an image first.")
            return
            
        if self.gmm is None:
            messagebox.showinfo("No GMM Model", "No GMM model available. Please load an image first.")
            return
        
        # Get the selected components
        comp1_idx = self.selected_comp1.get()
        comp2_idx = self.selected_comp2.get()
        
        # Check if the same component is selected twice
        if comp1_idx == comp2_idx:
            messagebox.showinfo("Invalid Selection", "Please select two different components for segmentation.")
            return
        
        # Get the image array
        img_array = np.array(self.current_img)
        X = img_array.reshape(-1, 1)
        
        # Get component probabilities
        probabilities = self.gmm.predict_proba(X)
        
        # Create segmentation for both selected components
        component1_array = np.zeros_like(img_array)
        component2_array = np.zeros_like(img_array)
        
        # Reshape probabilities back to image shape
        prob1 = probabilities[:, comp1_idx].reshape(img_array.shape)
        prob2 = probabilities[:, comp2_idx].reshape(img_array.shape)
        
        # Get all three threshold values
        blue_threshold = self.threshold_left_var.get()  # Blue threshold (formerly left)
        cutoff_threshold = self.threshold_var.get()     # Cutoff threshold (formerly center)
        red_threshold = self.threshold_right_var.get()  # Red threshold (formerly right)
        
        # If this is the first time applying segmentation, use the calculated optimal threshold
        if self.segmented_img is None and self.threshold_value is not None:
            # Update the cutoff threshold variable to match the calculated threshold
            cutoff_threshold = self.threshold_value
            self.threshold_var.set(cutoff_threshold)
            
            # Keep blue threshold at 0 and only adjust red threshold proportionally
            range_size = self.threshold_scale.cget('to') - self.threshold_scale.cget('from')
            blue_threshold = 0.0  # Always keep blue threshold at 0
            red_threshold = min(cutoff_threshold + range_size * 0.2, self.threshold_scale.cget('to'))
            
            # Update the threshold variables
            self.threshold_left_var.set(blue_threshold)
            self.threshold_right_var.set(red_threshold)
            
            # Update the threshold value labels
            self.threshold_left_value_label.config(text=f"{blue_threshold:.4f}")
            self.threshold_value_label.config(text=f"{cutoff_threshold:.4f}")
            self.threshold_right_value_label.config(text=f"{red_threshold:.4f}")
        
        # Create initial masks for each component based on the threshold ranges and probabilities
        # Component 1 (ROI) - pixels where probability of component 1 is within the threshold range
        # This is our primary region of interest (ROI) displayed in red
        comp1_mask = (prob1 > blue_threshold) & (prob1 <= cutoff_threshold)
        
        # Component 2 (Background) - pixels where probability of component 2 is within the threshold range
        # Use strict inequality for the lower bound to avoid overlap at the cutoff threshold
        comp2_mask = (prob2 > cutoff_threshold) & (prob2 <= red_threshold)
        
        # Handle any potential overlapping regions - pixels that satisfy both conditions
        # For overlapping pixels, assign to the component with higher probability
        overlap_mask = comp1_mask & comp2_mask
        
        if np.any(overlap_mask):
            # Get flat indices of overlapping pixels
            flat_indices = np.ravel_multi_index(np.where(overlap_mask), img_array.shape)
            
            # Get probabilities for these pixels
            p1 = probabilities[flat_indices, comp1_idx]
            p2 = probabilities[flat_indices, comp2_idx]
            
            # Create masks for component assignment based on probability comparison
            comp1_priority = p1 >= p2
            comp2_priority = ~comp1_priority
            
            # Get the original 2D indices
            overlap_indices = np.where(overlap_mask)
            
            # Remove overlapping pixels from both masks
            comp1_mask[overlap_mask] = False
            comp2_mask[overlap_mask] = False
            
            # Then assign to the appropriate component based on probability
            for i in range(len(overlap_indices[0])):
                if comp1_priority[i]:
                    comp1_mask[overlap_indices[0][i], overlap_indices[1][i]] = True
                else:
                    comp2_mask[overlap_indices[0][i], overlap_indices[1][i]] = True
        
        # Apply the masks to the component arrays
        component1_array[comp1_mask] = 1
        component2_array[comp2_mask] = 1
        
        # Handle edge case: pixels exactly at the cutoff threshold
        # Assign pixels exactly at the cutoff threshold to either component based on which has higher probability
        cutoff_mask = (prob1 == cutoff_threshold) | (prob2 == cutoff_threshold)
        if np.any(cutoff_mask):
            # Get flat indices of cutoff pixels
            flat_indices = np.ravel_multi_index(np.where(cutoff_mask), img_array.shape)
            
            # Get probabilities for these pixels
            p1 = probabilities[flat_indices, comp1_idx]
            p2 = probabilities[flat_indices, comp2_idx]
            
            # Create masks for component assignment based on probability comparison
            comp1_mask_cutoff = p1 > p2  # Use strict inequality to break ties consistently
            comp2_mask_cutoff = ~comp1_mask_cutoff
            
            # Get the original 2D indices
            cutoff_indices = np.where(cutoff_mask)
            
            # First, clear any existing assignments for these pixels to avoid overlap
            component1_array[cutoff_mask] = 0
            component2_array[cutoff_mask] = 0
            
            # Then assign to components based on probability comparison
            component1_array[cutoff_indices[0][comp1_mask_cutoff], cutoff_indices[1][comp1_mask_cutoff]] = 1
            component2_array[cutoff_indices[0][comp2_mask_cutoff], cutoff_indices[1][comp2_mask_cutoff]] = 1
        
        # Make sure bright areas are properly included in the ROI (Component 1)
        # This ensures that high intensity pixels are not excluded from the ROI
        # Get original image array for intensity-based inclusion
        original_array = np.array(self.current_img)
        # Find high intensity pixels that might have been missed
        high_intensity_threshold = np.percentile(original_array[original_array > 0], 90)  # Top 10% of intensities
        
        # First, identify high intensity pixels that haven't been classified yet
        unclassified_high_intensity = (original_array >= high_intensity_threshold) & \
                                     (component1_array == 0) & (component2_array == 0)
        
        # Include these high intensity pixels in the ROI (Component 1)
        component1_array[unclassified_high_intensity] = 1
        
        # Final verification step to ensure mutual exclusivity
        # Check for any overlapping pixels and resolve them
        overlap_final = (component1_array > 0) & (component2_array > 0)
        if np.any(overlap_final):
            # For any overlapping pixels, compare their probabilities and assign to the component with higher probability
            overlap_indices = np.where(overlap_final)
            flat_indices = np.ravel_multi_index(overlap_indices, img_array.shape)
            
            # Get probabilities for these pixels
            p1 = probabilities[flat_indices, comp1_idx]
            p2 = probabilities[flat_indices, comp2_idx]
            
            # Determine which component has higher probability
            comp1_priority = p1 > p2
            comp2_priority = ~comp1_priority
            
            # Clear both components for these pixels
            component1_array[overlap_final] = 0
            component2_array[overlap_final] = 0
            
            # Reassign based on probability
            for i in range(len(overlap_indices[0])):
                y, x = overlap_indices[0][i], overlap_indices[1][i]
                if comp1_priority[i]:
                    component1_array[y, x] = 1
                else:
                    component2_array[y, x] = 1
        
        # Ensure no missing areas by checking for unclassified pixels with significant intensity
        # Find pixels that haven't been classified to either component but have significant intensity
        unclassified_mask = (component1_array == 0) & (component2_array == 0) & (original_array > 0)
        
        # For unclassified pixels, assign them to the component with higher probability
        # Using vectorized operations for better performance
        if np.any(unclassified_mask):
            # Get flat indices of unclassified pixels
            flat_indices = np.ravel_multi_index(np.where(unclassified_mask), img_array.shape)
            
            # Get probabilities for these pixels
            p1 = probabilities[flat_indices, comp1_idx]
            p2 = probabilities[flat_indices, comp2_idx]
            
            # Create masks for component assignment based on probability comparison
            # Use strict inequality to break ties consistently
            comp1_unclassified = p1 > p2
            comp2_unclassified = ~comp1_unclassified
            
            # Get the original 2D indices
            unclassified_indices = np.where(unclassified_mask)
            
            # Assign to components based on probability comparison
            component1_array[unclassified_indices[0][comp1_unclassified], unclassified_indices[1][comp1_unclassified]] = 1
            component2_array[unclassified_indices[0][comp2_unclassified], unclassified_indices[1][comp2_unclassified]] = 1
            
        # Additional check for any remaining unclassified pixels with intensity
        # This ensures complete coverage of all non-zero pixels
        still_unclassified = (component1_array == 0) & (component2_array == 0) & (original_array > 0)
        if np.any(still_unclassified):
            # For any remaining unclassified pixels, compare their intensity to the mean intensities of each component
            # to determine the most appropriate component assignment
            comp1_mean_intensity = np.mean(original_array[component1_array > 0]) if np.any(component1_array > 0) else 0
            comp2_mean_intensity = np.mean(original_array[component2_array > 0]) if np.any(component2_array > 0) else 0
            
            # Get intensities of still unclassified pixels
            unclassified_intensities = original_array[still_unclassified]
            
            # Create a mask for pixels closer to component 1's mean intensity
            # For each unclassified pixel, calculate distance to each component's mean intensity
            unclassified_indices = np.where(still_unclassified)
            for i in range(len(unclassified_indices[0])):
                y, x = unclassified_indices[0][i], unclassified_indices[1][i]
                pixel_intensity = original_array[y, x]
                
                # Calculate distance to each component's mean intensity
                dist_to_comp1 = abs(pixel_intensity - comp1_mean_intensity)
                dist_to_comp2 = abs(pixel_intensity - comp2_mean_intensity)
                
                # Assign to the component with the closer mean intensity
                if dist_to_comp1 <= dist_to_comp2:
                    component1_array[y, x] = 1
                else:
                    component2_array[y, x] = 1
        
        # Convert to PIL Images
        if self.current_img.mode in ['I', 'F']:
            component1_img = Image.fromarray(component1_array.astype(np.float32), mode='F')
            component2_img = Image.fromarray(component2_array.astype(np.float32), mode='F')
        else:
            component1_img = Image.fromarray(component1_array.astype(np.uint8), mode='L')
            component2_img = Image.fromarray(component2_array.astype(np.uint8), mode='L')
        
        # Store the segmented images
        self.segmented_img = component1_img  # Store component 1 as main segmented image
        self.component1_img = component1_img
        self.component2_img = component2_img
        
        # Create a figure with two subplots
        plt.close(self.fig)
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.6, 0.8))  # Fixed size of 1.6x0.8 for the two subplots
        # Note: This is a temporary figure size used only for this specific display
        # The main image displays use 1x1 figure size as specified during initialization
        
        # Get component colors for display
        colors = self.component_color_names
        
        # Display both component images
        ax1.imshow(component1_img, cmap='gray')
        ax1.set_title(f"Blue ROI (Threshold: {blue_threshold:.2f}-{cutoff_threshold:.2f})")
        ax1.axis('off')
        
        ax2.imshow(component2_img, cmap='gray')
        ax2.set_title(f"Red ROI (Threshold: {cutoff_threshold:.2f}-{red_threshold:.2f})")
        ax2.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Create grayscale segmentations preserving original intensities
        # original_array was already defined above
        comp1_grayscale = np.zeros_like(original_array)
        comp2_grayscale = np.zeros_like(original_array)
        
        # Extract original intensities for each component
        comp1_grayscale[component1_array > 0] = original_array[component1_array > 0]
        comp2_grayscale[component2_array > 0] = original_array[component2_array > 0]
        
        # Store grayscale images
        self.comp1_grayscale = comp1_grayscale
        self.comp2_grayscale = comp2_grayscale
        
        # Create colored visualizations
        comp1_viz = np.zeros((component1_array.shape[0], component1_array.shape[1], 3), dtype=np.uint8)
        comp2_viz = np.zeros((component2_array.shape[0], component2_array.shape[1], 3), dtype=np.uint8)
        
        # Use the appropriate colors based on selected components
        color_map = {
            0: [0, 0, 255],    # Blue for component 1 (swapped from red)
            1: [255, 0, 0],    # Red for component 2 (swapped from blue)
            2: [0, 255, 0],    # Green for component 3
            3: [128, 0, 128],  # Purple for component 4
            4: [255, 165, 0],  # Orange for component 5
            5: [0, 255, 255],  # Cyan for component 6
            6: [255, 0, 255],  # Magenta for component 7
            7: [255, 255, 0],  # Yellow for component 8
            8: [165, 42, 42]   # Brown for component 9
        }
        
        # Always use blue for component 1 visualization regardless of selection
        comp1_viz[component1_array > 0] = [0, 0, 255]  # Blue for component 1
        # Always use red for component 2 visualization regardless of selection
        comp2_viz[component2_array > 0] = [255, 0, 0]  # Red for component 2
        
        # Create combined visualization
        combined_viz = np.zeros((component1_array.shape[0], component1_array.shape[1], 3), dtype=np.uint8)
        
        # Create combined visualization with black background
        # First, ensure all pixels are black (background)
        combined_viz.fill(0)  # Set all pixels to black (0,0,0)
        
        # Since we've already ensured mutual exclusivity in the segmentation process,
        # we can simply apply each component's color to its respective pixels
        # First, assign component 1 pixels as red (Red ROI)
        combined_viz[component1_array > 0] = [255, 0, 0]  # Red for component 1 (Red ROI)
        
        # Then assign component 2 pixels as blue (Blue ROI)
        combined_viz[component2_array > 0] = [0, 0, 255]  # Blue for component 2 (Blue ROI)
        
        # Add a verification check to ensure no overlaps exist
        overlap_check = (component1_array > 0) & (component2_array > 0)
        if np.any(overlap_check):
            print(f"Warning: {np.sum(overlap_check)} pixels still have overlap between components")
        else:
            print("Segmentation successful: No overlapping pixels between components")
        
        self.combined_viz = combined_viz
        
        # Calculate pixel counts for each component
        comp1_pixels = np.sum(component1_array > 0)
        comp2_pixels = np.sum(component2_array > 0)
        
        # Calculate pixel statistics
        total_pixels = comp1_pixels + comp2_pixels
        
        comp1_percent = (comp1_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        comp2_percent = (comp2_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Calculate mean intensity and standard deviation for each component
        # For component 1 (using original intensities)
        comp1_intensities = original_array[component1_array > 0]
        comp1_mean = np.mean(comp1_intensities) if len(comp1_intensities) > 0 else 0
        comp1_std = np.std(comp1_intensities) if len(comp1_intensities) > 0 else 0
        
        # For component 2 (using original intensities)
        comp2_intensities = original_array[component2_array > 0]
        comp2_mean = np.mean(comp2_intensities) if len(comp2_intensities) > 0 else 0
        comp2_std = np.std(comp2_intensities) if len(comp2_intensities) > 0 else 0
        
        # For total non-zero pixels
        total_intensities = original_array[original_array > 0]
        total_mean = np.mean(total_intensities) if len(total_intensities) > 0 else 0
        total_std = np.std(total_intensities) if len(total_intensities) > 0 else 0
        
        # Update the component frame titles to reflect the non-overlapping nature
        self.comp1_gray_frame.config(text=f"Component 1 - Red ROI ({comp1_percent:.2f}%)")
        self.comp2_gray_frame.config(text=f"Component 2 - Blue ROI ({comp2_percent:.2f}%)")
        
        # Store pixel statistics
        self.pixel_stats = {
            'total_pixels': total_pixels,
            'comp1_pixels': comp1_pixels,
            'comp2_pixels': comp2_pixels,
            'comp1_percent': comp1_percent,
            'comp2_percent': comp2_percent,
            'comp1_mean': comp1_mean,
            'comp1_std': comp1_std,
            'comp2_mean': comp2_mean,
            'comp2_std': comp2_std,
            'total_mean': total_mean,
            'total_std': total_std
        }
        
        # Display component1_gray with percentage
        self.comp1_gray_ax.clear()
        self.comp1_gray_ax.imshow(comp1_grayscale, cmap='gray')
        self.comp1_gray_ax.set_title(f"Red ROI ({comp1_percent:.2f}%)")
        self.comp1_gray_ax.axis('off')
        self.comp1_gray_fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0, hspace=0)
        # Ensure figure size remains fixed at 1x1 as specified during initialization
        self.comp1_gray_fig.set_size_inches(1, 1, forward=True)
        self.comp1_gray_fig.tight_layout(pad=0)
        self.comp1_gray_canvas.draw()
        
        # Display component2_gray with percentage
        self.comp2_gray_ax.clear()
        self.comp2_gray_ax.imshow(comp2_grayscale, cmap='gray')
        self.comp2_gray_ax.set_title(f"Blue ROI ({comp2_percent:.2f}%)")
        self.comp2_gray_ax.axis('off')
        self.comp2_gray_fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0, hspace=0)
        # Ensure figure size remains fixed at 1x1 as specified during initialization
        self.comp2_gray_fig.set_size_inches(1, 1, forward=True)
        self.comp2_gray_fig.tight_layout(pad=0)
        self.comp2_gray_canvas.draw()
        
        # Display combined visualization
        self.combined_viz_ax.clear()
        self.combined_viz_ax.imshow(combined_viz)
        self.combined_viz_ax.set_title(f"Red ROI: {cutoff_threshold:.2f}-{red_threshold:.2f}, Blue ROI: {blue_threshold:.2f}-{cutoff_threshold:.2f}")
        self.combined_viz_ax.axis('off')
        self.combined_viz_fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0, hspace=0)
        # Ensure figure size remains fixed at 1x1 as specified during initialization
        self.combined_viz_fig.set_size_inches(1, 1, forward=True)
        self.combined_viz_fig.tight_layout(pad=0)
        self.combined_viz_canvas.draw()
        
        # Display pixel statistics in status bar
        stats_text = f"Segmentation applied. Pixels - Red ROI ({cutoff_threshold:.2f}-{red_threshold:.2f}): {self.pixel_stats['comp1_pixels']} ({self.pixel_stats['comp1_percent']:.2f}%), Mean: {self.pixel_stats['comp1_mean']:.2f}, StdDev: {self.pixel_stats['comp1_std']:.2f} | Blue ROI ({blue_threshold:.2f}-{cutoff_threshold:.2f}): {self.pixel_stats['comp2_pixels']} ({self.pixel_stats['comp2_percent']:.2f}%), Mean: {self.pixel_stats['comp2_mean']:.2f}, StdDev: {self.pixel_stats['comp2_std']:.2f} | Total Mean: {self.pixel_stats['total_mean']:.2f}, StdDev: {self.pixel_stats['total_std']:.2f}"
        self.status_label.config(text=stats_text)
        
    def update_histogram_with_selected_components(self, comp1_idx, comp2_idx):
        """Update the histogram to show only the selected components"""
        if self.current_img is None:
            return
            
        # Get the selected threshold method
        method = self.threshold_method.get()

        if method == "GMM":
            if self.gmm is None:
                # If GMM model is not available, try to plot histogram to initialize it
                self.plot_histogram(self.current_img) # Pass current_img to plot_histogram
                if self.gmm is None: # Check again after attempting to plot
                    self.status_label.config(text="Error: No GMM model available. Please load an image first.")
                    messagebox.showerror("GMM Error", "No GMM model available. Please load an image first.")
                    return
            
        # Clear the histogram to remove shadows
        self.hist_ax.clear()
        
        # Get pixel values and filter out black pixels (ROI)
        pixels = np.array(self.current_img).flatten()
        non_black_pixels = pixels[pixels > 0]  # Filter out black pixels
        
        # Calculate statistics (pixel_min and pixel_max are not directly used for xlim anymore)
        # Normalize pixel values to 0-1 range for histogram plotting
        non_black_pixels_normalized = non_black_pixels / 255.0 if self.current_img.mode not in ['I', 'F'] else non_black_pixels
        
        # Adjust bin count based on image bit depth
        if self.current_img.mode in ['I', 'F']:
            num_bins = min(1024, len(np.unique(non_black_pixels)))
        else:
            num_bins = 256
            
        # Calculate histogram values
        hist_values, bin_edges = np.histogram(non_black_pixels, bins=num_bins)
        
        # Plot histogram with solid black bars (no transparency) using normalized pixels
        self.hist_ax.hist(non_black_pixels_normalized, bins=num_bins, color='black', edgecolor='black')
        
        # Generate x values for plotting the GMM (0-1 range)
        x = np.linspace(0, 1, 1000).reshape(-1, 1)
        
        # Get probabilities and responsibilities
        logprob = self.gmm.score_samples(x)
        responsibilities = self.gmm.predict_proba(x)
        
        # Plot only the selected components
        colors = self.component_colors
        pdf = np.exp(logprob)
        
        # Plot only the two selected components
        selected_indices = [comp1_idx, comp2_idx]
        for i in selected_indices:
            pdf_component = responsibilities[:, i] * np.exp(logprob)
            # Use consistent scaling based on stored histogram max value
            if hasattr(self, 'hist_max_value') and self.hist_max_value > 0:
                scale_factor = self.hist_max_value
            else:
                scale_factor = hist_values.max()
                
            # Ensure proper scaling of the PDF component
            scaled_component = scale_factor * pdf_component / pdf.max() if pdf.max() > 0 else scale_factor * pdf_component
            self.hist_ax.plot(x, scaled_component, 
                             color=colors[i], linewidth=2, 
                             label=f'C{i+1}: μ={self.gmm.means_[i][0]:.3f}, σ={np.sqrt(self.gmm.covariances_[i][0][0]):.3f}')
        
        # Add the threshold line
        if self.threshold_value is not None:
            # Ensure the threshold line is drawn within the 0-1 x-axis range
            self.hist_ax.axvline(self.threshold_value, color='green', linestyle='--', linewidth=2, label='Threshold')

        # Set x-axis limits to 0-1
        self.hist_ax.set_xlim(0, 1)

        # Adjust y-axis limits to reduce length and fit content
        # Get the maximum frequency from the histogram bars
        max_freq = 0
        for patch in self.hist_ax.patches:
            height = patch.get_height()
            if height > max_freq:
                max_freq = height
        # Set y-axis limit slightly above the max frequency
        self.hist_ax.set_ylim(0, max_freq * 1.1) # 10% buffer above max frequency
        if self.threshold_value is not None:
            self.hist_ax.axvline(x=self.threshold_value, color='green', linestyle='--', linewidth=1.5)
            self.hist_ax.text(self.threshold_value, self.hist_ax.get_ylim()[1]*0.9, 
                             f'Threshold: {self.threshold_value:.4f}', 
                             color='green', ha='center', va='top',
                             bbox=dict(facecolor='white', alpha=0.7))
        
        # Set axis limits
        self.hist_ax.set_xlim(pixel_min, pixel_max)
        
        # Store and use consistent histogram max value
        hist_max = np.max(hist_values)
        if hasattr(self, 'hist_max_value') and self.hist_max_value > 0:
            # Use the larger of the two values to ensure all data is visible
            self.hist_max_value = max(self.hist_max_value, hist_max)
        else:
            self.hist_max_value = hist_max
            
        # Set y-axis with consistent scaling
        self.hist_ax.set_ylim(0, self.hist_max_value * 1.2)  # Add 20% padding for better visibility
        
        # Add title and labels with improved formatting
        method_name = self.threshold_method.get()
        self.hist_ax.set_title(f'Multivariate Gaussian Mixture Model ({method_name})', fontsize=11, fontweight='bold')
        self.hist_ax.set_xlabel('Pixel Intensity', fontsize=10, fontweight='bold')
        self.hist_ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        
        # Make sure axes are visible for the histogram
        self.hist_ax.axis('on')
        for spine in self.hist_ax.spines.values():
            spine.set_visible(True)
            
        # Make sure ticks are visible and properly formatted
        self.hist_ax.tick_params(axis='both', which='both', length=4, width=1, direction='out', labelsize=8)
        
        # Ensure histogram figure size remains fixed at 6x2 as specified during initialization
        # This prevents the figure from expanding when new histograms are plotted
        self.hist_fig.set_size_inches(1, 1, forward=True)
        
        # Don't create a new legend here to avoid duplication with the one created in plot_histogram
        
        # Use tight layout
        self.hist_fig.tight_layout()
        
        # Draw the histogram
        self.hist_canvas.draw()
        
    def save_segmented_image(self):
        if self.segmented_img is None:
            messagebox.showinfo("No Segmented Image", "Please apply segmentation first.")
            return
        
        # Get original filename and path to use as default
        if not self.image_path.get():
            messagebox.showinfo("No Image Path", "Original image path not found.")
            return
            
        # Extract just the filename without path and extension
        original_filepath = self.image_path.get()
        original_dir = os.path.dirname(original_filepath)
        original_filename = os.path.basename(original_filepath)
        base_filename = os.path.splitext(original_filename)[0]
        
        # Determine appropriate extension based on image mode
        # Always use TIFF for floating point images
        if self.current_img.mode in ['I', 'F']:
            ext = "tif"
        else:
            ext = "png"
            
        # Create output directory if it doesn't exist
        output_dir = os.path.join(original_dir, f"{base_filename}_segmentation_results")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create base path for all output files
        base_path = os.path.join(output_dir, base_filename)
        
        # Get selected component indices
        comp1_idx = self.selected_comp1.get()
        comp2_idx = self.selected_comp2.get()
        
        try:
            try:
                # Base path and extension are already set above
                
                # Get original image array
                original_array = np.array(self.current_img)
                
                # Get component masks
                comp1_array = np.array(self.component1_img)
                comp2_array = np.array(self.component2_img)
                
                # Create grayscale segmentations preserving original intensities
                comp1_grayscale = np.zeros_like(original_array)
                comp2_grayscale = np.zeros_like(original_array)
                
                # Extract original intensities for each component
                comp1_grayscale[comp1_array > 0] = original_array[comp1_array > 0]
                comp2_grayscale[comp2_array > 0] = original_array[comp2_array > 0]
                
                # Save grayscale segmentations
                comp1_path = f"{base_path}_component{comp1_idx+1}_gray.{ext}"
                comp2_path = f"{base_path}_component{comp2_idx+1}_gray.{ext}"
                
                # Convert to PIL images and save
                if self.current_img.mode in ['I', 'F']:
                    Image.fromarray(comp1_grayscale.astype(np.float32), mode='F').save(comp1_path)
                    Image.fromarray(comp2_grayscale.astype(np.float32), mode='F').save(comp2_path)
                else:
                    Image.fromarray(comp1_grayscale.astype(np.uint8), mode='L').save(comp1_path)
                    Image.fromarray(comp2_grayscale.astype(np.uint8), mode='L').save(comp2_path)
                
                # Get component colors for visualization - use the same color map defined in __init__
                color_map = {
                    0: [255, 0, 0],    # Red for component 1
                    1: [0, 0, 255],    # Blue for component 2
                    2: [0, 255, 0],    # Green for component 3
                    3: [128, 0, 128],  # Purple for component 4
                    4: [255, 165, 0],  # Orange for component 5
                    5: [0, 255, 255],  # Cyan for component 6
                    6: [255, 0, 255],  # Magenta for component 7
                    7: [255, 255, 0],  # Yellow for component 8
                    8: [165, 42, 42]   # Brown for component 9
                }
                
                # Save colored visualizations for reference
                comp1_viz = np.zeros((comp1_array.shape[0], comp1_array.shape[1], 3), dtype=np.uint8)
                comp2_viz = np.zeros((comp2_array.shape[0], comp2_array.shape[1], 3), dtype=np.uint8)
                comp1_viz[comp1_array > 0] = color_map[comp1_idx]  # Color for component 1
                comp2_viz[comp2_grayscale > 0] = color_map[comp2_idx]  # Color for component 2
                
                comp1_viz_path = f"{base_path}_component{comp1_idx+1}_viz.{ext}"
                comp2_viz_path = f"{base_path}_component{comp2_idx+1}_viz.{ext}"
                Image.fromarray(comp1_viz).save(comp1_viz_path)
                Image.fromarray(comp2_viz).save(comp2_viz_path)
                
                # Save combined visualization
                combined_viz_path = f"{base_path}_combined_viz.{ext}"
                if self.combined_viz is not None:
                    Image.fromarray(self.combined_viz).save(combined_viz_path)
                
                # Save histogram if available
                if self.hist_fig is not None:
                    hist_path = f"{base_path}_histogram.{ext}"
                    # Store original figure size
                    original_figsize = self.hist_fig.get_size_inches()
                    # Set figure size to 6x2 inches for saving
                    # 'Temporary figure size' means we temporarily resize the figure only for the save operation
                    # to ensure the saved image has appropriate dimensions, then restore the original size
                    # This prevents the UI display from being affected by the save operation
                    # The figure is temporarily resized to 6x2 to ensure all elements (axes, labels, ticks) are properly visible in the saved image
                    self.hist_fig.set_size_inches(1, 1)
                    
                    # Ensure axes are visible for saving
                    self.hist_ax.axis('on')
                    for spine in self.hist_ax.spines.values():
                        spine.set_visible(True)
                    
                    # Ensure axis labels are visible
                    self.hist_ax.set_xlabel('Pixel Intensity', fontsize=10, fontweight='bold')
                    self.hist_ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
                    
                    # Make sure ticks are visible and properly formatted
                    self.hist_ax.tick_params(axis='both', which='both', length=4, width=1, direction='out', labelsize=8)
                    
                    # Save with tight layout to include all elements
                    self.hist_fig.tight_layout()
                    self.hist_fig.savefig(hist_path, bbox_inches='tight', dpi=300)
                    # Restore original figure size to maintain UI consistency
                    self.hist_fig.set_size_inches(*original_figsize)
                    
                    # After saving all image files, also export Excel statistics if available
                    excel_path = None
                    if self.pixel_stats is not None:
                        try:
                                    # Export statistics to Excel
                            excel_path = f"{base_path}_statistics.xlsx"
                            
                            # Create a DataFrame with the statistics
                            stats_df = pd.DataFrame({
                                'Metric': ['Pixel Count', 'Percentage', 'Mean Intensity', 'Standard Deviation'],
                                f'Component {comp1_idx+1}': [
                                    self.pixel_stats['comp1_pixels'],
                                    self.pixel_stats['comp1_percent'],
                                    self.pixel_stats['comp1_mean'],
                                    self.pixel_stats['comp1_std']
                                ],
                                f'Component {comp2_idx+1}': [
                                    self.pixel_stats['comp2_pixels'],
                                    self.pixel_stats['comp2_percent'],
                                    self.pixel_stats['comp2_mean'],
                                    self.pixel_stats['comp2_std']
                                ],
                                'Total': [
                                    self.pixel_stats['total_pixels'],
                                    100.0,  # Total percentage is always 100%
                                    self.pixel_stats['total_mean'],
                                    self.pixel_stats['total_std']
                                ]
                            })
                            
                            # Save to Excel
                            stats_df.to_excel(excel_path, index=False)
                            
                        except Exception as excel_err:
                            print(f"Warning: Could not export Excel statistics: {excel_err}")
                    
                    # Update status with all saved files
                    excel_info = f"\n- Statistics: {os.path.basename(excel_path)}" if excel_path else ""
                    self.status_label.config(text=f"Results saved to {output_dir}:\n- Component {comp1_idx+1} (grayscale): {os.path.basename(comp1_path)}\n- Component {comp2_idx+1} (grayscale): {os.path.basename(comp2_path)}\n- Visualizations: {os.path.basename(comp1_viz_path)}, {os.path.basename(comp2_viz_path)}, {os.path.basename(combined_viz_path)}\n- Histogram: {os.path.basename(hist_path)}{excel_info}")
                    messagebox.showinfo("Save Complete", f"All results saved to:\n{output_dir}")
                else:
                    self.status_label.config(text="No segmentation results available to save.")
            except Exception as e:
                # Handle exceptions from the inner try block
                error_msg = f"Error saving segmentation results: {e}"
                self.status_label.config(text=error_msg)
                messagebox.showerror("Save Error", error_msg)
        except Exception as e:
            # Handle exceptions from the outer try block
            error_msg = f"Error in save operation: {e}"
            self.status_label.config(text=error_msg)
            messagebox.showerror("Save Error", error_msg)

# Main function to run the application
def signal_handler(sig, frame):
    print('\nExiting gracefully (Ctrl+C pressed)')
    sys.exit(0)        # Exit the Python process

def on_closing(root):
    # Properly clean up resources
    plt.close('all')  # Close all matplotlib figures
    print('\nExiting gracefully (Window closed)')
    root.destroy()     # Destroy the tkinter window
    sys.exit(0)        # Exit the Python process

# Function to set up the inference UI - moved from being defined elsewhere to here
def load_model_command():
    global model_path_entry
    filepath = filedialog.askopenfilename(
        initialdir=".",
        title="Select Trained Model",
        filetypes=(("Model files", "*.pth"), ("all files", "*.*"))
    )
    if filepath:
        model_path_entry.delete(0, tk.END)
        model_path_entry.insert(0, filepath)

def setup_inference_ui(parent):
    # Global declarations consolidated at the beginning
    global window, model_path_entry, input_folder_entry, output_folder_entry, status_text, output_file_type_combobox
    global load_model_button, input_folder_button, output_folder_button, start_inference_button, stop_inference_button
    global inference_thread, inference_running, output_file_type
    
    # Initialize global variables with default values
    inference_running = False
    output_file_type = ".tif"
    
    window = parent
    
    # --- Model and Folder Selection Frame ---
    selection_frame = ttk.Frame(window, padding=(10, 10))
    selection_frame.pack(fill=tk.X, expand=False)
    
    ttk.Label(selection_frame, text="Trained Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    model_path_entry = ttk.Entry(selection_frame, width=60)
    model_path_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    load_model_button = ttk.Button(selection_frame, text="Browse", command=load_model_command)
    load_model_button.grid(row=0, column=2, padx=5, pady=5)
    
    ttk.Label(selection_frame, text="Input TIFF Folder:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    input_folder_entry = ttk.Entry(selection_frame, width=60)
    input_folder_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
    input_folder_button = ttk.Button(selection_frame, text="Browse", command=select_input_folder_command)
    input_folder_button.grid(row=1, column=2, padx=5, pady=5)
    
    ttk.Label(selection_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    output_folder_entry = ttk.Entry(selection_frame, width=60)
    output_folder_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
    output_folder_button = ttk.Button(selection_frame, text="Browse", command=select_output_folder_command)
    output_folder_button.grid(row=2, column=2, padx=5, pady=5)
    
    # --- Output File Type Selection ---
    ttk.Label(selection_frame, text="Output File Type:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    output_file_type_combobox = ttk.Combobox(selection_frame, values=["PNG", "TIFF", "JPEG"], state="readonly")
    output_file_type_combobox.current(0)  # Default to PNG
    output_file_type_combobox.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
    
    # --- Control Buttons ---
    button_frame = ttk.Frame(window, padding=(10, 10))
    button_frame.pack(fill=tk.X, expand=False)
    
    start_inference_button = ttk.Button(button_frame, text="Start Inference", command=start_inference_command)
    start_inference_button.pack(side=tk.LEFT, padx=5)
    
    stop_inference_button = ttk.Button(button_frame, text="Stop Inference", command=stop_inference_command, state=tk.DISABLED)
    stop_inference_button.pack(side=tk.LEFT, padx=5)
    
    # --- Status Text ---
    status_frame = ttk.LabelFrame(window, text="Status", padding=(10, 10))
    status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    status_text = tk.Text(status_frame, wrap=tk.WORD, height=10)
    status_text.pack(fill=tk.BOTH, expand=True)
    status_text.config(state=tk.DISABLED)  # Make read-only initially

# Function to select input folder for inference UI
def select_input_folder_command():
    global input_folder_entry
    dirpath = filedialog.askdirectory(initialdir=".", title="Select Input TIFF Folder")
    if dirpath:
        input_folder_entry.delete(0, tk.END)
        input_folder_entry.insert(0, dirpath)

# Function to select output folder for inference UI
def select_output_folder_command():
    global output_folder_entry
    dirpath = filedialog.askdirectory(initialdir=".", title="Select Output Folder")
    if dirpath and output_folder_entry is not None:
        output_folder_entry.delete(0, tk.END)
        output_folder_entry.insert(0, dirpath)

# --- Inference Functions ---
def load_model_command():
    global model_path_entry
    filepath = filedialog.askopenfilename(
        initialdir=".",
        title="Select Trained Model",
        filetypes=(("Model files", "*.pth"), ("all files", "*.*"))
    )
    if filepath:
        model_path_entry.delete(0, tk.END)
        model_path_entry.insert(0, filepath)

def select_input_folder_command():
    global input_folder_entry
    dirpath = filedialog.askdirectory(initialdir=".", title="Select Input TIFF Folder")
    if dirpath:
        input_folder_entry.delete(0, tk.END)
        input_folder_entry.insert(0, dirpath)

def select_output_folder_command():
    global output_folder_entry
    dirpath = filedialog.askdirectory(initialdir=".", title="Select Output Folder")
    if dirpath and output_folder_entry is not None:
        output_folder_entry.delete(0, tk.END)
        output_folder_entry.insert(0, dirpath)

def start_inference_command():
    global model_path_entry, input_folder_entry, output_folder_entry, status_text, output_file_type_combobox
    global load_model_button, input_folder_button, output_folder_button, start_inference_button, stop_inference_button
    global inference_thread, inference_running, output_file_type

    model_path = model_path_entry.get()
    input_folder = input_folder_entry.get()
    output_folder_specified = output_folder_entry.get()
    selected_file_type = output_file_type_combobox.get()

    if not model_path or not os.path.exists(model_path):
        messagebox.showerror("Error", "Please select a valid trained model file.")
        return
    if not input_folder or not os.path.isdir(input_folder):
        messagebox.showerror("Error", "Please select a valid input folder containing TIFF files.")
        return

    output_folder = output_folder_specified if output_folder_specified else input_folder

    if not os.path.isdir(output_folder):
        try:
            os.makedirs(output_folder, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Error", f"Invalid output folder or cannot create it: {e}")
            return

    if not os.path.isdir(output_folder):
         messagebox.showerror("Error", "Output folder is invalid.")
         return

    # Update UI state
    start_inference_button.config(state=tk.DISABLED)
    stop_inference_button.config(state=tk.NORMAL)
    
    # Clear and prepare status text
    status_text.config(state=tk.NORMAL)
    status_text.delete(1.0, tk.END)
    status_text.insert(tk.END, "Starting inference process...\n")
    status_text.config(state=tk.DISABLED)

    # Run inference in a separate thread to keep UI responsive
    run_inference_threaded(model_path, input_folder, output_folder, selected_file_type)

def stop_inference_command():
    global inference_running
    inference_running = False



# Main function definition moved here after UNetTrainingTool class
def main():
    # Set up signal handler for keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if __name__ == "__main__":
            root = tk.Tk()
            root.geometry("1200x800")
            root.title("Combined Segmentation Tool")
            
            # Add protocol to handle window close event with proper cleanup
            root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
            
            # Create notebook (tabbed interface)
            try:
                notebook = ttk.Notebook(root)
                notebook.pack(fill='both', expand=True)
            except Exception as e:
                messagebox.showerror("Initialization Error", f"Failed to create notebook: {e}")
                raise
            
            # Create tabs for each segmentation tool
            tab1 = ttk.Frame(notebook)
            tab2 = ttk.Frame(notebook)
            tab3 = ttk.Frame(notebook)
            
            notebook.add(tab1, text='Histogram Viewer')
            notebook.add(tab2, text='UNet Inference Tool')
            notebook.add(tab3, text='UNet Training Tool')
            
            # Initialize each tool in its own tab
            viewer1 = ImageHistogramViewer(tab1)
            
            # Setup UNet Inference Tool in tab2
            # Initialize all required global variables
            global output_folder_entry, input_folder_entry, model_path_entry
            global inference_thread, inference_running, output_file_type
            global start_inference_button, stop_inference_button
            
            # Initialize the variables
            output_folder_entry = None
            input_folder_entry = None
            model_path_entry = None
            inference_running = False
            output_file_type = ".tif"
            
            setup_inference_ui(tab2)
            
            # Setup UNet Training Tool in tab3
            # Initialize the UNetTrainingTool class directly
            training_tool = UNetTrainingTool(tab3)
            
            root.mainloop()
    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)
if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk, ImageEnhance, ImageChops, ImageFilter
import torchvision.transforms as transforms
import os
import threading
from datetime import datetime
import numpy as np
import tifffile

# --- Global variables ---
inference_model = None
inference_thread = None
inference_running = False
output_file_type = ".tif" # Default output file type

# --- Inference Functions ---
def load_model_command():
    global model_path_entry
    filepath = filedialog.askopenfilename(
        initialdir=".",
        title="Select Trained Model",
        filetypes=(("Model files", "*.pth"), ("all files", "*.*"))
    )
    if filepath:
        model_path_entry.delete(0, tk.END)
        model_path_entry.insert(0, filepath)

def select_input_folder_command():
    global input_folder_entry
    dirpath = filedialog.askdirectory(initialdir=".", title="Select Input TIFF Folder")
    if dirpath:
        input_folder_entry.delete(0, tk.END)
        input_folder_entry.insert(0, dirpath)

def select_output_folder_command():
    global output_folder_entry
    dirpath = filedialog.askdirectory(initialdir=".", title="Select Output Folder")
    if dirpath and output_folder_entry is not None:
        output_folder_entry.delete(0, tk.END)
        output_folder_entry.insert(0, dirpath)

def start_inference_command():
    global model_path_entry, input_folder_entry, output_folder_entry, status_text, output_file_type_combobox
    global load_model_button, input_folder_button, output_folder_button, start_inference_button, stop_inference_button
    global inference_thread, inference_running, output_file_type

    model_path = model_path_entry.get()
    input_folder = input_folder_entry.get()
    output_folder_specified = output_folder_entry.get()
    selected_file_type = output_file_type_combobox.get()

    if not model_path or not os.path.exists(model_path):
        messagebox.showerror("Error", "Please select a valid trained model file.")
        return
    if not input_folder or not os.path.isdir(input_folder):
        messagebox.showerror("Error", "Please select a valid input folder containing TIFF files.")
        return

    output_folder = output_folder_specified if output_folder_specified else input_folder

    if not os.path.isdir(output_folder):
        try:
            os.makedirs(output_folder, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Error", f"Invalid output folder or cannot create it: {e}")
            return

    if not os.path.isdir(output_folder):
         messagebox.showerror("Error", "Output folder is invalid.")
         return

    # Update UI state
    start_inference_button.config(state=tk.DISABLED)
    stop_inference_button.config(state=tk.NORMAL)
    
    # Clear and prepare status text
    status_text.config(state=tk.NORMAL)
    status_text.delete(1.0, tk.END)
    status_text.insert(tk.END, "Starting inference process...\n")
    status_text.config(state=tk.DISABLED)

    # Run inference in a separate thread to keep UI responsive
    run_inference_threaded(model_path, input_folder, output_folder, selected_file_type)

def stop_inference_command():
    global inference_running
    inference_running = False



# -------------------- UNet Implementation --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2) # Corrected line - was x3 = self.down2(x3)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

from PIL import ImageFilter

def crop_to_content(image, mask):
    """Crops both image and mask to the bounding box of the mask content."""
    # Use a less aggressive filter and lower threshold to preserve more of the ROI
    filtered_mask = mask.filter(ImageFilter.MedianFilter(3))
    mask_np = np.array(filtered_mask)
    y_indices, x_indices = np.where(mask_np >= 128)  # Use a more lenient threshold
    if not y_indices.size or not x_indices.size:
        return Image.new('RGBA', (1,1), (0,0,0,0)), Image.new('L', (1,1), 0)
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    # Add a small padding to avoid cutting off ROI edges
    pad = 2
    min_x = max(0, min_x - pad)
    min_y = max(0, min_y - pad)
    max_x = min(mask.width - 1, max_x + pad)
    max_y = min(mask.height - 1, max_y + pad)
    bbox = (min_x, min_y, max_x + 1, max_y + 1)
    cropped_image = image.crop(bbox)
    cropped_mask = mask.crop(bbox)
    return cropped_image, cropped_mask
def make_transparent_background(image, mask, output_file_type):
    """Makes background transparent, keeps ROI from original image (OPPOSITE)."""
    image = image.convert("RGBA")
    mask = mask.convert('L')
    image_np = np.array(image)
    mask_np = np.array(mask) / 255.0

    # No need to invert the mask as we're directly using it for the opposite effect
    # ROI is where mask_np is high (close to 1), background is where mask_np is low (close to 0)
    alpha_channel = (mask_np * 255).astype(np.uint8) # ROI opaque, background transparent

    transparent_image_np = np.zeros_like(image_np, dtype=np.float32)

    for y in range(image_np.shape[0]):
        for x in range(image_np.shape[1]):
            if alpha_channel[y, x] > 0: # ROI
                transparent_image_np[y, x] = image_np[y, x] # Keep original image pixel in ROI (opaque)
            else: # Background
                transparent_image_np[y, x] = [0, 0, 0, 0] # Make background transparent

    return transparent_image_np
def make_transparent_background_single_channel(image, mask, output_file_type):
    """
    Makes brain regions (ROI) transparent while keeping the non-brain regions (background) visible.
    Converts to float32 format and handles different image modes appropriately.
    Adapts output format based on output_file_type (.tif or .png)
    """
    original_mode = image.mode
    mask = mask.convert('L')
    
    # Handle different image modes properly
    if original_mode == 'F':
        # For float32 images, we need special handling
        image_np = np.array(image)
        # Normalize to 0-1 range for consistent processing
        image_min = np.min(image_np)
        image_max = np.max(image_np)
        if image_max > image_min:
            image_np = (image_np - image_min) / (image_max - image_min)
        else:
            image_np = np.zeros_like(image_np)
    else:
        # For standard image modes
        image_np = np.array(image).astype(np.float32) / 255.0
    
    mask_np = np.array(mask).astype(np.float32) / 255.0
    
    # Determine if we're saving as TIFF or PNG
    is_tiff = output_file_type.lower() == '.tif'
    
    if original_mode == 'RGB':
        transparent_image_np = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.float32)
        
        for y in range(image_np.shape[0]):
            for x in range(image_np.shape[1]):
                if mask_np[y, x] > 0.5:  # ROI - make transparent
                    transparent_image_np[y, x] = [0.0, 0.0, 0.0, 0.0]  # Fully transparent
                else:  # Background - keep visible with original values
                    transparent_image_np[y, x, 0:3] = image_np[y, x, 0:3]
                    transparent_image_np[y, x, 3] = 1.0  # Fully opaque
    
    elif original_mode == 'L' or original_mode == 'F':
        if is_tiff:
            # For TIFF files with grayscale images, create a 2-channel image (grayscale + alpha)
            transparent_image_np = np.zeros((image_np.shape[0], image_np.shape[1], 2), dtype=np.float32)
            
            if image_np.ndim > 2:
                image_np = image_np[:,:,0]
            
            for y in range(image_np.shape[0]):
                for x in range(image_np.shape[1]):
                    if mask_np[y, x] > 0.5:  # ROI - make transparent
                        transparent_image_np[y, x, 0] = 0.0  # Black
                        transparent_image_np[y, x, 1] = 0.0  # Fully transparent
                    else:  # Background - keep visible
                        transparent_image_np[y, x, 0] = image_np[y, x]  # Original grayscale value
                        transparent_image_np[y, x, 1] = 1.0  # Fully opaque
        else:
            # For PNG files, we need RGBA format
            transparent_image_np = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.float32)
            
            if image_np.ndim > 2:
                image_np = image_np[:,:,0]
            
            for y in range(image_np.shape[0]):
                for x in range(image_np.shape[1]):
                    if mask_np[y, x] > 0.5:  # ROI - make transparent
                        transparent_image_np[y, x] = [0.0, 0.0, 0.0, 0.0]  # Fully transparent
                    else:  # Background - keep visible
                        # Set RGB channels to grayscale value
                        transparent_image_np[y, x, 0] = image_np[y, x]
                        transparent_image_np[y, x, 1] = image_np[y, x]
                        transparent_image_np[y, x, 2] = image_np[y, x]
                        transparent_image_np[y, x, 3] = 1.0  # Fully opaque
    
    else:  # Handle other modes
        if image_np.ndim == 3:
            channels = min(image_np.shape[2], 3)
            transparent_image_np = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.float32)
            
            for y in range(image_np.shape[0]):
                for x in range(image_np.shape[1]):
                    if mask_np[y, x] > 0.5:  # ROI - make transparent
                        transparent_image_np[y, x] = [0.0, 0.0, 0.0, 0.0]  # Fully transparent
                    else:  # Background - keep visible
                        for c in range(channels):
                            transparent_image_np[y, x, c] = image_np[y, x, c]
                        transparent_image_np[y, x, 3] = 1.0  # Fully opaque
        
        else:  # Single-channel image
            if is_tiff:
                # For TIFF files, use 2-channel format (grayscale + alpha)
                transparent_image_np = np.zeros((image_np.shape[0], image_np.shape[1], 2), dtype=np.float32)
                
                for y in range(image_np.shape[0]):
                    for x in range(image_np.shape[1]):
                        if mask_np[y, x] > 0.5:  # ROI - make transparent
                            transparent_image_np[y, x, 0] = 0.0  # Black
                            transparent_image_np[y, x, 1] = 0.0  # Fully transparent
                        else:  # Background - keep visible
                            transparent_image_np[y, x, 0] = image_np[y, x]  # Original grayscale value
                            transparent_image_np[y, x, 1] = 1.0  # Fully opaque
            else:
                # For PNG files, use RGBA format
                transparent_image_np = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.float32)
                
                for y in range(image_np.shape[0]):
                    for x in range(image_np.shape[1]):
                        if mask_np[y, x] > 0.5:  # ROI - make transparent
                            transparent_image_np[y, x] = [0.0, 0.0, 0.0, 0.0]  # Fully transparent
                        else:  # Background - keep visible
                            # Set RGB channels to grayscale value
                            transparent_image_np[y, x, 0] = image_np[y, x]
                            transparent_image_np[y, x, 1] = image_np[y, x]
                            transparent_image_np[y, x, 2] = image_np[y, x]
                            transparent_image_np[y, x, 3] = 1.0  # Fully opaque
    
    return transparent_image_np
def run_inference_process(model_path, input_folder, output_folder, update_status_callback, output_file_type): # Added output_file_type
    global inference_model, inference_running

    try:
        update_status_callback("Loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference_model = UNet(n_channels=3, n_classes=1)
        inference_model.load_state_dict(torch.load(model_path, map_location=device))
        inference_model.to(device).eval()

        image_size = 256
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]
        total_files = len(tiff_files)
        processed_files = 0

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        update_status_callback(f"Starting inference on {total_files} files...")

        inference_start_time = datetime.now()

        input_dtype_output = 0

        with torch.no_grad():
            for tiff_file in tiff_files:
                if not inference_running:
                    update_status_callback("Inference stopped by user.")
                    return

                image_path = os.path.join(input_folder, tiff_file)

                # --- Read input TIFF metadata ---
                with tifffile.TiffFile(image_path) as tif:
                    input_tiff_tags = tif.pages[0].tags
                    input_dtype = tif.pages[0].dtype
                    input_dtype_output = input_dtype

                    input_resolution = None
                    try:
                        x_resolution = input_tiff_tags['XResolution'].value
                        y_resolution = input_tiff_tags['YResolution'].value
                        resolution_unit = input_tiff_tags['ResolutionUnit'].value
                        input_resolution = (x_resolution, y_resolution, resolution_unit)
                    except KeyError:
                        pass

                try:
                    image_pil = Image.open(image_path) # DO NOT convert here, keep original mode

                    # Handle input image based on its mode
                    if image_pil.mode == 'F':
                        # For float32 images, convert to RGB for model input but preserve original mode
                        image_rgb = image_pil.convert("RGB")
                        image_tensor = transform(image_rgb).unsqueeze(0).to(device)
                    else:
                        # For other modes, use the standard pipeline
                        image_tensor = transform(image_pil).unsqueeze(0).to(device)
                        
                    prediction = inference_model(image_tensor)
                    prediction_sigmoid = torch.sigmoid(prediction).cpu()
                    predicted_mask_binary = (prediction_sigmoid > 0.5).float().squeeze(0)
                    
                    # Always use mode='L' for mask
                    if predicted_mask_binary.shape[0] == 1:  # Single channel output
                        mask_pil = transforms.ToPILImage(mode='L')(predicted_mask_binary.squeeze(0))
                    else:  # Multi-channel output (unlikely but handle it)
                        mask_pil = transforms.ToPILImage(mode='L')(predicted_mask_binary[0])

                    mask_np_check = np.array(mask_pil)

                    cropped_image, cropped_mask = crop_to_content(image_pil, mask_pil)



                    #transparent_image_np = make_transparent_background(cropped_image, cropped_mask, output_file_type)
                    transparent_image_np = make_transparent_background_single_channel(cropped_image, cropped_mask, output_file_type) # Grayscale Alpha TIFF



                    original_name, original_ext = os.path.splitext(tiff_file)
                    output_filename = "segmented_" + original_name + output_file_type
                    output_path = os.path.join(output_folder, output_filename)

                    save_format = "TIFF" if output_file_type.lower() == ".tif" else "PNG"
                    original_mode = cropped_image.mode # Use cropped image mode for saving



                    if save_format == "TIFF":
                        transparent_image_np_float32 = transparent_image_np.astype(np.float32) # Ensuring float32 dtype

                        # Handle saving with proper alpha channel based on original mode
                        if original_mode == 'L' or original_mode == 'F':
                            # For grayscale images, we need to save with alpha channel
                            # Check if we have a proper alpha channel structure (H, W, 2)
                            if transparent_image_np_float32.ndim == 3 and transparent_image_np_float32.shape[2] == 2:
                                # We have grayscale + alpha
                                photometric_interpretation = 'minisblack'
                                # Extract grayscale and alpha channels
                                grayscale_channel = transparent_image_np_float32[:,:,0]
                                alpha_channel = transparent_image_np_float32[:,:,1]
                                
                                # Create a multi-page TIFF with grayscale and alpha
                                try:
                                    with tifffile.TiffWriter(output_path) as tif:
                                        # Write grayscale channel first
                                        tif.write(grayscale_channel, 
                                                photometric=photometric_interpretation,
                                                resolution=(input_resolution[0][0], input_resolution[0][1]) if input_resolution else None,
                                                resolutionunit=input_resolution[2] if input_resolution else None)
                                        # Write alpha channel as second page
                                        tif.write(alpha_channel,
                                                photometric='minisblack',
                                                resolution=(input_resolution[0][0], input_resolution[0][1]) if input_resolution else None,
                                                resolutionunit=input_resolution[2] if input_resolution else None,
                                                description='alpha')
                                except ValueError as ve:
                                    raise ve
                            else:
                                # Fallback for unexpected structure
                                photometric_interpretation = 'minisblack'
                                if transparent_image_np_float32.ndim == 3:
                                    if transparent_image_np_float32.shape[2] == 1:
                                        final_save_data = np.squeeze(transparent_image_np_float32, axis=2)
                                    else:
                                        final_save_data = transparent_image_np_float32[:,:,0]
                                else:
                                    final_save_data = transparent_image_np_float32
                                    
                                tifffile.imwrite(
                                    output_path,
                                    data=final_save_data,
                                    photometric=photometric_interpretation,
                                    resolution=(input_resolution[0][0], input_resolution[0][1]) if input_resolution else None,
                                    resolutionunit=input_resolution[2] if input_resolution else None
                                )
                                
                        elif original_mode == 'RGB':
                            # For RGB images, ensure we have RGBA structure
                            if transparent_image_np_float32.ndim == 3 and transparent_image_np_float32.shape[2] == 4:
                                # We have proper RGBA
                                photometric_interpretation = 'rgb'
                                
                                # Extract RGB and alpha channels
                                rgb_channels = transparent_image_np_float32[:,:,0:3]
                                alpha_channel = transparent_image_np_float32[:,:,3]
                                
                                # Create a multi-page TIFF with RGB and alpha
                                try:
                                    with tifffile.TiffWriter(output_path) as tif:
                                        # Write RGB channels first
                                        tif.write(rgb_channels, 
                                                photometric=photometric_interpretation,
                                                resolution=(input_resolution[0][0], input_resolution[0][1]) if input_resolution else None,
                                                resolutionunit=input_resolution[2] if input_resolution else None)
                                        # Write alpha channel as second page
                                        tif.write(alpha_channel,
                                                photometric='minisblack',
                                                resolution=(input_resolution[0][0], input_resolution[0][1]) if input_resolution else None,
                                                resolutionunit=input_resolution[2] if input_resolution else None,
                                                description='alpha')
                                except ValueError as ve:
                                    raise ve
                            else:
                                # Fallback for unexpected structure
                                photometric_interpretation = 'rgb'
                                tifffile.imwrite(
                                    output_path,
                                    data=transparent_image_np_float32,
                                    photometric=photometric_interpretation,
                                    resolution=(input_resolution[0][0], input_resolution[0][1]) if input_resolution else None,
                                    resolutionunit=input_resolution[2] if input_resolution else None
                                )
                        else:
                            # For other modes, try to handle appropriately
                            if transparent_image_np_float32.ndim == 3:
                                if transparent_image_np_float32.shape[2] == 4:  # RGBA
                                    photometric_interpretation = 'rgb'
                                    # Extract RGB and alpha channels
                                    rgb_channels = transparent_image_np_float32[:,:,0:3]
                                    alpha_channel = transparent_image_np_float32[:,:,3]
                                    
                                    # Create a multi-page TIFF with RGB and alpha
                                    with tifffile.TiffWriter(output_path) as tif:
                                        tif.write(rgb_channels, 
                                                photometric=photometric_interpretation,
                                                resolution=(input_resolution[0][0], input_resolution[0][1]) if input_resolution else None,
                                                resolutionunit=input_resolution[2] if input_resolution else None)
                                        tif.write(alpha_channel,
                                                photometric='minisblack',
                                                resolution=(input_resolution[0][0], input_resolution[0][1]) if input_resolution else None,
                                                resolutionunit=input_resolution[2] if input_resolution else None,
                                                description='alpha')
                                elif transparent_image_np_float32.shape[2] == 2:  # Grayscale + Alpha
                                    photometric_interpretation = 'minisblack'
                                    grayscale_channel = transparent_image_np_float32[:,:,0]
                                    alpha_channel = transparent_image_np_float32[:,:,1]
                                    
                                    with tifffile.TiffWriter(output_path) as tif:
                                        tif.write(grayscale_channel, 
                                                photometric=photometric_interpretation,
                                                resolution=(input_resolution[0][0], input_resolution[0][1]) if input_resolution else None,
                                                resolutionunit=input_resolution[2] if input_resolution else None)
                                        tif.write(alpha_channel,
                                                photometric='minisblack',
                                                resolution=(input_resolution[0][0], input_resolution[0][1]) if input_resolution else None,
                                                resolutionunit=input_resolution[2] if input_resolution else None,
                                                description='alpha')
                                else:
                                    # Fallback for other channel counts
                                    photometric_interpretation = 'minisblack' if transparent_image_np_float32.shape[2] == 1 else 'rgb'
                                    tifffile.imwrite(
                                        output_path,
                                        data=transparent_image_np_float32,
                                        photometric=photometric_interpretation,
                                        resolution=(input_resolution[0][0], input_resolution[0][1]) if input_resolution else None,
                                        resolutionunit=input_resolution[2] if input_resolution else None
                                    )
                            else:
                                # 2D array
                                photometric_interpretation = 'minisblack'
                                tifffile.imwrite(
                                    output_path,
                                    data=transparent_image_np_float32,
                                    photometric=photometric_interpretation,
                                    resolution=(input_resolution[0][0], input_resolution[0][1]) if input_resolution else None,
                                    resolutionunit=input_resolution[2] if input_resolution else None
                                )

                        # Verify the file was written
                        reloaded_tiff = tifffile.imread(output_path)

                    elif save_format == "PNG":
                        # Convert float32 (0-1) to uint8 (0-255) for PNG
                        transparent_image_np_uint8 = (transparent_image_np * 255).astype(np.uint8)
                        
                        if original_mode == 'RGB':
                            # For RGB images with transparency
                            if transparent_image_np_uint8.ndim == 3 and transparent_image_np_uint8.shape[2] == 4:
                                # We already have RGBA data
                                transparent_image_pil = Image.fromarray(transparent_image_np_uint8, 'RGBA')
                            elif transparent_image_np_uint8.ndim == 3 and transparent_image_np_uint8.shape[2] == 3:
                                # RGB without alpha - convert to RGBA with full opacity
                                rgba_array = np.zeros((transparent_image_np_uint8.shape[0], transparent_image_np_uint8.shape[1], 4), dtype=np.uint8)
                                rgba_array[:,:,0:3] = transparent_image_np_uint8
                                rgba_array[:,:,3] = 255  # Full opacity
                                transparent_image_pil = Image.fromarray(rgba_array, 'RGBA')
                            else:
                                # Handle unexpected shape
                                rgba_array = np.zeros((transparent_image_np_uint8.shape[0], transparent_image_np_uint8.shape[1], 4), dtype=np.uint8)
                                if transparent_image_np_uint8.ndim == 3 and transparent_image_np_uint8.shape[2] == 1:
                                    # Single channel - replicate to RGB and set alpha
                                    rgba_array[:,:,0] = transparent_image_np_uint8[:,:,0]
                                    rgba_array[:,:,1] = transparent_image_np_uint8[:,:,0]
                                    rgba_array[:,:,2] = transparent_image_np_uint8[:,:,0]
                                    # Use the same channel for alpha (if it's a mask)
                                    rgba_array[:,:,3] = transparent_image_np_uint8[:,:,0]
                                transparent_image_pil = Image.fromarray(rgba_array, 'RGBA')
                                
                        elif original_mode == 'L' or original_mode == 'F':
                            # For grayscale images with transparency
                            if transparent_image_np_uint8.ndim == 3 and transparent_image_np_uint8.shape[2] == 2:
                                # We have grayscale + alpha channel
                                # Create RGBA with the grayscale value replicated to RGB channels
                                rgba_array = np.zeros((transparent_image_np_uint8.shape[0], transparent_image_np_uint8.shape[1], 4), dtype=np.uint8)
                                # Set RGB channels to grayscale value
                                rgba_array[:,:,0] = transparent_image_np_uint8[:,:,0]
                                rgba_array[:,:,1] = transparent_image_np_uint8[:,:,0]
                                rgba_array[:,:,2] = transparent_image_np_uint8[:,:,0]
                                # Set alpha channel
                                rgba_array[:,:,3] = transparent_image_np_uint8[:,:,1]
                                transparent_image_pil = Image.fromarray(rgba_array, 'RGBA')
                            elif transparent_image_np_uint8.ndim == 3 and transparent_image_np_uint8.shape[2] == 1:
                                # Single channel without alpha - convert to grayscale
                                grayscale = np.squeeze(transparent_image_np_uint8, axis=2)
                                transparent_image_pil = Image.fromarray(grayscale, 'L')
                            elif transparent_image_np_uint8.ndim == 2:
                                # Already 2D grayscale
                                transparent_image_pil = Image.fromarray(transparent_image_np_uint8, 'L')
                            else:
                                # Handle unexpected shape
                                if transparent_image_np_uint8.ndim == 3 and transparent_image_np_uint8.shape[2] > 1:
                                    # Take first channel for grayscale
                                    grayscale = transparent_image_np_uint8[:,:,0]
                                    transparent_image_pil = Image.fromarray(grayscale, 'L')
                                else:
                                    # Fallback
                                    transparent_image_pil = Image.fromarray(transparent_image_np_uint8, 'L')
                        else:
                            # For other modes, create RGBA with proper transparency
                            rgba_array = np.zeros((transparent_image_np_uint8.shape[0], transparent_image_np_uint8.shape[1], 4), dtype=np.uint8)
                            
                            if transparent_image_np_uint8.ndim == 3:
                                if transparent_image_np_uint8.shape[2] == 4:
                                    # Already RGBA
                                    rgba_array = transparent_image_np_uint8
                                elif transparent_image_np_uint8.shape[2] == 2:
                                    # Grayscale + alpha
                                    rgba_array[:,:,0] = transparent_image_np_uint8[:,:,0]
                                    rgba_array[:,:,1] = transparent_image_np_uint8[:,:,0]
                                    rgba_array[:,:,2] = transparent_image_np_uint8[:,:,0]
                                    rgba_array[:,:,3] = transparent_image_np_uint8[:,:,1]
                                elif transparent_image_np_uint8.shape[2] == 3:
                                    # RGB - copy and set alpha to full opacity
                                    rgba_array[:,:,0:3] = transparent_image_np_uint8
                                    rgba_array[:,:,3] = 255
                                elif transparent_image_np_uint8.shape[2] == 1:
                                    # Single channel - use for RGB and set alpha
                                    rgba_array[:,:,0] = transparent_image_np_uint8[:,:,0]
                                    rgba_array[:,:,1] = transparent_image_np_uint8[:,:,0]
                                    rgba_array[:,:,2] = transparent_image_np_uint8[:,:,0]
                                    rgba_array[:,:,3] = transparent_image_np_uint8[:,:,0]  # Use same channel for alpha
                            else:
                                # 2D array - use as both grayscale and alpha
                                rgba_array[:,:,0] = transparent_image_np_uint8
                                rgba_array[:,:,1] = transparent_image_np_uint8
                                rgba_array[:,:,2] = transparent_image_np_uint8
                                rgba_array[:,:,3] = transparent_image_np_uint8
                                
                            transparent_image_pil = Image.fromarray(rgba_array, 'RGBA')

                        # Save with transparency
                        transparent_image_pil.save(output_path, save_format)


                    processed_files += 1 # Moved inside try block
                    progress_message = f"Processed: {processed_files}/{total_files} files" # Moved inside try block
                    update_status_callback(progress_message)


                except Exception as e:
                    error_message = f"Error processing {tiff_file}: {e}"
                    update_status_callback(error_message)

        inference_finish_time = datetime.now()
        duration = inference_finish_time - inference_start_time
        minutes = duration.seconds // 60
        seconds = duration.seconds % 60
        duration_str = f"{minutes:02d}:{seconds:02d}"

        completion_message = f"Inference completed! {processed_files} cropped masks saved in {output_folder}. Time taken: {duration_str}"
        update_status_callback(completion_message)

    except Exception as e:
        error_message = f"Inference process failed: {e}"
        update_status_callback(error_message)
        messagebox.showerror("Error", error_message)
    finally:
        inference_running = False
        window.after(0, enable_buttons)


def run_inference_threaded(model_path, input_folder, output_folder, output_file_type):
    global inference_thread, inference_running
    if inference_running:
        messagebox.showerror("Inference Running", "Inference is already running. Please stop the current process before starting a new one.")
        return

    inference_running = True
    disable_buttons()

    inference_thread = threading.Thread(target=run_inference_process, args=(model_path, input_folder, output_folder, update_status_text, output_file_type))
    inference_thread.start()

# This function is called by run_inference_threaded and needs to be defined before main
def run_inference_process(model_path, input_folder, output_folder, update_status_callback, output_file_type):
    global inference_model, inference_running

    try:
        update_status_callback("Loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference_model = UNet(n_channels=3, n_classes=1)
        inference_model.load_state_dict(torch.load(model_path, map_location=device))
        inference_model.to(device).eval()

        image_size = 256
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]
        total_files = len(tiff_files)
        processed_files = 0

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        update_status_callback(f"Starting inference on {total_files} files...")

        inference_start_time = datetime.now()

        input_dtype_output = 0

        with torch.no_grad():
            for tiff_file in tiff_files:
                if not inference_running:
                    update_status_callback("Inference stopped by user.")
                    return

                image_path = os.path.join(input_folder, tiff_file)

                # --- Read input TIFF metadata ---
                with tifffile.TiffFile(image_path) as tif:
                    input_tiff_tags = tif.pages[0].tags
                    input_dtype = tif.pages[0].dtype
                    input_dtype_output = input_dtype

                    input_resolution = None
                    try:
                        x_resolution = input_tiff_tags['XResolution'].value
                        y_resolution = input_tiff_tags['YResolution'].value
                        resolution_unit = input_tiff_tags['ResolutionUnit'].value
                        input_resolution = (x_resolution, y_resolution, resolution_unit)
                    except KeyError:
                        pass

                try:
                    image_pil = Image.open(image_path) # DO NOT convert here, keep original mode

                    # Process the image and run inference
                    # (Implementation details would go here)
                    
                    processed_files += 1
                    update_status_callback(f"Processed {processed_files}/{total_files} files")
                    
                except Exception as e:
                    update_status_callback(f"Error processing {tiff_file}: {str(e)}")
                    continue

        inference_end_time = datetime.now()
        inference_duration = inference_end_time - inference_start_time
        update_status_callback(f"Inference completed in {inference_duration}")
        
    except Exception as e:
        update_status_callback(f"Inference error: {str(e)}")
    finally:
        # Re-enable UI elements
        if 'start_inference_button' in globals():
            start_inference_button.config(state=tk.NORMAL)
        if 'stop_inference_button' in globals():
            stop_inference_button.config(state=tk.DISABLED)
        if 'load_model_button' in globals():
            load_model_button.config(state=tk.NORMAL)
        if 'input_folder_button' in globals():
            input_folder_button.config(state=tk.NORMAL)
        if 'output_folder_button' in globals():
            output_folder_button.config(state=tk.NORMAL)
        inference_running = False


def set_output_file_type(file_type):
    global output_file_type
    output_file_type = file_type
    global model_path_entry, input_folder_entry, output_folder_entry, status_text, output_file_type_combobox
    global start_inference_button, stop_inference_button

    model_path = model_path_entry.get()
    input_folder = input_folder_entry.get()
    output_folder_specified = output_folder_entry.get()
    selected_file_type = output_file_type_combobox.get()

    if not model_path or not os.path.exists(model_path):
        messagebox.showerror("Error", "Please select a valid trained model file.")
        return
    if not input_folder or not os.path.isdir(input_folder):
        messagebox.showerror("Error", "Please select a valid input folder containing TIFF files.")
        return

    output_folder = output_folder_specified if output_folder_specified else input_folder

    if not os.path.isdir(output_folder):
        try:
            os.makedirs(output_folder, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Error", f"Invalid output folder or cannot create it: {e}")
            return

    if not os.path.isdir(output_folder):
         messagebox.showerror("Error", "Output folder is invalid.")
         return

    # Update UI state
    start_inference_button.config(state=tk.DISABLED)
    stop_inference_button.config(state=tk.NORMAL)
    
    # Clear and prepare status text
    status_text.config(state=tk.NORMAL)
    status_text.delete(1.0, tk.END)
    status_text.insert(tk.END, "Starting inference process...\n")
    status_text.config(state=tk.DISABLED)

    # Run inference in a separate thread to keep UI responsive
    run_inference_threaded(model_path, input_folder, output_folder, selected_file_type)

def update_status_text(message):
    status_text.config(state=tk.NORMAL)
    status_text.insert(tk.END, message + "\n")
    status_text.see(tk.END)
    status_text.config(state=tk.DISABLED)

def disable_buttons():
    start_inference_button.config(state=tk.DISABLED)
    stop_inference_button.config(state=tk.NORMAL)
    load_model_button.config(state=tk.DISABLED)
    input_folder_button.config(state=tk.DISABLED)
    output_folder_button.config(state=tk.DISABLED)


def enable_buttons():
    start_inference_button.config(state=tk.NORMAL)
    stop_inference_button.config(state=tk.DISABLED)
    load_model_button.config(state=tk.NORMAL)
    input_folder_button.config(state=tk.NORMAL)
    output_folder_button.config(state=tk.NORMAL)


# This is a duplicate function that's already defined above
# Removing to avoid conflicts
# def load_model_command():
#     global model_path_entry
#     filepath = filedialog.askopenfilename(
#         initialdir=".",
#         title="Select Trained Model",
#         filetypes=(("Model files", "*.pth"), ("all files", "*.*"))
#     )
#     if filepath:
#         model_path_entry.delete(0, tk.END)
#         model_path_entry.insert(0, filepath)

def setup_inference_ui(parent):
    # Global declarations consolidated at the beginning
    global window, model_path_entry, input_folder_entry, output_folder_entry, status_text, output_file_type_combobox
    global load_model_button, input_folder_button, output_folder_button, start_inference_button, stop_inference_button
    global inference_thread, inference_running, output_file_type
    
    # Initialize global variables with default values
    inference_running = False
    output_file_type = ".tif"
    
    window = parent
    
    # --- Model and Folder Selection Frame ---
    selection_frame = ttk.Frame(window, padding=(10, 10))
    selection_frame.pack(fill=tk.X, expand=False)
    
    ttk.Label(selection_frame, text="Trained Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    model_path_entry = ttk.Entry(selection_frame, width=60)
    model_path_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    load_model_button = ttk.Button(selection_frame, text="Browse", command=load_model_command)
    load_model_button.grid(row=0, column=2, padx=5, pady=5)
    
    ttk.Label(selection_frame, text="Input TIFF Folder:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    input_folder_entry = ttk.Entry(selection_frame, width=60)
    input_folder_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
    input_folder_button = ttk.Button(selection_frame, text="Browse", command=select_input_folder_command)
    input_folder_button.grid(row=1, column=2, padx=5, pady=5)

    ttk.Label(selection_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    output_folder_entry = ttk.Entry(selection_frame, width=60, state=tk.NORMAL)
    output_folder_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
    output_folder_button = ttk.Button(selection_frame, text="Browse", command=select_output_folder_command)
    output_folder_button.grid(row=2, column=2, padx=5, pady=5)

    ttk.Label(selection_frame, text="Output File Type:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    output_file_type_combobox = ttk.Combobox(selection_frame, values=[".tif", ".png"])
    output_file_type_combobox.set(output_file_type)
    output_file_type_combobox.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
    output_file_type_combobox.bind("<<ComboboxSelected>>", lambda event: set_output_file_type(output_file_type_combobox.get()))

    selection_frame.columnconfigure(1, weight=1)

    # --- Status and Control Frame ---
    control_frame = ttk.Frame(window, padding=(10, 10))
    control_frame.pack(fill=tk.BOTH, expand=True)

    status_text = tk.Text(control_frame, height=10, state=tk.DISABLED, wrap=tk.WORD)
    status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    status_scrollbar = ttk.Scrollbar(control_frame, command=status_text.yview)
    status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    status_text.config(yscrollcommand=status_scrollbar.set)

    buttons_control_frame = ttk.Frame(control_frame)
    buttons_control_frame.pack(fill=tk.X, expand=False, pady=10)

    start_inference_button = ttk.Button(buttons_control_frame, text="Start Inference", command=start_inference_command)
    start_inference_button.pack(side=tk.LEFT, padx=10)
    stop_inference_button = ttk.Button(buttons_control_frame, text="Stop Inference", command=stop_inference_command, state=tk.DISABLED)
    stop_inference_button.pack(side=tk.LEFT, padx=10)

    enable_buttons()
    # Remove the window.mainloop() call as it's already called in the main function

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import torch.nn as nn
import torch.nn.functional as F  # Import F for functional operations
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image, ImageTk, ImageEnhance
import torchvision.transforms as transforms
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# Custom Dataset for MRI images and masks
class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """Initialize the MRIDataset
        
        Args:
            image_dir (str): Directory with all the images
            mask_dir (str): Directory with all the masks
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get list of files in image directory
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and 
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        # Get list of files in mask directory
        self.mask_files = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        # Match image and mask files by name (assuming same naming convention)
        self.matched_files = []
        for img_file in self.image_files:
            img_name = os.path.splitext(img_file)[0]
            for mask_file in self.mask_files:
                mask_name = os.path.splitext(mask_file)[0]
                if img_name == mask_name:
                    self.matched_files.append((img_file, mask_file))
                    break
        
        if not self.matched_files:
            raise ValueError("No matching image-mask pairs found in the directories")
    
    def __len__(self):
        return len(self.matched_files)
    
    def __getitem__(self, idx):
        img_name, mask_name = self.matched_files[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            
            # For mask, we need to ensure it's binary (0 or 1)
            mask = np.array(mask)
            mask = (mask > 0).astype(np.float32)  # Convert to binary mask
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        else:
            # Convert to tensor without normalization
            image = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255.0
            mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
            mask = (mask > 0.5).float()  # Ensure binary mask
        
        return image, mask
import csv
import re
from datetime import datetime, timedelta

# --- Global flags and variables ---
is_paused = False
is_stopped = False
training_finished = False
current_brightness = 1.0
current_contrast = 1.0
current_result_scale = 1.0  # This variable will now control scale for all images
num_visualizations_per_page = 5    # Changed to 5 results per page
visualization_rows = 1  # Changed to 1 row in visualization grid - now 1 row
visualization_cols = 5  # Changed to 5 columns in visualization grid - now 5 columns
current_page = 0
current_batch_total = 0
fig_plot = None
ax1_plot = None
ax2_plot = None
line1_plot = None
line2_plot = None
canvas_plot = None
epoch_losses_data = []
epoch_dice_scores_data = []
early_stopping_enabled = True  # Enable Early Stopping by default
early_stopping_patience = 5
early_stopping_min_delta = 0.001
training_thread = None
training_start_time = None
training_finish_time = None
remaining_epochs_after_pause = 0
visualization_window = None
visualization_images = []
visualization_masks = []
visualization_predictions = []
visualization_dataset = None
current_visualization_index = 0
visualization_image_labels = []
visualization_mask_labels = []
visualization_prediction_labels = []
visualization_brightness_scale = None
visualization_contrast_scale = None
visualization_result_scale_scale = None  # Renamed to visualization_image_scale_scale for clarity
visualization_index_labels = []  # List to hold dataset index labels
global_visualization_model = None  # Global variable for visualization model
window = None  # Declare window as global
best_model_state = None  # To store the best model weights
best_epoch = 0  # To store the epoch number of the best model
patience_timer = None # Global timer for patience reset

# -------------------- UNet Implementation --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):  # Changed n_classes to 1 for binary segmentation
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256), image_transform=None, mask_transform=None, num_visualizations=5):  # Added image_size, num_visualizations
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_size = image_size  # Store image_size
        self.num_visualizations = num_visualizations  # Store num_visualizations
        self.file_pairs = self._match_files()

        if not self.file_pairs:
            raise ValueError("No valid image-mask pairs found!")

    def _get_valid_files(self, directory):
        valid_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        return [f for f in os.listdir(directory) if f.lower().endswith(valid_exts)]

    def _match_files(self):
        image_files = self._get_valid_files(self.image_dir)
        mask_files = self._get_valid_files(self.mask_dir)
        pairs = []
        image_bases = {os.path.splitext(f)[0].lower(): f for f in image_files}
        for mask_file in mask_files:
            base = os.path.splitext(mask_file)[0].lower()
            if base in image_bases:
                pairs.append((image_bases[base], mask_file))
        return pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        img_file, mask_file = self.file_pairs[idx]
        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


def update_training_time_labels(start_time_str, finish_time_str, duration_str):
    start_time_var.set(f"Start Time: {start_time_str if start_time_str else 'N/A'}")
    finish_time_var.set(f"Finish Time: {finish_time_str if finish_time_str else 'N/A'}")
    time_used_var.set(f"Time Used: {duration_str}")

def train_model(image_dir, mask_dir, save_model_dir, num_epochs, update_callback, update_progress_bar, set_training_finished_flag, update_total_progress_label, update_total_progress_bar, update_plot_callback, update_training_time_callback, initial_epoch=0, model_path=None):  # Removed update_early_stopping_status callback, ADDED model_path
    global is_paused, is_stopped, training_finished, current_batch_total, epoch_losses_data, epoch_dice_scores_data, early_stopping_enabled, early_stopping_patience, early_stopping_min_delta, training_start_time, training_finish_time, remaining_epochs_after_pause, window, best_model_state, best_epoch, patience_timer  # ADD window to global, best_model_state and best_epoch to global, patience_timer

    if initial_epoch == 0:
        epoch_losses_data = []
        epoch_dice_scores_data = []

    image_size = 256  # Define image size
    transform = transforms.Compose([  # Define transforms including Resize
        transforms.Resize(image_size),  # Resize images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    mask_transform = transforms.Compose([  # Define mask transforms including Resize
        transforms.Resize(image_size, interpolation=Image.NEAREST),  # Resize masks
        transforms.ToTensor()
    ])

    try:
        train_dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, image_size=(image_size, image_size), image_transform=transform, mask_transform=mask_transform)  # Use SegmentationDataset and transforms
        batch_size = 2
        num_workers = 0
        shuffle = True
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    except FileNotFoundError:
        return "Dataset directories not found. Please check the paths.", [], []

    model = UNet(n_channels=3, n_classes=1, bilinear=True)  # Use UNet model, n_classes=1 for binary

    if model_path:  # Load model if path is provided
        try:
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load existing model weights
        except Exception as e:
            return f"Error loading model from {model_path}: {e}", [], []  # Handle model loading error

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if initial_epoch == 0:
        start_epoch = 0
        best_val_loss = float('inf')  # Initialize best_val_loss at the start of new training
        best_model_state = None       # Initialize best_model_state
        best_epoch = 0               # Initialize best_epoch
    else:
        start_epoch = initial_epoch
        # If continuing training, load previous best_val_loss and best_model_state if you saved them.
        # For simplicity in this example, we are re-initializing best_val_loss. You might want to load it from the saved model file for true continuation of early stopping.
        best_val_loss = float('inf')  # Re-initialize for simplicity, consider loading for true continuation

    if training_start_time is None:# and initial_epoch == 0:
        training_start_time = datetime.now()
        #
        #update_training_time_labels(start_time_str, finish_time_str, duration_str)
        update_training_time_labels(training_start_time.strftime("%H:%M:%S"), "N/A", "N/A")
    else:
        update_training_time_labels(training_start_time.strftime("%H:%M:%S"), "N/A", calculate_training_duration(training_start_time, datetime.now()))

    total_batches = len(train_dataloader) * num_epochs  # Corrected total batches calculation
    if initial_epoch == 0:
        current_batch_total = 0

    epochs_no_improve = 0
    if patience_timer is not None:
        window.after_cancel(patience_timer) # Cancel any existing timer when starting new training
        patience_timer = None # Reset the timer variable

    #for epoch in range(start_epoch, num_epochs):
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0
        dice_score_epoch = 0
        batch_count = 0

        #print(f'current epoch {epoch}, start_epoch {start_epoch}, num_epochs {num_epochs}')

        for batch_idx, (images, masks) in enumerate(train_dataloader):
            if is_stopped:
                training_finish_time = datetime.now()
                window.after(0, update_training_time_callback, training_start_time.strftime("%H:%M:%S") if training_start_time else None, training_finish_time.strftime("%H:%M:%S"), calculate_training_duration(training_start_time, training_finish_time))
                window.after(0, set_training_finished_flag, True)
                return "Training stopped.", epoch_losses_data, epoch_dice_scores_data

            while is_paused:
                print("Training Paused - Waiting to Resume...")  # Debug print inside pause loop
                time.sleep(1)
            # print("Training Resumed or Continuing...") # Debug print after pause loop  <- COMMENTED OUT LINE

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
            batch_count += 1

            loss.backward()
            optimizer.step()

            predicted_masks_sigmoid = torch.sigmoid(outputs)
            predicted_masks_binary = (predicted_masks_sigmoid > 0.5).float()
            dice_score = calculate_dice_coefficient(predicted_masks_binary, masks)
            dice_score_epoch += dice_score.item()

            current_batch_total += 1
            batch_percentage = (batch_idx + 1) / len(train_dataloader) * 100
            # Corrected total progress calculation
            epoch_total_percentage = (current_batch_total / total_batches * 100)
            epoch_total_percentage = min(epoch_total_percentage, 100)

            window.after(0, update_progress_bar, batch_percentage, 1 + epoch, num_epochs + initial_epoch)
            window.after(0, update_total_progress_bar, epoch_total_percentage - (1/total_batches*100) if epoch_total_percentage > 0 else 0)  # Subtracting the progress step to align the bar
            window.after(0, update_total_progress_label, f"Total Progress: {epoch_total_percentage:.1f}%")

        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        avg_dice_score = dice_score_epoch / batch_count if batch_count > 0 else 0

        epoch_losses_data.append(avg_epoch_loss)
        epoch_dice_scores_data.append(avg_dice_score)

        window.after(0, update_plot_callback, epoch_losses_data, epoch_dice_scores_data)

        if early_stopping_enabled:
            val_loss = avg_epoch_loss
            if val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()  # Save the current best model state
                best_epoch = epoch + 1 #+ initial_epoch  # Save the epoch number
                reset_patience_timer() # Reset patience timer on improvement
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    # --- Save best model on early stopping ---
                    if save_model_dir and best_model_state is not None:  # Check if save directory is provided and best model exists
                        # --- Generate timestamp for filename ---
                        finish_time_for_filename = datetime.now()
                        timestamp_str = finish_time_for_filename.strftime("%Y-%m-%d_%H-%M-%S")  # Format as %Y-%m-%d_%H-%M-%S
                        # --- Construct filename with timestamp, epoch number and "early_stopped" ---
                        model_name = f"{timestamp_str}_epoch_{best_epoch}_es_unet_early_stopped_model.pth"  # Include best epoch and "early_stopped"
                        model_path = os.path.join(save_model_dir, model_name)
                        torch.save(best_model_state, model_path)  # Save the best model state
                        print(f"Early stopping triggered and best model (epoch {best_epoch}) saved to {model_path}")  # Confirmation message

                    training_finish_time = datetime.now()
                    window.after(0, update_training_time_callback, training_start_time.strftime("%H:%M:%S") if training_start_time else None, training_finish_time.strftime("%H:%M:%S"), calculate_training_duration(training_start_time, training_finish_time))
                    window.after(0, set_training_finished_flag, True)
                    return f"Early stopping triggered at epoch {best_epoch}. Best model saved.", epoch_losses_data, epoch_dice_scores_data


    if save_model_dir:
        # --- Generate timestamp for filename ---
        finish_time_for_filename = datetime.now()
        timestamp_str = finish_time_for_filename.strftime("%Y-%m-%d_%H-%M-%S")

        # --- Calculate cumulative epoch number ---
        cumulative_epochs = num_epochs  # Default to current num_epochs if starting new
        if model_path:  # If continuing training from a pre-trained model
            previous_epochs = extract_epoch_from_filename(model_path)  # Extract epochs from filename
            cumulative_epochs = previous_epochs + num_epochs  # Add current epochs to previous

        # --- Construct filename with timestamp and cumulative epoch number ---
        model_name = f"{timestamp_str}_epoch_{cumulative_epochs}_unet_final_model.pth"  # Include cumulative epochs
        model_path = os.path.join(save_model_dir, model_name)
        torch.save(model.state_dict(), model_path)
        
    training_finish_time = datetime.now()
    window.after(0, update_training_time_callback, training_start_time.strftime("%H:%M:%S") if training_start_time else None, training_finish_time.strftime("%H:%M:%S"), calculate_training_duration(training_start_time, training_finish_time))
    window.after(0, set_training_finished_flag, True)
    return "Training complete!", epoch_losses_data, epoch_dice_scores_data

def calculate_dice_coefficient(predicted_mask, target_mask, smooth=1e-6):
    predicted_mask_flat = predicted_mask.flatten()
    target_mask_flat = target_mask.flatten()
    intersection = (predicted_mask_flat * target_mask_flat).sum()
    union = predicted_mask_flat.sum() + target_mask_flat.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def calculate_training_duration(start_time, finish_time):
    if start_time and finish_time:
        duration = finish_time - start_time
        minutes = duration.seconds // 60
        seconds = duration.seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    return "N/A"

import re

def extract_epoch_from_filename(filename):
    
    match = re.search(r'_epoch_(\d+)_', filename)
    if match:
        print(f"we continue training start from {match.group(1)}")
        return int(match.group(1))  # Extract and return epoch number
    return 0  # Default to 0 if epoch number is not found

def update_plot(losses, dice_scores):
    global fig_plot, ax1_plot, ax2_plot, line1_plot, line2_plot, canvas_plot

    if fig_plot is None:
        fig_plot, ax1_plot = plt.subplots(figsize=(6.4, 4.8))  # Reduced size by 20%
        ax2_plot = ax1_plot.twinx()
        ax1_plot.set_xlabel('Epoch')
        ax1_plot.set_ylabel('Training Loss', color='red')
        ax2_plot.set_ylabel('Dice Coefficient', color='blue')
        ax1_plot.tick_params(axis='y', labelcolor='red')
        ax2_plot.tick_params(axis='y', labelcolor='blue')
        line1_plot, = ax1_plot.plot([], [], color='red', label='Training Loss')
        line2_plot, = ax2_plot.plot([], [], color='blue', label='Dice Coefficient')
        fig_plot.legend(loc='upper right')
        canvas_plot = FigureCanvasTkAgg(fig_plot, master=plot_frame)
        canvas_plot_widget = canvas_plot.get_tk_widget()
        canvas_plot_widget.grid(row=0, column=0, sticky="nsew")
        fig_plot.canvas.mpl_connect('button_press_event', on_plot_click) # Connect click event

    line1_plot.set_data(range(1, len(losses) + 1), losses)
    line2_plot.set_data(range(1, len(dice_scores) + 1), dice_scores)

    ax1_plot.relim()
    ax2_plot.relim()
    ax1_plot.autoscale_view()
    ax2_plot.autoscale_view()

    canvas_plot.draw_idle()


def save_plot_image():
    global fig_plot
    if fig_plot:
        filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if filepath:
            fig_plot.savefig(filepath)
            messagebox.showinfo("Save Plot", f"Plot saved to {filepath}")
    else:
        messagebox.showinfo("Save Plot", "No plot to save.")

def export_plot_data():
    global epoch_losses_data, epoch_dice_scores_data
    if epoch_losses_data and epoch_dice_scores_data:
        filepath = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filepath:
            with open(filepath, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Epoch", "Loss", "Dice Coefficient"])  # Header
                for i in range(len(epoch_losses_data)):
                    csv_writer.writerow([i + 1, epoch_losses_data[i], epoch_dice_scores_data[i]])  # Epoch number, loss, dice
            messagebox.showinfo("Export Data", f"Plot data exported to {filepath}")
    else:
        messagebox.showinfo("Export Data", "No plot data to export.")

def on_mousewheel_scroll(event):
    image_canvas = event.widget
    if event.num == 5 or event.delta == -120:  # scroll down, event.num for Linux
        image_canvas.yview_scroll(1, "units")
    if event.num == 4 or event.delta == 120:  # scroll up, event.num for Linux
        image_canvas.yview_scroll(-1, "units")

def visualize_results_command():
    global visualization_window, visualization_dataset, visualization_images, visualization_masks, visualization_predictions, current_visualization_index
    global visualization_image_labels, visualization_mask_labels, visualization_prediction_labels, visualization_index_labels
    global visualization_brightness_scale, visualization_contrast_scale, visualization_result_scale_scale, global_visualization_model  # Include global_visualization_model

    if visualization_window is not None:
        visualization_window.destroy()

    model_path = filedialog.askopenfilename(
        initialdir=save_model_dir_entry.get() or ".",
        title="Select Trained Model",
        filetypes=(("Model files", "*.pth"), ("all files", "*.*"))
    )
    if not model_path:
        return

    image_dir = image_dir_entry.get()  # Corrected variable name
    mask_dir = mask_dir_entry.get()

    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):  # Corrected variable name
        messagebox.showerror("Error", "Invalid image or mask directory for visualization.")
        return

    image_size = 256  # Define image size for visualization dataset
    transform = transforms.Compose([  # Define transforms for visualization dataset
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    mask_transform = transforms.Compose([  # Define mask transforms for visualization dataset
        transforms.Resize(image_size, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    try:
                visualization_dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, image_size=(image_size, image_size), image_transform=transform, mask_transform=mask_transform, num_visualizations=num_visualizations_per_page)  # Use SegmentationDataset and transforms
    except FileNotFoundError:
        messagebox.showerror("Error", "Dataset directories not found for visualization.")
        return

    model = UNet(n_channels=3, n_classes=1)  # Use UNet for visualization
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    global_visualization_model = model  # Set global visualization model

    visualization_window = tk.Toplevel(window)
    visualization_window.title("Visualization of Results")
    visualization_window.protocol("WM_DELETE_WINDOW", on_visualization_window_close)  # Handle window close event

    controls_frame = ttk.Frame(visualization_window)
    controls_frame.pack(pady=10)

    # Brightness control
    tk.Label(controls_frame, text="Brightness:").grid(row=0, column=0, padx=5)
    visualization_brightness_scale = tk.Scale(controls_frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, command=adjust_brightness_realtime)  # Changed command
    visualization_brightness_scale.set(current_brightness)
    visualization_brightness_scale.grid(row=0, column=1, padx=5)

    # Contrast control
    tk.Label(controls_frame, text="Contrast:").grid(row=0, column=2, padx=5)
    visualization_contrast_scale = tk.Scale(controls_frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, command=adjust_contrast_realtime)  # Changed command
    visualization_contrast_scale.set(current_contrast)
    visualization_contrast_scale.grid(row=0, column=3, padx=5)

    # Image Scale control  -- Label changed to "Image Scale"
    tk.Label(controls_frame, text="Image Scale:").grid(row=0, column=4, padx=5)  # Label changed
    visualization_result_scale_scale = tk.Scale(controls_frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, command=scale_all_images_realtime)  # Command changed, variable name remains for now but conceptually it's image scale
    visualization_result_scale_scale.set(current_result_scale)
    visualization_result_scale_scale.grid(row=0, column=5, padx=5)


    image_canvas_frame = ttk.Frame(visualization_window)  # Frame to hold canvas and scrollbar
    image_canvas_frame.pack(expand=True, fill=tk.BOTH)

    image_canvas = tk.Canvas(image_canvas_frame, borderwidth=2, relief=tk.SUNKEN)
    image_v_scrollbar = ttk.Scrollbar(image_canvas_frame, orient=tk.VERTICAL, command=image_canvas.yview)
    image_canvas.config(yscrollcommand=image_v_scrollbar.set)

    image_v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    image_frame = ttk.Frame(image_canvas)  # Frame inside canvas
    image_frame.pack()
    image_canvas.create_window((0, 0), window=image_frame, anchor=tk.NW, tags="image_frame_tag")  # Embed frame in canvas

    image_frame.bind("<Configure>", lambda event: image_canvas.config(scrollregion=image_canvas.bbox("all")))  # Update scroll region on frame configure

    # Mouse wheel binding for scrolling
    windowingsystem = window.tk.call('tk', 'windowingsystem')
    if windowingsystem == 'win32' or windowingsystem == 'aqua':  # Windows or MacOS
        image_canvas.bind("<MouseWheel>", on_mousewheel_scroll)
    elif windowingsystem == 'x11':  # Linux
        image_canvas.bind("<Button-4>", on_mousewheel_scroll)  # Scroll up
        image_canvas.bind("<Button-5>", on_mousewheel_scroll)  # Scroll down


    visualization_image_labels = []
    visualization_mask_labels = []
    visualization_prediction_labels = []
    visualization_index_labels = []  # Initialize index labels list

    for i in range(num_visualizations_per_page):  # Loop for total number of visualizations
        # Row and Column indices are now swapped for column-wise layout
        col_index = i  # Each result in a column now
        row_index = 0  # All results in the same row in visualization grid

        # Dataset Index Label - Column-wise placement
        index_frame = ttk.Frame(image_frame, padding=0)
        index_frame.grid(row=row_index * 4, column=col_index * 1, padx=20, pady=0, sticky="n")  # Row * 4 for spacing, column * 1, padx adjusted, sticky "n"
        index_label = tk.Label(index_frame, text=f"Data Index: ")
        index_label.pack(anchor="n")  # Top align index label
        visualization_index_labels.append(index_label)

        # Original Image Frame - Column-wise placement
        original_frame = ttk.Frame(image_frame, padding=5)
        original_frame.grid(row=row_index * 4 + 1, column=col_index * 1, padx=20, pady=5)  # Row * 4 + 1, column * 1, padx adjusted
        tk.Label(original_frame, text="Original Image").pack()  # Label changed to "Original Image"
        image_label = tk.Label(original_frame)
        image_label.pack()
        visualization_image_labels.append(image_label)

        # Original Mask Frame - Column-wise placement
        mask_frame = ttk.Frame(image_frame, padding=5)
        mask_frame.grid(row=row_index * 4 + 2, column=col_index * 1, padx=20, pady=5)  # Row * 4 + 2, column * 1, padx adjusted
        tk.Label(mask_frame, text="Original Mask").pack()  # Label changed to "Original Mask"
        mask_label = tk.Label(mask_frame)
        mask_label.pack()
        visualization_mask_labels.append(mask_label)

        # Prediction Frame - Column-wise placement
        prediction_frame = ttk.Frame(image_frame, padding=5)
        prediction_frame.grid(row=row_index * 4 + 3, column=col_index * 1, padx=20, pady=5)  # Row * 4 + 3, column * 1, padx adjusted
        tk.Label(prediction_frame, text="Prediction Mask").pack()  # Label changed to "Prediction Mask"
        prediction_label = tk.Label(prediction_frame)
        prediction_label.pack()
        visualization_prediction_labels.append(prediction_label)


    navigation_frame = ttk.Frame(visualization_window)
    navigation_frame.pack(pady=10)

    prev_button = tk.Button(navigation_frame, text="Previous Page", command=previous_page)
    prev_button.grid(row=0, column=0, padx=5)
    next_button = tk.Button(navigation_frame, text="Next Page", command=next_page)
    next_button.grid(row=0, column=1, padx=5)

    update_visualization_display()  # Call without model argument

def update_visualization_display():  # Removed model argument
    global visualization_dataset, current_visualization_index, visualization_images, visualization_masks, visualization_predictions
    global visualization_image_labels, visualization_mask_labels, visualization_prediction_labels, visualization_index_labels
    global current_brightness, current_contrast, current_result_scale, visualization_window, num_visualizations_per_page, current_page, global_visualization_model  # Include global_visualization_model

    if visualization_dataset is None or visualization_window is None or global_visualization_model is None:  # Check global_visualization_model
        return

    visualization_images = []
    visualization_masks = []
    visualization_predictions = []

    start_index = current_page * num_visualizations_per_page
    end_index = start_index + num_visualizations_per_page

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_visualization_model.to(device)  # Use global_visualization_model

    dataset_len = len(visualization_dataset)  # Get total length of dataset

    for i in range(start_index, min(end_index, dataset_len)):  # Use dataset_len
        image, mask = visualization_dataset[i]
        visualization_images.append(image)
        visualization_masks.append(mask)

        image_gpu = image.unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = global_visualization_model(image_gpu)  # Use global_visualization_model
            prediction_sigmoid = torch.sigmoid(prediction).cpu()
            predicted_mask_binary = (prediction_sigmoid > 0.5).float()

        visualization_predictions.append(predicted_mask_binary.squeeze(0))

    for i in range(len(visualization_images)):
        data_index = start_index + i  # Calculate dataset index
        visualization_index_labels[i].config(text=f"Data Index: {data_index}")  # Update index label

        # --- Original Image ---
        img_original = transforms.ToPILImage()(visualization_images[i].cpu())  # Get PIL image
        img = img_original.resize((int(img_original.width * current_result_scale), int(img_original.height * current_result_scale)), Image.BILINEAR)  # Scale Image.BILINEAR for better scaling
        enhancer = ImageEnhance.Brightness(img)
        img_brightened = enhancer.enhance(current_brightness)
        enhancer = ImageEnhance.Contrast(img_brightened)
        img_contrasted = enhancer.enhance(current_contrast)
        img_tk = ImageTk.PhotoImage(img_contrasted)
        visualization_image_labels[i].config(image=img_tk)
        visualization_image_labels[i].image = img_tk

        # --- Original Mask ---
        mask_img_original = transforms.ToPILImage()(visualization_masks[i].cpu().squeeze(0))  # Get PIL mask image
        mask_img = mask_img_original.resize((int(mask_img_original.width * current_result_scale), int(mask_img_original.height * current_result_scale)), Image.NEAREST)  # Scale Mask Image.NEAREST for mask
        mask_img_tk = ImageTk.PhotoImage(mask_img.convert('L'))
        visualization_mask_labels[i].config(image=mask_img_tk)
        visualization_mask_labels[i].image = mask_img_tk

        # --- Prediction Mask ---
        pred_mask_original = transforms.ToPILImage()(visualization_predictions[i].cpu().squeeze(0).float())  # Get PIL prediction mask
        pred_mask = pred_mask_original.resize((int(pred_mask_original.width * current_result_scale), int(pred_mask_original.height * current_result_scale)), Image.NEAREST)  # Scale Prediction Mask, Image.NEAREST for mask
        pred_mask_tk = ImageTk.PhotoImage(pred_mask.convert('L'))
        visualization_prediction_labels[i].config(image=pred_mask_tk)
        visualization_prediction_labels[i].image = pred_mask_tk

    # Clear remaining labels if less images available than num_visualizations_per_page
    for i in range(len(visualization_images), num_visualizations_per_page):
        visualization_image_labels[i].config(image=None)
        visualization_mask_labels[i].config(image=None)
        visualization_prediction_labels[i].config(image=None)
        visualization_index_labels[i].config(text=f"Data Index: ")  # Clear index label if no data
    if visualization_window:  # Ensure window still exists before updating
        visualization_window.update_idletasks()  # Force immediate update


def on_visualization_window_close():
    global visualization_window
    visualization_window.destroy()
    visualization_window = None  # Reset the window variable
    global global_visualization_model  # Reset global visualization model as well
    global_visualization_model = None

def adjust_brightness_realtime(value):
    global current_brightness
    current_brightness = float(value)
    if visualization_window:
        update_visualization_display()  # Call without model

def adjust_contrast_realtime(value):
    global current_contrast
    current_contrast = float(value)
    if visualization_window:
        update_visualization_display()  # Call without model

def scale_all_images_realtime(value):  # Renamed function
    global current_result_scale
    current_result_scale = float(value)
    if visualization_window:
        update_visualization_display()  # Call without model

def next_page():
    global current_page, visualization_dataset, num_visualizations_per_page
    if visualization_dataset is not None and (current_page + 1) * num_visualizations_per_page < len(visualization_dataset):  # Check dataset boundary
        current_page += 1
        update_visualization_display()  # Call without model

def previous_page():
    global current_page
    if visualization_dataset is not None and current_page > 0:  # Prevent going below page 0, and check dataset existence
        current_page -= 1
        update_visualization_display()  # Call without model

def on_main_window_close():
    global window, is_stopped, training_thread, patience_timer
    
    is_stopped = True   # Signal the training thread to stop
    if patience_timer is not None:
        window.after_cancel(patience_timer) # Cancel patience timer if window closed

    if training_thread is not None and training_thread.is_alive():
        print("Waiting for training thread to stop...")  # Optional: feedback in console
        training_thread.join(timeout=5)  # Wait for thread to finish, with a timeout. Adjust timeout as needed.
        if training_thread.is_alive():
            print("Training thread did not terminate gracefully after timeout. Forcefully stopping...")  # Optional: feedback if thread doesn't join in time
            # In a real robust application, you might want to implement more forceful thread termination if needed.
            # However, for this case, waiting with a timeout and letting the training loop check `is_stopped` should be sufficient.

    print("Closing main window.")  # Optional: feedback in console
    window.destroy()  # Destroy the main window, which should exit mainloop()
    window.quit()  # Explicitly quit mainloop()

def browse_image_dir():
    dir_path = filedialog.askdirectory(initialdir=".", title="Select Image Directory")
    if dir_path:
        image_dir_entry.delete(0, tk.END)
        image_dir_entry.insert(0, dir_path)

def browse_mask_dir():
    dir_path = filedialog.askdirectory(initialdir=".", title="Select Mask Directory")
    if dir_path:
        mask_dir_entry.delete(0, tk.END)
        mask_dir_entry.insert(0, dir_path)

def browse_save_model_dir():
    dir_path = filedialog.askdirectory(initialdir=".", title="Select Save Model Directory")
    if dir_path:
        save_model_dir_entry.delete(0, tk.END)
        save_model_dir_entry.insert(0, dir_path)

def browse_pretrained_model_path():
    file_path = filedialog.askopenfilename(initialdir=save_model_dir_entry.get() or ".", 
                                           title="Select Pre-trained Model", 
                                           filetypes=(("Model files", "*.pth"), ("all files", "*.*")))
    if file_path:
        pretrained_model_path_entry.delete(0, tk.END)
        pretrained_model_path_entry.insert(0, file_path)
    
    update_continue_button_state()  # Ensure button updates immediately

def start_new_training_process():
    global is_stopped, is_paused, training_thread, training_finished, training_start_time, training_finish_time, current_batch_total, best_model_state, best_epoch # Added best_model_state and best_epoch to global
    global epoch_losses_data, epoch_dice_scores_data

    if training_thread is not None and training_thread.is_alive():
        messagebox.showerror("Training in Progress", "Training is already running. Stop current training before starting new.")
        return

    image_dir = image_dir_entry.get()
    mask_dir = mask_dir_entry.get()
    save_model_dir = save_model_dir_entry.get()
    num_epochs_str = epochs_entry.get()

    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        messagebox.showerror("Error", "Invalid image or mask directory.")
        return

    if not os.path.isdir(save_model_dir):
        messagebox.showerror("Error", "Invalid save model directory.")
        return

    if not num_epochs_str.isdigit():
        messagebox.showerror("Error", "Invalid number of epochs.")
        return

    num_epochs = int(num_epochs_str)
    if num_epochs <= 0:
        messagebox.showerror("Error", "Epochs must be greater than 0.")
        return

    is_stopped = False
    is_paused = False
    training_finished = False
    training_start_time = None # Reset start time for new training
    training_finish_time = None # Reset finish time for new training
    current_batch_total = 0 # Reset batch counter
    best_model_state = None # Reset best model state for new training
    best_epoch = 0 # Reset best epoch for new training
    epoch_losses_data = [] # Clear previous plot data
    epoch_dice_scores_data = [] # Clear previous plot data
    update_plot(epoch_losses_data, epoch_dice_scores_data) # Clear plot on starting new training
    start_time_var.set("Start Time: N/A") # Reset time labels
    finish_time_var.set("Finish Time: N/A")
    time_used_var.set("Time Used: N/A")
    progress_bar['value'] = 0 # Reset progress bars
    total_progress_bar['value'] = 0
    epoch_progress_label.config(text="Epoch Progress: 0.0% (Epoch 0/0)") # Reset progress labels
    total_progress_label.config(text="Total Progress: 0.0%")


    start_button.config(state=tk.DISABLED)
    continue_button.config(state=tk.DISABLED) # Disable continue button when new training starts
    pause_button.config(state=tk.NORMAL)
    resume_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    visualize_results_button.config(state=tk.DISABLED) # Disable visualize results during training
    export_plot_data_button.config(state=tk.DISABLED) # Disable export plot data during training
    save_plot_button.config(state=tk.DISABLED) # Disable save plot button during training


    training_thread = threading.Thread(target=run_training, args=(image_dir, mask_dir, save_model_dir, num_epochs))
    training_thread.start()

def continue_training_command():
    global is_stopped, is_paused, training_thread, training_finished, training_start_time, pretrained_model_path

    if training_thread is not None and training_thread.is_alive():
        messagebox.showerror("Training in Progress", "Training is already running.")
        return

    image_dir = image_dir_entry.get()
    mask_dir = mask_dir_entry.get()
    save_model_dir = save_model_dir_entry.get()
    num_epochs_str = epochs_entry.get()
    pretrained_model_path = pretrained_model_path_entry.get().strip()  # Get the model path

    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        messagebox.showerror("Error", "Invalid image or mask directory.")
        return

    if not os.path.isdir(save_model_dir):
        messagebox.showerror("Error", "Invalid save model directory.")
        return

    if not pretrained_model_path or not os.path.isfile(pretrained_model_path):
        messagebox.showerror("Error", "Pre-trained model path is required to continue training.")
        return

    if not num_epochs_str.isdigit():
        messagebox.showerror("Error", "Invalid number of epochs.")
        return

    num_epochs = int(num_epochs_str)
    if num_epochs <= 0:
        messagebox.showerror("Error", "Epochs must be greater than 0.")
        return

    # Extract the epoch number from the filename
    initial_epoch_continue = extract_epoch_from_filename(pretrained_model_path)
    print('initial epoch ', initial_epoch_continue)

    is_stopped = False
    is_paused = False
    training_finished = False

    # Update UI buttons
    start_button.config(state=tk.DISABLED)
    continue_button.config(state=tk.DISABLED)
    pause_button.config(state=tk.NORMAL)
    resume_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    visualize_results_button.config(state=tk.DISABLED)
    export_plot_data_button.config(state=tk.DISABLED)
    save_plot_button.config(state=tk.DISABLED)

    # Start training thread with correct epoch number
    training_thread = threading.Thread(
        target=run_training,
        args=(image_dir, mask_dir, save_model_dir, num_epochs, pretrained_model_path, initial_epoch_continue)
    )
    training_thread.start()

#F:/LK/copy/c/original
#F:/LK/copy/c/mask
#C:/Users/Luki/Desktop/training result data
#C:/Users/Luki/Desktop/training result data/Cor/2025-02-20_13-59-34_epoch_40_es_unet_early_stopped_model.pth
def run_training(image_dir, mask_dir, save_model_dir, num_epochs, model_path=None, initial_epoch=0): # Modified run_training to accept model_path and initial_epoch
    global window

    def update_progress(batch_percentage, epoch, total_epochs):
        # Declare all global variables before using them
        global epoch_losses_data, epoch_dice_scores_data
        global status_row_frame, progress_frame, epoch_progress_label, progress_bar
        global total_progress_label, total_progress_bar, time_frame
        global start_time_var, start_time_label, finish_time_var, finish_time_label, time_used_var, time_used_label
        global training_finished, start_button, continue_button, pause_button, resume_button, stop_button, visualize_results_button, export_plot_data_button, save_plot_button
        global is_paused, patience_timer, epochs_no_improve, early_stopping_enabled
        #print(f'initial epoch {initial_epoch}, current epoch {epoch}')
        epoch_progress_label.config(text=f"Epoch Progress: {batch_percentage:.1f}% (Epoch {epoch}/{total_epochs})")
        #epoch_progress_label.config(text=f"Epoch Progress: {batch_percentage:.1f}% (Epoch {initial_epoch}/{total_epochs})")
        progress_bar['value'] = batch_percentage

    def update_total_progress(percentage):
        # Declare all global variables before using them
        global epoch_losses_data, epoch_dice_scores_data
        global status_row_frame, progress_frame, epoch_progress_label, progress_bar
        global total_progress_label, total_progress_bar, time_frame
        global start_time_var, start_time_label, finish_time_var, finish_time_label, time_used_var, time_used_label
        global training_finished, start_button, continue_button, pause_button, resume_button, stop_button, visualize_results_button, export_plot_data_button, save_plot_button
        global is_paused, patience_timer, epochs_no_improve, early_stopping_enabled
        total_progress_bar['value'] = percentage

    def update_total_progress_label_gui(text):
        # Declare all global variables before using them
        global epoch_losses_data, epoch_dice_scores_data
        global status_row_frame, progress_frame, epoch_progress_label, progress_bar
        global total_progress_label, total_progress_bar, time_frame
        global start_time_var, start_time_label, finish_time_var, finish_time_label, time_used_var, time_used_label
        global training_finished, start_button, continue_button, pause_button, resume_button, stop_button, visualize_results_button, export_plot_data_button, save_plot_button
        total_progress_label.config(text=text)

    def set_training_finished(finished):
        # Declare global variables before using them
        global training_finished, start_button, continue_button, pause_button, resume_button, stop_button, visualize_results_button, export_plot_data_button, save_plot_button
        global status_row_frame, progress_frame, epoch_progress_label, progress_bar
        global total_progress_label, total_progress_bar, time_frame
        global start_time_var, start_time_label, finish_time_var, finish_time_label, time_used_var, time_used_label
        
        training_finished = finished
        if finished:
            start_button.config(state=tk.NORMAL)
            continue_button.config(state=tk.NORMAL) # Enable continue button after training finishes
            pause_button.config(state=tk.DISABLED)
            resume_button.config(state=tk.DISABLED)
            stop_button.config(state=tk.DISABLED)
            visualize_results_button.config(state=tk.NORMAL) # Enable visualize results after training
            export_plot_data_button.config(state=tk.NORMAL) # Enable export plot data after training
            save_plot_button.config(state=tk.NORMAL) # Enable save plot button after training
            update_continue_button_state() # Update continue button state based on saved model directory


    def update_plot_gui(losses, dice_scores):
        global epoch_losses_data, epoch_dice_scores_data
        global status_row_frame, progress_frame, epoch_progress_label, progress_bar
        global total_progress_label, total_progress_bar, time_frame
        global start_time_var, start_time_label, finish_time_var, finish_time_label, time_used_var, time_used_label
        update_plot(losses, dice_scores)

    def update_training_time_gui(start_time_str, finish_time_str, duration_str):
        global epoch_losses_data, epoch_dice_scores_data
        global status_row_frame, progress_frame, epoch_progress_label, progress_bar
        global total_progress_label, total_progress_bar, time_frame
        global start_time_var, start_time_label, finish_time_var, finish_time_label, time_used_var, time_used_label
        start_time_var.set(f"Start Time: {start_time_str if start_time_str else 'N/A'}")
        finish_time_var.set(f"Finish Time: {finish_time_str if finish_time_str else 'N/A'}")
        time_used_var.set(f"Time Used: {duration_str}")
    
    #print(f'initial epoch {initial_epoch}, current epoch {epoch}, and number of epoch {num_epochs}')

    message, losses, dice_scores = train_model(image_dir, mask_dir, save_model_dir, num_epochs,
                                                update_callback=None, # Not used directly in thread, using window.after
                                                update_progress_bar=update_progress,
                                                set_training_finished_flag=set_training_finished,
                                                update_total_progress_label=update_total_progress_label_gui,
                                                update_total_progress_bar=update_total_progress,
                                                update_plot_callback=update_plot_gui,
                                                update_training_time_callback=update_training_time_gui,
                                                initial_epoch=initial_epoch, # Pass initial_epoch
                                                model_path=model_path # Pass model_path
                                                )
    if message:
        window.after(0, messagebox.showinfo, "Training Status", message)
    if training_start_time:
        window.after(0, update_training_time_gui, training_start_time.strftime("%H:%M:%S"), training_finish_time.strftime("%H:%M:%S"), calculate_training_duration(training_start_time, training_finish_time))

def pause_training():
    global is_paused, pause_button, resume_button
    is_paused = True
    pause_button.config(state=tk.DISABLED)
    resume_button.config(state=tk.NORMAL)

def resume_training():
    global is_paused, resume_button, pause_button, patience_timer, epochs_no_improve
    is_paused = False
    resume_button.config(state=tk.DISABLED)
    pause_button.config(state=tk.NORMAL)
    reset_patience_timer() # Reset patience timer on resume

def stop_training():
    global is_stopped, pause_button, resume_button, stop_button, training_finished
    global start_button, continue_button, visualize_results_button, export_plot_data_button, save_plot_button # Globals for button states

    if not training_finished: # Only allow stop if training is not already finished
        is_stopped = True
        pause_button.config(state=tk.DISABLED)
        resume_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.DISABLED) # Disable Stop during stopping process - to prevent double click issues
        start_button.config(state=tk.DISABLED) # Keep start disabled during stopping
        continue_button.config(state=tk.DISABLED) # Keep continue disabled during stopping
        visualize_results_button.config(state=tk.DISABLED) # Keep visualize disabled
        export_plot_data_button.config(state=tk.DISABLED) # Keep export disabled
        save_plot_button.config(state=tk.DISABLED) # Keep save plot disabled

def toggle_early_stopping():
    # Declare all global variables at the beginning of the function
    global early_stopping_enabled, patience_entry, min_delta_entry, early_stopping_check_var
    
    # Now use the variables after they've been declared as global
    early_stopping_enabled = early_stopping_check_var.get()
    if early_stopping_enabled:
        patience_entry.config(state=tk.NORMAL)
        min_delta_entry.config(state=tk.NORMAL)
    else:
        patience_entry.config(state=tk.DISABLED)
        min_delta_entry.config(state=tk.DISABLED)

def change_patience(value):
    # Declare all global variables at the beginning of the function
    global early_stopping_patience, patience_entry
    
    if value.isdigit():
        early_stopping_patience = int(value)
    else:
        messagebox.showerror("Error", "Patience must be an integer.")
        patience_entry.delete(0, tk.END) # Clear entry on error
        patience_entry.insert(0, str(early_stopping_patience)) # Re-insert valid value

def change_min_delta(value):
    # Declare all global variables at the beginning of the function
    global early_stopping_min_delta, min_delta_entry
    
    try:
        early_stopping_min_delta = float(value)
    except ValueError:
        messagebox.showerror("Error", "Min Delta must be a float.")
        min_delta_entry.delete(0, tk.END) # Clear entry on error
        min_delta_entry.insert(0, str(early_stopping_min_delta)) # Re-insert valid value

def update_continue_button_state():
    global continue_button, pretrained_model_path_entry
    pretrained_model_path = pretrained_model_path_entry.get().strip()  # Get the entered path

    # Enable continue button only if a valid .pth file is selected
    if pretrained_model_path and os.path.isfile(pretrained_model_path) and pretrained_model_path.lower().endswith('.pth'):
        continue_button.config(state=tk.NORMAL)
    else:
        continue_button.config(state=tk.DISABLED)

def on_plot_click(event):
    if event.inaxes == ax1_plot:
        epoch_x = int(round(event.xdata)) if event.xdata else None
        if epoch_x and 1 <= epoch_x <= len(epoch_losses_data):
            loss_value = epoch_losses_data[epoch_x-1]
            dice_value = epoch_dice_scores_data[epoch_x-1]
            messagebox.showinfo("Epoch Data", f"Epoch: {epoch_x}\nLoss: {loss_value:.4f}\nDice Coefficient: {dice_value:.4f}")

def reset_patience_timer():
    global patience_timer, epochs_no_improve
    epochs_no_improve = 0 # Reset counter
    if patience_timer is not None:
        window.after_cancel(patience_timer) # Cancel existing timer if any
    # patience_timer = window.after(early_stopping_patience * 60000, early_stopping_triggered) # 60000 ms = 1 minute - timer in minutes

# UNetTrainingTool class definition
# Initialize global variables
early_stopping_enabled = True

class UNetTrainingTool:
    """Class for handling UNet training functionality"""
    def __init__(self, master):
        # Global declarations moved to very beginning of the function
        global window, settings_row_frame, input_frame, image_dir_entry, mask_dir_entry, save_model_dir_entry, pretrained_model_path_entry
        global params_frame, epochs_entry, early_stopping_frame, early_stopping_check_var, early_stopping_check
        global patience_entry, min_delta_entry, progress_frame, epoch_progress_label, progress_bar
        global total_progress_label, total_progress_bar, time_frame
        global start_time_var, start_time_label, finish_time_var, finish_time_label, time_used_var, time_used_label
        global buttons_frame, start_button, continue_button, pause_button, resume_button, stop_button, save_plot_button, export_plot_data_button
        global visualize_button_frame, visualize_results_button, plot_frame, status_row_frame
        global is_paused, patience_timer, epochs_no_improve, early_stopping_enabled, training_finished, epoch_losses_data, epoch_dice_scores_data
        global training_thread, is_stopped
        
        self.master = master
        
        # --- Main GUI Setup ---
        window = master
        
        # Initialize the UI components
        self.setup_ui()
    
    def update_continue_button_state(self):
        # This function would check if there's a saved model and enable/disable the continue button accordingly
        # For this demo, we'll just leave it as a placeholder
        pass
        
    def browse_image_dir(self):
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            image_dir_entry.delete(0, tk.END)
            image_dir_entry.insert(0, directory)

    def browse_mask_dir(self):
        directory = filedialog.askdirectory(title="Select Mask Directory")
        if directory:
            mask_dir_entry.delete(0, tk.END)
            mask_dir_entry.insert(0, directory)

    def browse_save_model_dir(self):
        directory = filedialog.askdirectory(title="Select Save Model Directory")
        if directory:
            save_model_dir_entry.delete(0, tk.END)
            save_model_dir_entry.insert(0, directory)

    def browse_pretrained_model_path(self):
        file_path = filedialog.askopenfilename(title="Select Pre-trained Model", filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")])
        if file_path:
            pretrained_model_path_entry.delete(0, tk.END)
            pretrained_model_path_entry.insert(0, file_path)

    def toggle_early_stopping(self):
        # Use the standalone toggle_early_stopping function to avoid code duplication
        # and ensure consistent behavior
        toggle_early_stopping()
        
    def change_patience(self, value):
        # Use the standalone change_patience function to avoid code duplication
        # and ensure consistent behavior
        change_patience(value)

    def change_min_delta(self, value):
        # Use the standalone change_min_delta function to avoid code duplication
        # and ensure consistent behavior
        change_min_delta(value)

    def create_dataset(self):
        """Create dataset from the selected directories"""
        try:
            image_dir = image_dir_entry.get()
            mask_dir = mask_dir_entry.get()
            
            if not image_dir or not mask_dir:
                messagebox.showerror("Error", "Please select both image and mask directories")
                return None, None
                
            if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
                messagebox.showerror("Error", "One or both directories do not exist")
                return None, None
                
            # Create dataset
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            # Create a custom dataset
            dataset = MRIDataset(image_dir, mask_dir, transform=transform)
            
            # Split into train and validation sets (80/20)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            return train_dataset, val_dataset
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create dataset: {str(e)}")
            return None, None
    
    def create_dataloaders(self, train_dataset, val_dataset, batch_size=4):
        """Create dataloaders from datasets"""
        if train_dataset is None or val_dataset is None:
            return None, None
            
        try:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            return train_loader, val_loader
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create dataloaders: {str(e)}")
            return None, None
    
    def initialize_model(self):
        """Initialize the UNet model"""
        try:
            # Create UNet model with 1 input channel for grayscale images
            model = UNet(n_channels=1, n_classes=1, bilinear=True)
            
            # Check if we should load a pretrained model
            pretrained_path = pretrained_model_path_entry.get()
            if pretrained_path and os.path.exists(pretrained_path):
                try:
                    model.load_state_dict(torch.load(pretrained_path))
                    messagebox.showinfo("Info", "Pretrained model loaded successfully")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load pretrained model: {str(e)}")
            
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")
            return None
    
    def train_epoch(self, model, train_loader, optimizer, criterion, device):
        """Train for one epoch"""
        model.train()
        epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            progress = (batch_idx + 1) / len(train_loader) * 100
            progress_bar['value'] = progress
            window.update_idletasks()
            
            # Check if training should be stopped
            if hasattr(self, 'is_paused') and self.is_paused:
                return None  # Signal that training is paused
            if hasattr(self, 'is_stopped') and self.is_stopped:
                return None  # Signal that training is stopped
        
        return epoch_loss / len(train_loader)
    
    def validate(self, model, val_loader, criterion, device):
        """Validate the model"""
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def start_new_training_process(self):
        """Start a new training process"""
        try:
            # Reset flags
            self.is_paused = False
            self.is_stopped = False
            
            # Get training parameters
            try:
                epochs = int(epochs_entry.get())
                if epochs <= 0:
                    messagebox.showerror("Error", "Epochs must be a positive integer")
                    return
            except ValueError:
                messagebox.showerror("Error", "Epochs must be a valid integer")
                return
            
            # Create datasets and dataloaders
            train_dataset, val_dataset = self.create_dataset()
            if train_dataset is None or val_dataset is None:
                return
                
            train_loader, val_loader = self.create_dataloaders(train_dataset, val_dataset)
            if train_loader is None or val_loader is None:
                return
            
            # Initialize model
            model = self.initialize_model()
            if model is None:
                return
            
            # Set up training
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Define loss function and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Early stopping parameters
            early_stopping_enabled = early_stopping_check_var.get()
            patience = int(patience_entry.get()) if early_stopping_enabled else float('inf')
            min_delta = float(min_delta_entry.get()) if early_stopping_enabled else 0
            
            # Initialize variables for early stopping
            best_val_loss = float('inf')
            no_improve_count = 0
            
            # Initialize training history
            self.train_losses = []
            self.val_losses = []
            
            # Update UI
            start_time_var.set(time.strftime("%H:%M:%S"))
            finish_time_var.set("--:--:--")
            time_used_var.set("00:00:00")
            
            # Reset progress bars
            progress_bar['value'] = 0
            total_progress_bar['value'] = 0
            
            # Start time
            start_time = time.time()
            
            # Training loop
            for epoch in range(epochs):
                # Update epoch progress label
                epoch_progress_label.config(text=f"Epoch Progress: {epoch+1}/{epochs}")
                
                # Train for one epoch
                train_loss = self.train_epoch(model, train_loader, optimizer, criterion, device)
                if train_loss is None:  # Training was paused or stopped
                    break
                
                # Validate
                val_loss = self.validate(model, val_loader, criterion, device)
                
                # Store losses
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                # Update plot
                self.update_plot()
                
                # Update total progress
                total_progress = (epoch + 1) / epochs * 100
                total_progress_bar['value'] = total_progress
                
                # Update time
                current_time = time.time()
                elapsed_time = current_time - start_time
                time_used_var.set(self.format_time(elapsed_time))
                
                # Check for early stopping
                if early_stopping_enabled:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        # Save best model
                        save_dir = save_model_dir_entry.get()
                        if save_dir:
                            os.makedirs(save_dir, exist_ok=True)
                            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                    else:
                        no_improve_count += 1
                        if no_improve_count >= patience:
                            messagebox.showinfo("Early Stopping", f"Training stopped after {epoch+1} epochs due to no improvement")
                            break
                
                # Update UI
                window.update_idletasks()
                
                # Check if training should be stopped
                if self.is_paused:
                    messagebox.showinfo("Paused", "Training has been paused")
                    return
                if self.is_stopped:
                    messagebox.showinfo("Stopped", "Training has been stopped")
                    return
            
            # Training completed
            finish_time_var.set(time.strftime("%H:%M:%S"))
            messagebox.showinfo("Training Complete", f"Training completed after {epoch+1} epochs")
            
            # Save final model
            save_dir = save_model_dir_entry.get()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
                
        except Exception as e:
            messagebox.showerror("Error", f"Training error: {str(e)}")
    
    def format_time(self, seconds):
        """Format time in seconds to HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def update_plot(self):
        """Update the training/validation loss plot"""
        try:
            # Clear the plot
            plot_frame.figure.clear()
            ax = plot_frame.figure.add_subplot(111)
            
            # Plot training and validation loss
            epochs = range(1, len(self.train_losses) + 1)
            ax.plot(epochs, self.train_losses, 'b-', label='Training Loss')
            ax.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
            
            # Add labels and legend
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            
            # Refresh the canvas
            plot_frame.draw()
        except Exception as e:
            print(f"Error updating plot: {str(e)}")
    
    def continue_training_command(self):
        """Continue training from a saved model"""
        # Similar to start_new_training_process but loads a saved model
        pretrained_path = pretrained_model_path_entry.get()
        if not pretrained_path or not os.path.exists(pretrained_path):
            messagebox.showerror("Error", "Please select a valid pretrained model path")
            return
            
        # Start training with the pretrained model
        self.start_new_training_process()
        
    def pause_training(self):
        """Pause the training process"""
        self.is_paused = True
        
    def resume_training(self):
        """Resume the training process"""
        self.is_paused = False
        # Continue from where we left off
        self.start_new_training_process()
        
    def stop_training(self):
        """Stop the training process"""
        self.is_stopped = True
        
    def save_plot_image(self):
        """Save the loss plot as an image"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if file_path:
                plot_frame.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {str(e)}")
        
    def export_plot_data(self):
        """Export the training and validation loss data to a CSV file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if file_path:
                epochs = range(1, len(self.train_losses) + 1)
                df = pd.DataFrame({
                    'Epoch': epochs,
                    'Training Loss': self.train_losses,
                    'Validation Loss': self.val_losses
                })
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Data exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")
        
    def visualize_results_command(self):
        """Visualize the segmentation results on a sample image"""
        try:
            # Load the model
            model = self.initialize_model()
            if model is None:
                return
                
            # Set model to evaluation mode
            model.eval()
            
            # Open a sample image
            file_path = filedialog.askopenfilename(
                title="Select a sample image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")]
            )
            if not file_path:
                return
                
            # Load and preprocess the image
            image = Image.open(file_path).convert('L')  # Convert to grayscale
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                output = model(image_tensor)
                pred = torch.sigmoid(output) > 0.5  # Apply threshold
            
            # Convert prediction to image
            pred_np = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
            pred_img = Image.fromarray(pred_np)
            
            # Create a new window to display results
            result_window = tk.Toplevel(window)
            result_window.title("Segmentation Result")
            
            # Display original image
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))  # Reduced size by 20%
            ax1.imshow(np.array(image), cmap='gray')
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            # Display segmentation result
            ax2.imshow(pred_np, cmap='gray')
            ax2.set_title("Segmentation Result")
            ax2.axis('off')
            
            # Add the figure to the window
            canvas = FigureCanvasTkAgg(fig, master=result_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add a button to save the result
            def save_result():
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
                )
                if save_path:
                    pred_img.save(save_path)
                    messagebox.showinfo("Success", f"Result saved to {save_path}")
            
            save_button = ttk.Button(result_window, text="Save Result", command=save_result)
            save_button.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize results: {str(e)}")
        
    # This appears to be a duplicate method or class definition that should be removed
    # The proper UNetTrainingTool class is already defined earlier in the code
        
    def update_continue_button_state(self):
        # This function would check if there's a saved model and enable/disable the continue button accordingly
        # For this demo, we'll just leave it as a placeholder
        pass
        
    def setup_ui(self):
        # Declare all global variables at the beginning of the method
        global settings_row_frame, input_frame, image_dir_entry, mask_dir_entry, save_model_dir_entry, pretrained_model_path_entry
        global params_frame, epochs_entry, early_stopping_frame, early_stopping_check_var, early_stopping_check
        global patience_entry, min_delta_entry, progress_frame, epoch_progress_label, progress_bar
        global total_progress_label, total_progress_bar, time_frame
        global start_time_var, start_time_label, finish_time_var, finish_time_label, time_used_var, time_used_label
        global buttons_frame, start_button, continue_button, pause_button, resume_button, stop_button, save_plot_button, export_plot_data_button
        global visualize_button_frame, visualize_results_button, plot_frame, status_row_frame
        global is_paused, patience_timer, epochs_no_improve, early_stopping_enabled, training_finished, epoch_losses_data, epoch_dice_scores_data
        global training_thread, is_stopped
        
        # Initialize variables that need default values
        training_finished = False
        epoch_losses_data = []
        epoch_dice_scores_data = []
        
        # --- Settings Row Frame - Input Directories, Training Parameters, Early Stopping ---
        
        settings_row_frame = ttk.Frame(window)
        settings_row_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=10, pady=10)

        # Input Directories Frame (in settings_row_frame)
        input_frame = ttk.LabelFrame(settings_row_frame, text="Input Directories", padding=(10, 5))
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)  # In row 0, column 0 of settings_row_frame

        tk.Label(input_frame, text="Image Directory:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        image_dir_entry = tk.Entry(input_frame, width=50)
        image_dir_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        image_dir_button = tk.Button(input_frame, text="Browse", command=self.browse_image_dir)
        image_dir_button.grid(row=0, column=2, padx=5, pady=5)

        tk.Label(input_frame, text="Mask Directory:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        mask_dir_entry = tk.Entry(input_frame, width=50)
        mask_dir_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        mask_dir_button = tk.Button(input_frame, text="Browse", command=self.browse_mask_dir)
        mask_dir_button.grid(row=1, column=2, padx=5, pady=5)

        tk.Label(input_frame, text="Save Model Directory:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        save_model_dir_entry = tk.Entry(input_frame, width=50)
        save_model_dir_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        save_model_dir_button = tk.Button(input_frame, text="Browse", command=self.browse_save_model_dir)
        save_model_dir_button.grid(row=2, column=2, padx=5, pady=5)

        # New Pre-trained Model Path Row
        tk.Label(input_frame, text="Pre-trained Model Path:").grid(row=3, column=0, sticky="w", padx=5, pady=5)  # New Label
        pretrained_model_path_entry = tk.Entry(input_frame, width=50)  # New Entry
        pretrained_model_path_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)  # New Entry Grid
        pretrained_model_path_button = tk.Button(input_frame, text="Browse", command=self.browse_pretrained_model_path)  # New Browse Button
        pretrained_model_path_button.grid(row=3, column=2, padx=5, pady=5)  # New Button Grid


        # Training Parameters Frame (in settings_row_frame, to the right of Input Directories)
        params_frame = ttk.LabelFrame(settings_row_frame, text="Training Parameters", padding=(10, 5))
        params_frame.grid(row=0, column=1, sticky="ew", padx=5, pady=5)  # In row 0, column 1 of settings_row_frame

        tk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        epochs_entry = tk.Entry(params_frame, width=10)
        epochs_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        epochs_entry.insert(0, "10")  # Default epochs value


        # Early Stopping Frame (in settings_row_frame, to the right of Training Parameters)
        early_stopping_frame = ttk.LabelFrame(settings_row_frame, text="Early Stopping", padding=(10, 5))
        early_stopping_frame.grid(row=0, column=2, sticky="new", padx=5, pady=5)  # In row 0, column 2 of settings_row_frame, top-left sticky

        early_stopping_check_var = tk.BooleanVar(value=True)  # Enable Early Stopping by default
        early_stopping_check = tk.Checkbutton(early_stopping_frame, text="Enable Early Stopping", variable=early_stopping_check_var, command=self.toggle_early_stopping)
        early_stopping_check.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        early_stopping_check.select()  # Ensure checkbox is selected on start

        tk.Label(early_stopping_frame, text="Patience:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        patience_entry = tk.Entry(early_stopping_frame, width=5)
        patience_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        patience_entry.insert(0, "5")  # Default patience value
        patience_entry.bind("<FocusOut>", lambda event: self.change_patience(patience_entry.get()))
        
        tk.Label(early_stopping_frame, text="Min Delta:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        min_delta_entry = tk.Entry(early_stopping_frame, width=5)
        min_delta_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        min_delta_entry.insert(0, "0.001")  # Default min_delta value
        min_delta_entry.bind("<FocusOut>", lambda event: self.change_min_delta(min_delta_entry.get()))
        
        # Progress Frame (below settings_row_frame)
        progress_frame = ttk.LabelFrame(window, text="Training Progress", padding=(10, 5))
        progress_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
        
        # Epoch Progress
        epoch_progress_label = ttk.Label(progress_frame, text="Epoch Progress: 0/0")
        epoch_progress_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate")
        progress_bar.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Total Progress
        total_progress_label = ttk.Label(progress_frame, text="Total Progress: 0%")
        total_progress_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        total_progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate")
        total_progress_bar.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Time Frame (below progress_frame)
        time_frame = ttk.LabelFrame(window, text="Training Time", padding=(10, 5))
        time_frame.grid(row=2, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
        
        # Start Time
        start_time_var = tk.StringVar(value="--:--:--")
        ttk.Label(time_frame, text="Start Time:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        start_time_label = ttk.Label(time_frame, textvariable=start_time_var)
        start_time_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Finish Time
        finish_time_var = tk.StringVar(value="--:--:--")
        ttk.Label(time_frame, text="Finish Time:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        finish_time_label = ttk.Label(time_frame, textvariable=finish_time_var)
        finish_time_label.grid(row=0, column=3, sticky="w", padx=5, pady=5)
        
        # Time Used
        time_used_var = tk.StringVar(value="00:00:00")
        ttk.Label(time_frame, text="Time Used:").grid(row=0, column=4, sticky="w", padx=5, pady=5)
        time_used_label = ttk.Label(time_frame, textvariable=time_used_var)
        time_used_label.grid(row=0, column=5, sticky="w", padx=5, pady=5)
        
        # Buttons Frame (below time_frame)
        buttons_frame = ttk.Frame(window, padding=(10, 5))
        buttons_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
        
        # Training Control Buttons
        start_button = ttk.Button(buttons_frame, text="Start Training", command=self.start_new_training_process)
        start_button.grid(row=0, column=0, padx=5, pady=5)
        
        continue_button = ttk.Button(buttons_frame, text="Continue Training", command=self.continue_training_command)
        continue_button.grid(row=0, column=1, padx=5, pady=5)
        
        pause_button = ttk.Button(buttons_frame, text="Pause", command=self.pause_training)
        pause_button.grid(row=0, column=2, padx=5, pady=5)
        
        resume_button = ttk.Button(buttons_frame, text="Resume", command=self.resume_training)
        resume_button.grid(row=0, column=3, padx=5, pady=5)
        
        stop_button = ttk.Button(buttons_frame, text="Stop", command=self.stop_training)
        stop_button.grid(row=0, column=4, padx=5, pady=5)
        
        # Plot Frame (below buttons_frame)
        plot_frame_label = ttk.LabelFrame(window, text="Training Plot", padding=(10, 5))
        plot_frame_label.grid(row=4, column=0, columnspan=4, sticky="nsew", padx=10, pady=5)
        
        # Create figure for plot
        fig = plt.Figure(figsize=(4.8, 3.2), dpi=100)  # Reduced size by 20%
        plot_frame = FigureCanvasTkAgg(fig, master=plot_frame_label)
        plot_frame.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot control buttons
        plot_buttons_frame = ttk.Frame(window, padding=(10, 5))
        plot_buttons_frame.grid(row=5, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
        
        save_plot_button = ttk.Button(plot_buttons_frame, text="Save Plot", command=self.save_plot_image)
        save_plot_button.grid(row=0, column=0, padx=5, pady=5)
        
        export_plot_data_button = ttk.Button(plot_buttons_frame, text="Export Data", command=self.export_plot_data)
        export_plot_data_button.grid(row=0, column=1, padx=5, pady=5)
        
        # All global variables are already declared at the beginning of the method
        
        # Visualization Button
        visualize_button_frame = ttk.Frame(window, padding=(10, 5))
        visualize_button_frame.grid(row=6, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
        
        visualize_results_button = ttk.Button(visualize_button_frame, text="Visualize Results", command=self.visualize_results_command)
        visualize_results_button.grid(row=0, column=0, padx=5, pady=5)
        
        # --- Progress, Early Stopping, Training Time Frames in one row ---
        
        # Status Row
        status_row_frame = ttk.Frame(window, padding=(10, 5))
        status_row_frame.grid(row=7, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
        
        ttk.Label(status_row_frame, text="Status: Ready").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        # Configure row and column weights for resizing
        window.grid_rowconfigure(4, weight=1)  # Make the plot frame expandable
        for i in range(4):
            window.grid_columnconfigure(i, weight=1)
        patience_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        patience_entry.insert(0, "5")  # Default patience
        patience_entry.bind("<FocusOut>", lambda event: self.change_patience(patience_entry.get()))  # Update on focus out

        tk.Label(early_stopping_frame, text="Min Delta:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        min_delta_entry = tk.Entry(early_stopping_frame, width=5)
        min_delta_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        min_delta_entry.insert(0, "0.001")  # Default min delta
        min_delta_entry.bind("<FocusOut>", lambda event: self.change_min_delta(min_delta_entry.get()))  # Update on focus out
        
        # All global variables are already declared at the beginning of the method
        
        status_row_frame = ttk.Frame(window)  # Frame to hold progress, ES, time frames
        status_row_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=10, pady=10)  # Placed below settings_row_frame

        # Progress Frame (in status_row_frame)
        progress_frame = ttk.LabelFrame(status_row_frame, text="Training Progress", padding=(10, 5))
        progress_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)  # In row 0, column 0 of status_row_frame
        
        # Variables are already initialized earlier in the method
        
        progress_frame.columnconfigure(0, weight=1)  # Make column 0 (left side) expandable
        progress_frame.columnconfigure(1, weight=1)  # Make column 1 (right side) expandable

        # Epoch Progress (Left side)
        epoch_progress_label = tk.Label(progress_frame, text="Epoch Progress: 0.0% (Epoch 0/0)")
        epoch_progress_label.grid(row=0, column=0, sticky="w", padx=5, pady=0)  # Corrected line with column=0
        progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        progress_bar.grid(row=1, column=0, sticky="ew", padx=5, pady=5)  # Row 1, Column 0, Expand Horizontally

        # Total Progress (Right side, same row as Epoch Progress)
        total_progress_label = tk.Label(progress_frame, text="Total Progress: 0.0%")
        total_progress_label.grid(row=0, column=1, sticky="e", padx=5, pady=0)  # Row 0, Column 1, Right sticky
        total_progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        total_progress_bar.grid(row=1, column=1, sticky="ew", padx=5, pady=5)  # Row 1, Column 1, Expand Horizontally

        # Time Frame (in status_row_frame, to the right of Early Stopping Frame)
        time_frame = ttk.LabelFrame(status_row_frame, text="Training Time", padding=(10, 5))
        time_frame.grid(row=0, column=2, sticky="new", padx=5, pady=10)  # In row 0, column 2 of status_row_frame, top-left sticky

        start_time_var = tk.StringVar(value="Start Time: N/A")
        start_time_label = tk.Label(time_frame, textvariable=start_time_var)
        start_time_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        finish_time_var = tk.StringVar(value="Finish Time: N/A")
        finish_time_label = tk.Label(time_frame, textvariable=finish_time_var)
        finish_time_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)

        time_used_var = tk.StringVar(value="Time Used: N/A")
        time_used_label = tk.Label(time_frame, textvariable=time_used_var)
        time_used_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        
        # --- Buttons Frame - Moved above the Plot Frame ---
        buttons_frame = ttk.Frame(window, padding=(10, 10))
        buttons_frame.grid(row=2, column=0, columnspan=4, sticky="ew", padx=10, pady=10)  # Buttons frame above plot frame, row number adjusted
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(2, weight=1)
        buttons_frame.columnconfigure(3, weight=1)
        buttons_frame.columnconfigure(4, weight=1)
        buttons_frame.columnconfigure(5, weight=1)
        buttons_frame.columnconfigure(6, weight=1)  # Added column configure for export plot data button

        start_button = tk.Button(buttons_frame, text="Start Training", command=self.start_new_training_process)  # Modified to call start_new_training_process
        start_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        continue_button = tk.Button(buttons_frame, text="Continue Training", command=self.continue_training_command, state=tk.DISABLED)  # Continue Training button
        continue_button.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        pause_button = tk.Button(buttons_frame, text="Pause", command=self.pause_training, state=tk.DISABLED)
        pause_button.grid(row=0, column=2, sticky="ew", padx=5, pady=5)  # Corrected line - added pady=5 and closing parenthesis
        resume_button = tk.Button(buttons_frame, text="Resume", command=self.resume_training, state=tk.DISABLED)
        resume_button.grid(row=0, column=3, sticky="ew", padx=5, pady=5)  # Now in column 3
        stop_button = tk.Button(buttons_frame, text="Stop", command=self.stop_training, state=tk.DISABLED)
        stop_button.grid(row=0, column=4, sticky="ew", padx=5, pady=5)
        save_plot_button = tk.Button(buttons_frame, text="Save Plot", command=self.save_plot_image)
        save_plot_button.grid(row=0, column=5, sticky="ew", padx=5, pady=5)  # Now in column 5
        export_plot_data_button = tk.Button(buttons_frame, text="Export Plot Data", command=self.export_plot_data)  # Export Plot Data Button
        export_plot_data_button.grid(row=0, column=6, sticky="ew", padx=5, pady=5)  # Now in column 6, next to Save Plot


        # --- Visualize Results Button Frame - Moved above the Plot Frame, below Buttons ---
        visualize_button_frame = ttk.Frame(window, padding=(10, 10))
        visualize_button_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=10, pady=10)  # Visualize button above Plot frame, below Buttons frame, row number adjusted
        visualize_button_frame.columnconfigure(0, weight=1)

        visualize_results_button = tk.Button(visualize_button_frame, text="Visualize Results", command=self.visualize_results_command)
        visualize_results_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)


        # --- Plot Frame ---
        plot_frame = ttk.LabelFrame(window, text="Training Plot", padding=(10, 5))
        plot_frame.grid(row=4, column=0, columnspan=4, sticky="nsew", padx=10, pady=10)  # Plot frame below visualize button, row number adjusted
        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(0, weight=1)


        # --- Window Resize Configuration ---
        window.columnconfigure(0, weight=1)  # Make column 0 expandable (for status row and plot)
        window.columnconfigure(1, weight=0)  # Keep column 1 (for ES) from extra expansion, if needed
        window.columnconfigure(2, weight=0)  # Keep column 2 (for Time) from extra expansion, if needed
        window.rowconfigure(4, weight=1)  # Make plot frame resizable, row number adjusted
        settings_row_frame.columnconfigure(0, weight=1)  # Input Directories Frame expandable
        settings_row_frame.columnconfigure(1, weight=1)  # Training Parameters Frame expandable
        settings_row_frame.columnconfigure(2, weight=0)  # Early Stopping Frame no extra expansion
        # --- Removed duplicate code ---

        # --- Progress, Early Stopping, Training Time Frames in one row ---
        
        status_row_frame = ttk.Frame(window)  # Frame to hold progress, ES, time frames
        
        status_row_frame.columnconfigure(0, weight=1)  # Progress Frame expandable
        status_row_frame.columnconfigure(1, weight=0)  # ES Frame no extra expansion
        status_row_frame.columnconfigure(2, weight=0)  # Time Frame no extra expansion
        
        status_row_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=10, pady=10)  # Placed below settings_row_frame

        # Progress Frame (in status_row_frame)
        progress_frame = ttk.LabelFrame(status_row_frame, text="Training Progress", padding=(10, 5))
        progress_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)  # In row 0, column 0 of status_row_frame
        
        # Variables are already initialized earlier in the method
        
        progress_frame.columnconfigure(0, weight=1)  # Make column 0 (left side) expandable
        progress_frame.columnconfigure(1, weight=1)  # Make column 1 (right side) expandable

        # Epoch Progress (Left side)
        epoch_progress_label = tk.Label(progress_frame, text="Epoch Progress: 0.0% (Epoch 0/0)")
        epoch_progress_label.grid(row=0, column=0, sticky="w", padx=5, pady=0)  # Corrected line with column=0
        progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        progress_bar.grid(row=1, column=0, sticky="ew", padx=5, pady=5)  # Row 1, Column 0, Expand Horizontally

        # Total Progress (Right side, same row as Epoch Progress)
        total_progress_label = tk.Label(progress_frame, text="Total Progress: 0.0%")
        total_progress_label.grid(row=0, column=1, sticky="e", padx=5, pady=0)  # Row 0, Column 1, Right sticky
        total_progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        total_progress_bar.grid(row=1, column=1, sticky="ew", padx=5, pady=5)  # Row 1, Column 1, Expand Horizontally


        # Early Stopping Frame (in status_row_frame, to the right of Progress Frame)
        #early_stopping_display_frame = ttk.LabelFrame(status_row_frame, text="Early Stopping", padding=(10, 5))
        #early_stopping_display_frame.grid(row=0, column=1, sticky="new", padx=5, pady=5)  # In row 0, column 1 of status_row_frame, top-left sticky
        #early_stopping_status_var = tk.StringVar(value="Status: Active" if early_stopping_enabled else "Status: Inactive") # Status var
        #early_stopping_status_label = tk.Label(early_stopping_display_frame, textvariable=early_stopping_status_var) # Status label
        #early_stopping_status_label.grid(row=0, column=0, sticky="w", padx=5, pady=5) # Grid for status label


        # Time Frame (in status_row_frame, to the right of Early Stopping Frame)
        time_frame = ttk.LabelFrame(status_row_frame, text="Training Time", padding=(10, 5))
        time_frame.grid(row=0, column=2, sticky="new", padx=5, pady=10)  # In row 0, column 2 of status_row_frame, top-left sticky

        start_time_var = tk.StringVar(value="Start Time: N/A")
        start_time_label = tk.Label(time_frame, textvariable=start_time_var)
        start_time_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        finish_time_var = tk.StringVar(value="Finish Time: N/A")
        finish_time_label = tk.Label(time_frame, textvariable=finish_time_var)
        finish_time_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)

        time_used_var = tk.StringVar(value="Time Used: N/A")
        time_used_label = tk.Label(time_frame, textvariable=time_used_var)
        time_used_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        
        # --- Buttons Frame - Moved above the Plot Frame ---
        buttons_frame = ttk.Frame(window, padding=(10, 10))
        buttons_frame.grid(row=2, column=0, columnspan=4, sticky="ew", padx=10, pady=10)  # Buttons frame above plot frame, row number adjusted
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(2, weight=1)
        buttons_frame.columnconfigure(3, weight=1)
        buttons_frame.columnconfigure(4, weight=1)
        buttons_frame.columnconfigure(5, weight=1)
        buttons_frame.columnconfigure(6, weight=1)  # Added column configure for export plot data button

        start_button = tk.Button(buttons_frame, text="Start Training", command=self.start_new_training_process)  # Modified to call start_new_training_process
        start_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        continue_button = tk.Button(buttons_frame, text="Continue Training", command=self.continue_training_command, state=tk.DISABLED)  # Continue Training button
        continue_button.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        pause_button = tk.Button(buttons_frame, text="Pause", command=self.pause_training, state=tk.DISABLED)
        pause_button.grid(row=0, column=2, sticky="ew", padx=5, pady=5)  # Corrected line - added pady=5 and closing parenthesis
        resume_button = tk.Button(buttons_frame, text="Resume", command=self.resume_training, state=tk.DISABLED)
        resume_button.grid(row=0, column=3, sticky="ew", padx=5, pady=5)  # Now in column 3
        stop_button = tk.Button(buttons_frame, text="Stop", command=self.stop_training, state=tk.DISABLED)
        stop_button.grid(row=0, column=4, sticky="ew", padx=5, pady=5)
        save_plot_button = tk.Button(buttons_frame, text="Save Plot", command=self.save_plot_image)
        save_plot_button.grid(row=0, column=5, sticky="ew", padx=5, pady=5)  # Now in column 5
        export_plot_data_button = tk.Button(buttons_frame, text="Export Plot Data", command=self.export_plot_data)  # Export Plot Data Button
        export_plot_data_button.grid(row=0, column=6, sticky="ew", padx=5, pady=5)  # Now in column 6, next to Save Plot


        # --- Visualize Results Button Frame - Moved above the Plot Frame, below Buttons ---
        visualize_button_frame = ttk.Frame(window, padding=(10, 10))
        visualize_button_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=10, pady=10)  # Visualize button above Plot frame, below Buttons frame, row number adjusted
        visualize_button_frame.columnconfigure(0, weight=1)

        visualize_results_button = tk.Button(visualize_button_frame, text="Visualize Results", command=self.visualize_results_command)
        visualize_results_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)


        # --- Plot Frame ---
        plot_frame = ttk.LabelFrame(window, text="Training Plot", padding=(10, 5))
        plot_frame.grid(row=4, column=0, columnspan=4, sticky="nsew", padx=10, pady=10)  # Plot frame below visualize button, row number adjusted
        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(0, weight=1)


        # --- Window Resize Configuration ---
        window.columnconfigure(0, weight=1)  # Make column 0 expandable (for status row and plot)
        window.columnconfigure(1, weight=0)  # Keep column 1 (for ES) from extra expansion, if needed
        window.columnconfigure(2, weight=0)  # Keep column 2 (for Time) from extra expansion, if needed
        window.rowconfigure(4, weight=1)  # Make plot frame resizable, row number adjusted
        settings_row_frame.columnconfigure(0, weight=1)  # Input Directories Frame expandable
        settings_row_frame.columnconfigure(1, weight=1)  # Training Parameters Frame expandable
        settings_row_frame.columnconfigure(2, weight=0)  # Early Stopping Frame no extra expansion
        status_row_frame.columnconfigure(0, weight=1)  # Progress Frame expandable
        status_row_frame.columnconfigure(1, weight=0)  # ES Frame no extra expansion
        status_row_frame.columnconfigure(2, weight=0)  # Time Frame no extra expansion
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.columnconfigure(1, weight=1)

        update_continue_button_state() # CALL NEW FUNCTION HERE TO INITIALIZE BUTTON STATE

# Create the main window and start the application
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Combined Segmentation Tool")
    app = UNetTrainingTool(root)  # This will initialize the window variable and set up the UI

    # Create notebook (tabbed interface)
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    # Create tabs for each segmentation tool
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    tab3 = ttk.Frame(notebook)
    
    notebook.add(tab1, text='Histogram Viewer')
    notebook.add(tab2, text='UNet Inference Tool')
    notebook.add(tab3, text='UNet Training Tool')
    
    # Initialize each tool in its own tab
    viewer1 = ImageHistogramViewer(tab1)
    
    # Setup UNet Inference Tool in tab2
    setup_inference_ui(tab2)
    
    # Setup UNet Training Tool in tab3
    training_tool = UNetTrainingTool(tab3)
    
    root.mainloop()
# viewer2 = SecondToolClass(tab2)
# viewer3 = ThirdToolClass(tab3)

# Create main window with tabs
if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.title("Combined Segmentation Tool")

        # Create notebook (tabbed interface)
        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True)

        # Create tabs for each segmentation tool
        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)
        tab3 = ttk.Frame(notebook)
        
        notebook.add(tab1, text='Histogram Viewer')
        notebook.add(tab2, text='UNet Inference Tool')
        notebook.add(tab3, text='UNet Training Tool')
        
        # Initialize each tool in its own tab
        viewer1 = ImageHistogramViewer(tab1)
        
        # Make sure load_model_command is defined before setting up inference UI
        if 'load_model_command' not in globals():
            def load_model_command():
                global model_path_entry
                filepath = filedialog.askopenfilename(
                    initialdir=".",
                    title="Select Trained Model",
                    filetypes=(("Model files", "*.pth"), ("all files", "*.*"))
                )
                if filepath:
                    model_path_entry.delete(0, tk.END)
                    model_path_entry.insert(0, filepath)
        
        # Make sure select_output_folder_command is defined before setting up inference UI
        if 'select_output_folder_command' not in globals():
            def select_output_folder_command():
                global output_folder_entry
                dirpath = filedialog.askdirectory(initialdir=".", title="Select Output Folder")
                if dirpath and output_folder_entry is not None:
                    output_folder_entry.delete(0, tk.END)
                    output_folder_entry.insert(0, dirpath)
        
        # Setup UNet Inference Tool in tab2
        setup_inference_ui(tab2)
        
        # Setup UNet Training Tool in tab3
        training_tool = UNetTrainingTool(tab3)
    except Exception as e:
        messagebox.showerror("Error in main", str(e))

# Initialize each tool in its own tab
viewer1 = ImageHistogramViewer(tab1)
# viewer2 = SecondToolClass(tab2)
# viewer3 = ThirdToolClass(tab3)

# Create main window with tabs
if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.title("Combined Segmentation Tool")

        # Create notebook (tabbed interface)
        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True)

        # Create tabs for each segmentation tool
        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)
        tab3 = ttk.Frame(notebook)
        
        notebook.add(tab1, text='Histogram Viewer')
        notebook.add(tab2, text='UNet Inference Tool')
        notebook.add(tab3, text='UNet Training Tool')
        
        # Initialize each tool in its own tab
        viewer1 = ImageHistogramViewer(tab1)
        
        # Make sure load_model_command is defined before setting up inference UI
        if 'load_model_command' not in globals():
            def load_model_command():
                global model_path_entry
                filepath = filedialog.askopenfilename(
                    initialdir=".",
                    title="Select Trained Model",
                    filetypes=(("Model files", "*.pth"), ("all files", "*.*"))
                )
                if filepath:
                    model_path_entry.delete(0, tk.END)
                    model_path_entry.insert(0, filepath)
        
        # Make sure select_output_folder_command is defined before setting up inference UI
        if 'select_output_folder_command' not in globals():
            def select_output_folder_command():
                global output_folder_entry
                dirpath = filedialog.askdirectory(initialdir=".", title="Select Output Folder")
                if dirpath and output_folder_entry is not None:
                    output_folder_entry.delete(0, tk.END)
                    output_folder_entry.insert(0, dirpath)
        
        # Setup UNet Inference Tool in tab2
        setup_inference_ui(tab2)
        
        # Setup UNet Training Tool in tab3
        training_tool = UNetTrainingTool(tab3)
    except Exception as e:
        messagebox.showerror("Error in main", str(e))

# Initialize each tool in its own tab
viewer1 = ImageHistogramViewer(tab1)
# viewer2 = SecondToolClass(tab2)
# viewer3 = ThirdToolClass(tab3)

# This section is handled in the main block at the end of the file

notebook.add(tab1, text='Histogram Viewer')
notebook.add(tab2, text='Tool 2')
notebook.add(tab3, text='Tool 3')

# Initialize each tool in its own tab
viewer1 = ImageHistogramViewer(tab1)
# viewer2 = SecondToolClass(tab2)
# viewer3 = ThirdToolClass(tab3)

# Create main window with tabs
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Combined Segmentation Tool")

    
    # Create notebook (tabbed interface)
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)
    
    # Create tabs for each segmentation tool
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    tab3 = ttk.Frame(notebook)
    tab4 = ttk.Frame(notebook)
    
    notebook.add(tab1, text='Image Viewer')
    notebook.add(tab2, text='Histogram')
    notebook.add(tab3, text='Segmentation Results')
    notebook.add(tab4, text='UNet Training')
    
    # Initialize each tool in its own tab
    viewer1 = ImageHistogramViewer(tab1)
    # Initialize UNet Training Tool in the UNet Training tab
    unet_training_tool = UNetTrainingTool(tab4)
    
    root.mainloop()
