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

class ImageHistogramViewer:
    def __init__(self, master):
        self.master = master
        master.title("Image Histogram Viewer")

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
        file_frame = ttk.Frame(master, padding="10 10 10 10")
        file_frame.pack(fill=tk.X)

        ttk.Label(file_frame, text="Image File:").grid(row=0, column=0, sticky=tk.W)
        self.file_path_label = ttk.Label(file_frame, textvariable=self.image_path, wraplength=300)
        self.file_path_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(file_frame, text="Open Image", command=self.open_image_file).grid(row=0, column=2, sticky=tk.E, padx=5)
        
        # Add Gaussian components selection
        components_frame = ttk.Frame(master, padding="10 5 10 5")
        components_frame.pack(fill=tk.X)
        ttk.Label(components_frame, text="Number of Gaussian Distributions:").pack(side=tk.LEFT, padx=5)
        components_spinbox = ttk.Spinbox(components_frame, from_=1, to=9, width=5, textvariable=self.n_components)
        components_spinbox.pack(side=tk.LEFT, padx=5)

        # --- Output Frame (for Image and Histogram Display) ---
        self.output_frame = ttk.Frame(master, padding="10 10 10 10")
        self.output_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left frame for image
        self.image_frame = ttk.Frame(self.output_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create right frame for histogram
        self.histogram_frame = ttk.Frame(self.output_frame)
        self.histogram_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create results frame for segmentation results
        self.results_frame = ttk.Frame(master, padding="10 10 10 10")
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create frames for each result image
        self.comp1_gray_frame = ttk.Frame(self.results_frame)
        self.comp1_gray_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.comp2_gray_frame = ttk.Frame(self.results_frame)
        self.comp2_gray_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.combined_viz_frame = ttk.Frame(self.results_frame)
        self.combined_viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Setup image display
        self.fig, self.ax = plt.subplots(figsize=(1, 1))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Remove axes from main image display
        self.ax.axis('off')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.patch.set_visible(False)
        self.fig.set_facecolor('none')
        
        # Setup result image displays
        self.comp1_gray_fig, self.comp1_gray_ax = plt.subplots(figsize=(1, 1))
        self.comp1_gray_canvas = FigureCanvasTkAgg(self.comp1_gray_fig, master=self.comp1_gray_frame)
        self.comp1_gray_canvas_widget = self.comp1_gray_canvas.get_tk_widget()
        self.comp1_gray_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Remove axes from component 1 display
        self.comp1_gray_ax.axis('off')
        self.comp1_gray_ax.set_xticks([])
        self.comp1_gray_ax.set_yticks([])
        self.comp1_gray_fig.patch.set_visible(False)
        self.comp1_gray_fig.set_facecolor('none')
        
        self.comp2_gray_fig, self.comp2_gray_ax = plt.subplots(figsize=(1, 1))
        self.comp2_gray_canvas = FigureCanvasTkAgg(self.comp2_gray_fig, master=self.comp2_gray_frame)
        self.comp2_gray_canvas_widget = self.comp2_gray_canvas.get_tk_widget()
        self.comp2_gray_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Remove axes from component 2 display
        self.comp2_gray_ax.axis('off')
        self.comp2_gray_ax.set_xticks([])
        self.comp2_gray_ax.set_yticks([])
        self.comp2_gray_fig.patch.set_visible(False)
        self.comp2_gray_fig.set_facecolor('none')
        
        self.combined_viz_fig, self.combined_viz_ax = plt.subplots(figsize=(1, 1))
        self.combined_viz_canvas = FigureCanvasTkAgg(self.combined_viz_fig, master=self.combined_viz_frame)
        self.combined_viz_canvas_widget = self.combined_viz_canvas.get_tk_widget()
        self.combined_viz_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Remove axes from combined visualization display
        self.combined_viz_ax.axis('off')
        self.combined_viz_ax.set_xticks([])
        self.combined_viz_ax.set_yticks([])
        self.combined_viz_fig.patch.set_visible(False)
        self.combined_viz_fig.set_facecolor('none')
        
        # Setup histogram display
        self.hist_fig, self.hist_ax = plt.subplots(figsize=(2.5, 1))
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.histogram_frame)
        self.hist_canvas_widget = self.hist_canvas.get_tk_widget()
        self.hist_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Remove axes from histogram display - completely remove all elements
        self.hist_ax.axis('off')  # Turn off axes completely
        self.hist_ax.set_xticks([])
        self.hist_ax.set_yticks([])
        self.hist_ax.set_xticklabels([])
        self.hist_ax.set_yticklabels([])
        # Remove all spines
        for spine in self.hist_ax.spines.values():
            spine.set_visible(False)
        self.hist_fig.patch.set_visible(False)  # Make figure background transparent
        self.hist_fig.set_facecolor('none')  # Make figure face color transparent
        
        # Initialize threshold variable
        self.threshold_var = tk.DoubleVar(value=0.5)
        
        # We'll store the radio buttons in lists for easy access
        self.comp1_radio_buttons = []
        self.comp2_radio_buttons = []
        
        # Define colors for up to 9 components
        self.component_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown']
        self.component_color_names = ['Red', 'Blue', 'Green', 'Purple', 'Orange', 'Cyan', 'Magenta', 'Yellow', 'Brown']
                
        # Add a method to update the component selection UI when the number of components changes
        def update_component_selection(*args):
            n_comp = self.n_components.get()
            # Update component 1 radio buttons
            for i, rb in enumerate(self.comp1_radio_buttons):
                if i < n_comp:
                    rb.pack(side=tk.LEFT, padx=5)
                else:
                    rb.pack_forget()
            # Update component 2 radio buttons
            for i, rb in enumerate(self.comp2_radio_buttons):
                if i < n_comp:
                    rb.pack(side=tk.LEFT, padx=5)
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
        self.threshold_container_frame = ttk.Frame(master)
        self.threshold_container_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add threshold method selection frame
        self.threshold_method_frame = ttk.LabelFrame(self.threshold_container_frame, text="Threshold Method")
        self.threshold_method_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Add threshold method selection controls
        ttk.Label(self.threshold_method_frame, text="Select threshold method:").pack(anchor=tk.W, padx=5, pady=2)
        
        # Threshold method selection
        method_frame = ttk.Frame(self.threshold_method_frame)
        method_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Create radio buttons for threshold methods
        methods = ["GMM", "Otsu", "Triangle", "IsoData"]
        for method in methods:
            rb = ttk.Radiobutton(method_frame, text=method, variable=self.threshold_method, value=method,
                                command=self.update_threshold_method)
            rb.pack(side=tk.LEFT, padx=5)
        
        # Add threshold adjustment frame
        self.threshold_control_frame = ttk.LabelFrame(self.threshold_container_frame, text="Threshold Adjustment")
        self.threshold_control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add component selection frame
        self.component_selection_frame = ttk.LabelFrame(self.threshold_container_frame, text="Component Selection")
        self.component_selection_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add threshold adjustment control
        self.threshold_var = tk.DoubleVar(value=0.5)
        
        threshold_frame = ttk.Frame(self.threshold_control_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(threshold_frame, text="Value:").pack(side=tk.LEFT)
        # Initialize with default values, will be updated when image is loaded
        self.threshold_scale = tk.Scale(threshold_frame, from_=0, to=255, resolution=0.0002, orient=tk.HORIZONTAL,
                                    variable=self.threshold_var, command=self.update_threshold)
        self.threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add threshold value display
        self.threshold_value_label = ttk.Label(threshold_frame, text="0.5000")
        self.threshold_value_label.pack(side=tk.RIGHT, padx=5)
        
        # Add component selection controls
        ttk.Label(self.component_selection_frame, text="Select components for segmentation:").pack(anchor=tk.W, padx=5, pady=2)
        
        # Component 1 selection
        comp1_frame = ttk.Frame(self.component_selection_frame)
        comp1_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(comp1_frame, text="Component 1:").pack(side=tk.LEFT)
        
        # Create radio buttons for component 1 (up to 9)
        for i in range(9):
            rb = ttk.Radiobutton(comp1_frame, text=f"C{i+1} ({self.component_color_names[i]})", 
                                variable=self.selected_comp1, value=i)
            rb.pack(side=tk.LEFT, padx=5)
            self.comp1_radio_buttons.append(rb)
            # Hide buttons beyond the current number of components
            if i >= self.n_components.get():
                rb.pack_forget()
        
        # Component 2 selection
        comp2_frame = ttk.Frame(self.component_selection_frame)
        comp2_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(comp2_frame, text="Component 2:").pack(side=tk.LEFT)
        
        # Create radio buttons for component 2 (up to 9)
        for i in range(9):
            rb = ttk.Radiobutton(comp2_frame, text=f"C{i+1} ({self.component_color_names[i]})", 
                                variable=self.selected_comp2, value=i)
            rb.pack(side=tk.LEFT, padx=5)
            self.comp2_radio_buttons.append(rb)
            # Hide buttons beyond the current number of components
            if i >= self.n_components.get():
                rb.pack_forget()
        
        # Add segmentation and save controls
        self.control_frame = ttk.Frame(master)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Segmentation button
        ttk.Button(self.control_frame, text="Apply Threshold Segmentation", 
                  command=self.apply_segmentation).pack(side=tk.LEFT, padx=5)
        
        # Save button
        ttk.Button(self.control_frame, text="Save Results", 
                  command=self.save_segmented_image).pack(side=tk.LEFT, padx=5)
        
        # Add status bar
        self.status_label = ttk.Label(master, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
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
            except Exception as e:
                self.status_label.config(text=f"Error displaying image: {e}")

    def display_image(self, img_pil, title=None):
        self.ax.clear()
        # Use appropriate normalization based on image mode
        if img_pil.mode in ['I', 'F']:
            # For high bit-depth images, normalize properly
            img_array = np.array(img_pil)
            vmin = np.min(img_array)
            vmax = np.max(img_array)
            self.ax.imshow(img_array, cmap='gray', vmin=vmin, vmax=vmax)
        else:
            self.ax.imshow(img_pil, cmap='gray')
            
        # Completely remove frame and axes
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
        
        self.canvas.draw()

    def update_threshold_method(self):
        """Calculate and update threshold based on the selected threshold method"""
        if self.current_img is None or self.gmm is None:
            return
            
        # Get the selected threshold method
        method = self.threshold_method.get()
        
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
    
    def update_threshold(self, *args, step=0.0001):
        if self.current_img is not None and self.gmm is not None:
            # Get the new threshold value
            if len(args) > 0 and isinstance(args[0], float):
                # Round to the nearest step value for precise control
                threshold = round(args[0] / step) * step
                self.threshold_var.set(threshold)
            else:
                threshold = self.threshold_var.get()
            
            # Update the threshold value for segmentation
            self.threshold_value = threshold
            
            # Update the threshold value label with 4 decimal places
            self.threshold_value_label.config(text=f"{threshold:.4f}")
            
            # Redraw the histogram with the new threshold line
            self.update_histogram_threshold()
    
    def update_histogram_threshold(self):
        if self.current_img is not None and self.gmm is not None and self.threshold_value is not None:
            # Get the current y-axis limits
            y_min, y_max = self.hist_ax.get_ylim()
            
            # Remove any existing threshold line
            for line in self.hist_ax.lines:
                if line.get_color() == 'green' and line.get_linestyle() == '--':
                    line.remove()
            
            # Remove any existing threshold text
            for text in self.hist_ax.texts:
                if 'Threshold' in text.get_text():
                    text.remove()
            
            # Add the new threshold line
            self.hist_ax.axvline(x=self.threshold_value, color='green', linestyle='--', linewidth=1.5)
            self.hist_ax.text(self.threshold_value, y_max*0.9, 
                             f'Threshold: {self.threshold_value:.4f}', 
                             color='green', ha='center', va='top',
                             bbox=dict(facecolor='white', alpha=0.7))
            
            # Make sure axes are visible for the histogram
            self.hist_ax.axis('on')
            for spine in self.hist_ax.spines.values():
                spine.set_visible(True)
                
            # Make sure ticks are visible and properly formatted
            self.hist_ax.tick_params(axis='both', which='both', length=4, width=1, direction='out', labelsize=8)
            
            # Ensure axis labels remain visible
            self.hist_ax.set_xlabel('Pixel Intensity', fontsize=10, fontweight='bold')
            self.hist_ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            
            # Redraw the canvas
            self.hist_canvas.draw()

    def plot_histogram(self, img):
        # Clear previous histogram
        self.hist_ax.clear()

        # Get pixel values and filter out black pixels (ROI)
        pixels = np.array(img).flatten()
        non_black_pixels = pixels[pixels > 0]  # Filter out black pixels
        
        # Calculate statistics using non-black pixels
        n_pixels = len(non_black_pixels)
        pixel_min = np.min(non_black_pixels)
        pixel_max = np.max(non_black_pixels)
        pixel_mean = np.mean(non_black_pixels)
        pixel_std = np.std(non_black_pixels)
        
        # Set initial axis limits
        self.hist_ax.set_xlim(pixel_min, pixel_max)
        
        # Set initial threshold value to middle of range
        initial_threshold = (pixel_min + pixel_max) / 2
        self.threshold_var.set(initial_threshold)
        
        # Update threshold scale range to match histogram range
        self.threshold_scale.config(from_=pixel_min, to=pixel_max)
        
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
        
        # Plot histogram with black bars and no transparency (without label)
        hist_data = self.hist_ax.hist(non_black_pixels, bins=num_bins, color='black', edgecolor='black', alpha=0.6)
        
        # Fit Gaussian Mixture Model with user-specified number of components
        n_components = self.n_components.get()
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        
        # Reshape data for GMM fitting
        X = non_black_pixels.reshape(-1, 1)
        gmm.fit(X)
        self.gmm = gmm  # Store the GMM model for later use
        
        # Generate x values for plotting the GMM
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
            self.hist_ax.plot(x, hist_values.max() * pdf_component / pdf.max(), 
                             color=colors[i], linewidth=2, 
                             label=f'C{i+1}: μ={gmm.means_[i][0]:.3f}, σ={np.sqrt(gmm.covariances_[i][0][0]):.3f}')
        
        # Add legend with draggable option
        legend = self.hist_ax.legend(loc='upper right', fontsize='small')
        legend.set_draggable(True)
        
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
        
        # Calculate threshold based on the selected method
        self.update_threshold_method()
        
        # Redraw the canvas
        self.hist_canvas.draw()
        
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
        
        # Use the threshold value from the slider for segmentation
        threshold = self.threshold_var.get()
        
        # If this is the first time applying segmentation, use the calculated optimal threshold
        if self.segmented_img is None and self.threshold_value is not None:
            # Update the threshold variable to match the calculated threshold
            self.threshold_var.set(self.threshold_value)
            threshold = self.threshold_value
        
        # Create masks for each component based on probability threshold
        # For component 1
        component1_array[prob1 > threshold] = 255
        
        # For component 2 - make it the original image minus component 1
        # First identify all non-zero pixels in the original image
        non_zero_mask = img_array > 0
        # Then exclude component 1 pixels
        component2_array[non_zero_mask & (component1_array == 0)] = 255
        
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
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 1))
        
        # Get component colors for display
        colors = self.component_color_names
        
        # Display both component images
        ax1.imshow(component1_img, cmap='gray')
        ax1.set_title(f"C{comp1_idx+1} ({colors[comp1_idx]} in Histogram)")
        ax1.axis('off')
        
        ax2.imshow(component2_img, cmap='gray')
        ax2.set_title(f"C{comp2_idx+1} ({colors[comp2_idx]} in Histogram)")
        ax2.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Create grayscale segmentations preserving original intensities
        original_array = np.array(self.current_img)
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
        
        # Always use red for component 1 visualization regardless of selection
        comp1_viz[component1_array > 0] = [255, 0, 0]  # Red for component 1
        # Always use blue for component 2 visualization regardless of selection
        comp2_viz[component2_array > 0] = [0, 0, 255]  # Blue for component 2
        
        # Create combined visualization
        combined_viz = np.zeros((component1_array.shape[0], component1_array.shape[1], 3), dtype=np.uint8)
        
        # Create combined visualization with black background
        # First, ensure all pixels are black (background)
        combined_viz.fill(0)  # Set all pixels to black (0,0,0)
        
        # Then assign component 2 pixels as blue
        combined_viz[component2_array > 0] = [0, 0, 255]  # Blue for component 2
        
        # Then assign component 1 pixels as red (this will override any overlapping areas)
        combined_viz[component1_array > 0] = [255, 0, 0]  # Red for component 1
        
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
        self.comp1_gray_ax.set_title(f"Component {comp1_idx+1} Grayscale ({comp1_percent:.2f}%)")
        self.comp1_gray_ax.axis('off')
        self.comp1_gray_fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0, hspace=0)
        self.comp1_gray_fig.tight_layout(pad=0)
        self.comp1_gray_canvas.draw()
        
        # Display component2_gray with percentage
        self.comp2_gray_ax.clear()
        self.comp2_gray_ax.imshow(comp2_grayscale, cmap='gray')
        self.comp2_gray_ax.set_title(f"Component {comp2_idx+1} Grayscale ({comp2_percent:.2f}%)")
        self.comp2_gray_ax.axis('off')
        self.comp2_gray_fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0, hspace=0)
        self.comp2_gray_fig.tight_layout(pad=0)
        self.comp2_gray_canvas.draw()
        
        # Display combined visualization
        self.combined_viz_ax.clear()
        self.combined_viz_ax.imshow(combined_viz)
        self.combined_viz_ax.set_title("Combined Visualization")
        self.combined_viz_ax.axis('off')
        self.combined_viz_fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0, hspace=0)
        self.combined_viz_fig.tight_layout(pad=0)
        self.combined_viz_canvas.draw()
        
        # Display pixel statistics in status bar
        stats_text = f"Segmentation applied. Pixels - Component {comp1_idx+1}: {self.pixel_stats['comp1_pixels']} ({self.pixel_stats['comp1_percent']:.2f}%), Mean: {self.pixel_stats['comp1_mean']:.2f}, StdDev: {self.pixel_stats['comp1_std']:.2f} | Component {comp2_idx+1}: {self.pixel_stats['comp2_pixels']} ({self.pixel_stats['comp2_percent']:.2f}%), Mean: {self.pixel_stats['comp2_mean']:.2f}, StdDev: {self.pixel_stats['comp2_std']:.2f} | Total Mean: {self.pixel_stats['total_mean']:.2f}, StdDev: {self.pixel_stats['total_std']:.2f}"
        self.status_label.config(text=stats_text)
        
    def update_histogram_with_selected_components(self, comp1_idx, comp2_idx):
        """Update the histogram to show only the selected components"""
        if self.current_img is None or self.gmm is None:
            return
            
        # Get pixel values and filter out black pixels (ROI)
        pixels = np.array(self.current_img).flatten()
        non_black_pixels = pixels[pixels > 0]  # Filter out black pixels
        
        # Calculate statistics
        pixel_min = np.min(non_black_pixels)
        pixel_max = np.max(non_black_pixels)
        
        # Adjust bin count based on image bit depth
        if self.current_img.mode in ['I', 'F']:
            num_bins = min(1024, len(np.unique(non_black_pixels)))
        else:
            num_bins = 256
            
        # Calculate histogram values
        hist_values, bin_edges = np.histogram(non_black_pixels, bins=num_bins)
        
        # Clear previous histogram
        self.hist_ax.clear()
        
        # Plot histogram with black bars
        self.hist_ax.hist(non_black_pixels, bins=num_bins, color='black', edgecolor='black', alpha=0.6)
        
        # Generate x values for plotting the GMM
        x = np.linspace(pixel_min, pixel_max, 1000).reshape(-1, 1)
        
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
            self.hist_ax.plot(x, hist_values.max() * pdf_component / pdf.max(), 
                             color=colors[i], linewidth=2, 
                             label=f'C{i+1}: μ={self.gmm.means_[i][0]:.3f}, σ={np.sqrt(self.gmm.covariances_[i][0][0]):.3f}')
        
        # Add the threshold line
        if self.threshold_value is not None:
            self.hist_ax.axvline(x=self.threshold_value, color='green', linestyle='--', linewidth=1.5)
            self.hist_ax.text(self.threshold_value, self.hist_ax.get_ylim()[1]*0.9, 
                             f'Threshold: {self.threshold_value:.4f}', 
                             color='green', ha='center', va='top',
                             bbox=dict(facecolor='white', alpha=0.7))
        
        # Set axis limits
        self.hist_ax.set_xlim(pixel_min, pixel_max)
        self.hist_ax.set_ylim(0, np.max(hist_values) * 1.05)
        
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
                    # Set figure size to 7.5x5 inches for saving
                    self.hist_fig.set_size_inches(7.5, 5)
                    
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
                    # Restore original figure size
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

def main():
    # Set up signal handler for keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        root = tk.Tk()
        root.geometry("1200x800")
        # Add protocol to handle window close event with proper cleanup
        root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
        app = ImageHistogramViewer(root)
        root.mainloop()
    except KeyboardInterrupt:
        print('\nExiting gracefully (KeyboardInterrupt caught)')
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()