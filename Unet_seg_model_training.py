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
        fig_plot, ax1_plot = plt.subplots(figsize=(8, 6))
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
    global is_stopped, training_thread, window, patience_timer

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
        #print(f'initial epoch {initial_epoch}, current epoch {epoch}')
        epoch_progress_label.config(text=f"Epoch Progress: {batch_percentage:.1f}% (Epoch {epoch}/{total_epochs})")
        #epoch_progress_label.config(text=f"Epoch Progress: {batch_percentage:.1f}% (Epoch {initial_epoch}/{total_epochs})")
        progress_bar['value'] = batch_percentage

    def update_total_progress(percentage):
        total_progress_bar['value'] = percentage

    def update_total_progress_label_gui(text):
        total_progress_label.config(text=text)

    def set_training_finished(finished):
        global training_finished
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
        update_plot(losses, dice_scores)

    def update_training_time_gui(start_time_str, finish_time_str, duration_str):
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
    global early_stopping_enabled, patience_entry, min_delta_entry
    early_stopping_enabled = early_stopping_check_var.get()
    if early_stopping_enabled:
        patience_entry.config(state=tk.NORMAL)
        min_delta_entry.config(state=tk.NORMAL)
    else:
        patience_entry.config(state=tk.DISABLED)
        min_delta_entry.config(state=tk.DISABLED)

def change_patience(value):
    global early_stopping_patience
    if value.isdigit():
        early_stopping_patience = int(value)
    else:
        messagebox.showerror("Error", "Patience must be an integer.")
        patience_entry.delete(0, tk.END) # Clear entry on error
        patience_entry.insert(0, str(early_stopping_patience)) # Re-insert valid value

def change_min_delta(value):
    global early_stopping_min_delta
    try:
        early_stopping_min_delta = float(value)
    except ValueError:
        messagebox.showerror("Error", "Min Delta must be a float.")
        min_delta_entry.delete(0, tk.END) # Clear entry on error
        min_delta_entry.insert(0, str(early_stopping_min_delta)) # Re-insert valid value

def update_continue_button_state():
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

# --- Main GUI Setup ---
window = tk.Tk()
window.title("UNet Training GUI")  # Changed title to UNet
window.protocol("WM_DELETE_WINDOW", on_main_window_close)  # Handle main window close event

# --- Settings Row Frame - Input Directories, Training Parameters, Early Stopping ---
settings_row_frame = ttk.Frame(window)
settings_row_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=10, pady=10)

# Input Directories Frame (in settings_row_frame)
input_frame = ttk.LabelFrame(settings_row_frame, text="Input Directories", padding=(10, 5))
input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)  # In row 0, column 0 of settings_row_frame

tk.Label(input_frame, text="Image Directory:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
image_dir_entry = tk.Entry(input_frame, width=50)
image_dir_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
image_dir_button = tk.Button(input_frame, text="Browse", command=browse_image_dir)
image_dir_button.grid(row=0, column=2, padx=5, pady=5)

tk.Label(input_frame, text="Mask Directory:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
mask_dir_entry = tk.Entry(input_frame, width=50)
mask_dir_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
mask_dir_button = tk.Button(input_frame, text="Browse", command=browse_mask_dir)
mask_dir_button.grid(row=1, column=2, padx=5, pady=5)

tk.Label(input_frame, text="Save Model Directory:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
save_model_dir_entry = tk.Entry(input_frame, width=50)
save_model_dir_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
save_model_dir_button = tk.Button(input_frame, text="Browse", command=browse_save_model_dir)
save_model_dir_button.grid(row=2, column=2, padx=5, pady=5)

# New Pre-trained Model Path Row
tk.Label(input_frame, text="Pre-trained Model Path:").grid(row=3, column=0, sticky="w", padx=5, pady=5)  # New Label
pretrained_model_path_entry = tk.Entry(input_frame, width=50)  # New Entry
pretrained_model_path_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)  # New Entry Grid
pretrained_model_path_button = tk.Button(input_frame, text="Browse", command=browse_pretrained_model_path)  # New Browse Button
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
early_stopping_check = tk.Checkbutton(early_stopping_frame, text="Enable Early Stopping", variable=early_stopping_check_var, command=toggle_early_stopping)
early_stopping_check.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
early_stopping_check.select()  # Ensure checkbox is selected on start

tk.Label(early_stopping_frame, text="Patience:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
patience_entry = tk.Entry(early_stopping_frame, width=5)
patience_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
patience_entry.insert(0, "5")  # Default patience
patience_entry.bind("<FocusOut>", lambda event: change_patience(patience_entry.get()))  # Update on focus out

tk.Label(early_stopping_frame, text="Min Delta:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
min_delta_entry = tk.Entry(early_stopping_frame, width=5)
min_delta_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
min_delta_entry.insert(0, "0.001")  # Default min delta
min_delta_entry.bind("<FocusOut>", lambda event: change_min_delta(min_delta_entry.get()))  # Update on focus out

# --- Removed early_stopping_status_label and variable ---


# --- Progress, Early Stopping, Training Time Frames in one row ---
status_row_frame = ttk.Frame(window)  # Frame to hold progress, ES, time frames
status_row_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=10, pady=10)  # Placed below settings_row_frame

# Progress Frame (in status_row_frame)
progress_frame = ttk.LabelFrame(status_row_frame, text="Training Progress", padding=(10, 5))
progress_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)  # In row 0, column 0 of status_row_frame

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

progress_frame.columnconfigure(0, weight=1)  # Make column 0 (left side) expandable
progress_frame.columnconfigure(1, weight=1)  # Make column 1 (right side) expandable


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

start_button = tk.Button(buttons_frame, text="Start Training", command=start_new_training_process)  # Modified to call start_new_training_process
start_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
continue_button = tk.Button(buttons_frame, text="Continue Training", command=continue_training_command, state=tk.DISABLED)  # Continue Training button
continue_button.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
pause_button = tk.Button(buttons_frame, text="Pause", command=pause_training, state=tk.DISABLED)
pause_button.grid(row=0, column=2, sticky="ew", padx=5, pady=5)  # Corrected line - added pady=5 and closing parenthesis
resume_button = tk.Button(buttons_frame, text="Resume", command=resume_training, state=tk.DISABLED)
resume_button.grid(row=0, column=3, sticky="ew", padx=5, pady=5)  # Now in column 3
stop_button = tk.Button(buttons_frame, text="Stop", command=stop_training, state=tk.DISABLED)
stop_button.grid(row=0, column=4, sticky="ew", padx=5, pady=5)
save_plot_button = tk.Button(buttons_frame, text="Save Plot", command=save_plot_image)
save_plot_button.grid(row=0, column=5, sticky="ew", padx=5, pady=5)  # Now in column 5
export_plot_data_button = tk.Button(buttons_frame, text="Export Plot Data", command=export_plot_data)  # Export Plot Data Button
export_plot_data_button.grid(row=0, column=6, sticky="ew", padx=5, pady=5)  # Now in column 6, next to Save Plot


# --- Visualize Results Button Frame - Moved above the Plot Frame, below Buttons ---
visualize_button_frame = ttk.Frame(window, padding=(10, 10))
visualize_button_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=10, pady=10)  # Visualize button above Plot frame, below Buttons frame, row number adjusted
visualize_button_frame.columnconfigure(0, weight=1)

visualize_results_button = tk.Button(visualize_button_frame, text="Visualize Results", command=visualize_results_command)
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

window.mainloop()
