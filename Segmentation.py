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
    """
    original_mode = image.mode
    mask = mask.convert('L')
    image_np = np.array(image).astype(np.float32) / 255.0
    mask_np = np.array(mask).astype(np.float32) / 255.0
    
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
            transparent_image_np = np.zeros((image_np.shape[0], image_np.shape[1], 2), dtype=np.float32)
            
            for y in range(image_np.shape[0]):
                for x in range(image_np.shape[1]):
                    if mask_np[y, x] > 0.5:  # ROI - make transparent
                        transparent_image_np[y, x, 0] = 0.0  # Black
                        transparent_image_np[y, x, 1] = 0.0  # Fully transparent
                    else:  # Background - keep visible
                        transparent_image_np[y, x, 0] = image_np[y, x]  # Original grayscale value
                        transparent_image_np[y, x, 1] = 1.0  # Fully opaque
    
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
        input_folder_entry.insert(0, dirpath) # Corrected line - was filepath

def select_output_folder_command():
    global output_folder_entry
    dirpath = filedialog.askdirectory(initialdir=".", title="Select Output Folder")
    if dirpath:
        output_folder_entry.delete(0, tk.END)
        output_folder_entry.insert(0, dirpath)

def set_output_file_type(file_type):
    global output_file_type
    output_file_type = file_type

def start_inference_command():
    global model_path_entry, input_folder_entry, output_folder_entry, status_text, output_file_type_combobox

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

    status_text.config(state=tk.NORMAL)
    status_text.delete(1.0, tk.END)
    status_text.config(state=tk.DISABLED)

    run_inference_threaded(model_path, input_folder, output_folder, selected_file_type)


def stop_inference_command():
    global inference_running
    inference_running = False

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


# --- Main GUI Setup ---
window = tk.Tk()
window.title("UNet TIFF Inference GUI")

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

window.mainloop()
