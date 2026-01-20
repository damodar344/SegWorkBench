import os
import sys
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- Optional imports guarded ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
except Exception:
    torch = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    plt = None

try:
    from PIL import Image, ImageTk, ImageFilter
except Exception:
    Image = None
    ImageTk = None

# Optional for Tab 3
SKLEARN_OK = True
SKIMAGE_OK = True
try:
    from sklearn.mixture import GaussianMixture
except Exception:
    SKLEARN_OK = False
try:
    from skimage import filters
except Exception:
    SKIMAGE_OK = False

APP_TITLE = "Segmentation Suite — 4 Tabs (Training • Inference • GMM • Mask Editing)"


# -----------------------------
# Shared UNet implementation
# -----------------------------
if torch is not None:
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.double_conv(x)

    class Down(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels),
            )

        def forward(self, x):
            return self.maxpool_conv(x)

    class Up(nn.Module):
        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class OutConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, n_channels=3, n_classes=1, bilinear=True):
            super().__init__()
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


# -----------------------------
# Utilities (thread-safe dialogs)
# -----------------------------
def info_dialog(title, msg):
    try:
        messagebox.showinfo(title, msg)
    except Exception:
        print(f"[{title}] {msg}")


def err_dialog(title, msg):
    try:
        messagebox.showerror(title, msg)
    except Exception:
        print(f"[{title}] {msg}")


# =============================
# Tab 1: Training (responsive)
# =============================
class TrainingTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        if torch is None or plt is None or Image is None or np is None:
            ttk.Label(self, text="Required packages missing (torch, matplotlib, PIL, numpy).").pack(padx=20, pady=20)
            return

        # State
        self.training_thread = None
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_dir = tk.StringVar()
        self.mask_dir = tk.StringVar()
        self.save_dir = tk.StringVar()

        self.epochs = tk.IntVar(value=5)
        self.batch_size = tk.IntVar(value=2)
        self.lr = tk.DoubleVar(value=1e-4)
        self.status = tk.StringVar(value="Idle")

        # Top controls
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="Images:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.image_dir, width=45).grid(row=0, column=1, sticky="we", padx=5)
        ttk.Button(top, text="Browse", command=self.browse_images).grid(row=0, column=2, padx=5)

        ttk.Label(top, text="Masks:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.mask_dir, width=45).grid(row=1, column=1, sticky="we", padx=5)
        ttk.Button(top, text="Browse", command=self.browse_masks).grid(row=1, column=2, padx=5)

        ttk.Label(top, text="Save models to:").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.save_dir, width=45).grid(row=2, column=1, sticky="we", padx=5)
        ttk.Button(top, text="Browse", command=self.browse_save).grid(row=2, column=2, padx=5)

        top.columnconfigure(1, weight=1)

        grid2 = ttk.Frame(self)
        grid2.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(grid2, text="Epochs:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(grid2, from_=1, to=1000, textvariable=self.epochs, width=6).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(grid2, text="Batch Size:").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(grid2, from_=1, to=32, textvariable=self.batch_size, width=6).grid(row=0, column=3, sticky="w", padx=5)
        ttk.Label(grid2, text="Learning Rate:").grid(row=0, column=4, sticky="w")
        ttk.Entry(grid2, textvariable=self.lr, width=10).grid(row=0, column=5, sticky="w", padx=5)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(btns, text="Start Training", command=self.start_training).pack(side="left")
        ttk.Button(btns, text="Pause", command=self.pause_training).pack(side="left", padx=5)
        ttk.Button(btns, text="Resume", command=self.resume_training).pack(side="left", padx=5)
        ttk.Button(btns, text="Stop", command=self.stop_training).pack(side="left", padx=5)

        ttk.Label(self, textvariable=self.status).pack(anchor="w", padx=12)

        # Plot
        fig = plt.figure(figsize=(5, 3))
        self.ax1 = fig.add_subplot(111)
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("Loss")
        self.line_loss, = self.ax1.plot([], [], label="Loss")
        self.ax1.legend(loc="upper right")
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self.losses = []

    # --- thread-safe UI helper ---
    def ui(self, fn, *args, **kwargs):
        self.after(0, lambda: fn(*args, **kwargs))

    def browse_images(self):
        d = filedialog.askdirectory(title="Select Training Images Folder")
        if d:
            self.image_dir.set(d)

    def browse_masks(self):
        d = filedialog.askdirectory(title="Select Training Masks Folder")
        if d:
            self.mask_dir.set(d)

    def browse_save(self):
        d = filedialog.askdirectory(title="Select Folder to Save Models")
        if d:
            self.save_dir.set(d)

    def start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            err_dialog("Training", "Training already running.")
            return
        if not all([self.image_dir.get(), self.mask_dir.get(), self.save_dir.get()]):
            err_dialog("Training", "Please select image folder, mask folder, and save folder.")
            return

        self.stop_flag.clear()
        self.pause_flag.clear()
        self.losses = []
        self.status.set("Training...")

        self.training_thread = threading.Thread(target=self._train_loop, daemon=True)
        self.training_thread.start()

    def pause_training(self):
        self.pause_flag.set()
        self.status.set("Paused.")

    def resume_training(self):
        self.pause_flag.clear()
        self.status.set("Resumed.")

    def stop_training(self):
        self.stop_flag.set()
        self.status.set("Stopping...")

    def _make_dataset(self, img_dir, msk_dir, size=256):
        class SegDataset(Dataset):
            def __init__(self, image_dir, mask_dir, size=256):
                self.image_dir = image_dir
                self.mask_dir = mask_dir
                self.files = self._match_files()
                self.img_t = T.Compose([T.Resize(size), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                self.msk_t = T.Compose([T.Resize(size), T.ToTensor()])

            def _match_files(self):
                imgs = {
                    os.path.splitext(f)[0].lower(): f
                    for f in os.listdir(self.image_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
                }
                msks = [f for f in os.listdir(self.mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
                pairs = []
                for m in msks:
                    b = os.path.splitext(m)[0].lower()
                    if b in imgs:
                        pairs.append((imgs[b], m))
                return pairs

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                imgf, mskf = self.files[idx]
                img = Image.open(os.path.join(self.image_dir, imgf)).convert("RGB")
                msk = Image.open(os.path.join(self.mask_dir, mskf)).convert("L")
                return self.img_t(img), self.msk_t(msk)

        return SegDataset(img_dir, msk_dir, size)

    def _train_loop(self):
        try:
            dataset = self._make_dataset(self.image_dir.get(), self.mask_dir.get(), 256)
            if len(dataset) == 0:
                self.ui(err_dialog, "Training", "No matched image-mask pairs found.")
                self.ui(self.status.set, "Idle")
                return

            loader = DataLoader(dataset, batch_size=self.batch_size.get(), shuffle=True, num_workers=0)
            model = UNet(n_channels=3, n_classes=1).to(self.device)
            opt = torch.optim.AdamW(model.parameters(), lr=float(self.lr.get()))
            criterion = nn.BCEWithLogitsLoss()

            total_epochs = int(self.epochs.get())
            for epoch in range(1, total_epochs + 1):
                if self.stop_flag.is_set():
                    break

                model.train()
                ep_loss = 0.0
                nb = 0

                for imgs, msks in loader:
                    while self.pause_flag.is_set() and not self.stop_flag.is_set():
                        time.sleep(0.1)
                    if self.stop_flag.is_set():
                        break

                    imgs = imgs.to(self.device)
                    msks = msks.to(self.device)

                    opt.zero_grad()
                    out = model(imgs)
                    loss = criterion(out, msks)
                    loss.backward()
                    opt.step()

                    ep_loss += float(loss.item())
                    nb += 1

                if nb > 0:
                    self.losses.append(ep_loss / nb)

                # Update plot + status from main thread
                def _update_ui(epoch=epoch, total_epochs=total_epochs):
                    xs = list(range(1, len(self.losses) + 1))
                    self.line_loss.set_data(xs, self.losses)
                    self.ax1.relim()
                    self.ax1.autoscale_view()
                    self.canvas.draw_idle()
                    last = self.losses[-1] if self.losses else 0.0
                    self.status.set(f"Epoch {epoch}/{total_epochs} | Loss: {last:.4f}")

                self.ui(_update_ui)

            # Save if not stopped
            if not self.stop_flag.is_set():
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                outp = os.path.join(self.save_dir.get(), f"{ts}_unet_final.pth")
                torch.save(model.state_dict(), outp)
                self.ui(info_dialog, "Training", f"Training complete. Model saved:\n{outp}")

        except Exception as e:
            self.ui(err_dialog, "Training Error", str(e))
        finally:
            self.ui(self.status.set, "Idle")


# =============================
# Tab 2: Inference (responsive)
# =============================
class InferenceTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        if torch is None or Image is None or np is None:
            ttk.Label(self, text="Required packages missing (torch, PIL, numpy).").pack(padx=20, pady=20)
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = tk.StringVar()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.status = tk.StringVar(value="Idle")
        self.running = False
        self.worker = None
        self.out_ext = tk.StringVar(value=".png")

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)
        ttk.Label(top, text="Model (.pth):").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.model_path, width=45).grid(row=0, column=1, sticky="we", padx=5)
        ttk.Button(top, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5)

        ttk.Label(top, text="Input Folder:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.input_dir, width=45).grid(row=1, column=1, sticky="we", padx=5)
        ttk.Button(top, text="Browse", command=self.browse_input).grid(row=1, column=2, padx=5)

        ttk.Label(top, text="Output Folder:").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.output_dir, width=45).grid(row=2, column=1, sticky="we", padx=5)
        ttk.Button(top, text="Browse", command=self.browse_output).grid(row=2, column=2, padx=5)

        top.columnconfigure(1, weight=1)

        opts = ttk.Frame(self)
        opts.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(opts, text="Save as:").pack(side="left")
        ttk.Combobox(opts, textvariable=self.out_ext, values=[".png", ".tif"], width=6, state="readonly").pack(side="left", padx=5)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(btns, text="Start Inference", command=self.start).pack(side="left")
        ttk.Button(btns, text="Stop", command=self.stop).pack(side="left", padx=5)
        ttk.Label(self, textvariable=self.status).pack(anchor="w", padx=12)

    def ui(self, fn, *args, **kwargs):
        self.after(0, lambda: fn(*args, **kwargs))

    def browse_model(self):
        p = filedialog.askopenfilename(title="Select Model", filetypes=[("PyTorch", "*.pth"), ("All files", "*.*")])
        if p:
            self.model_path.set(p)

    def browse_input(self):
        d = filedialog.askdirectory(title="Select Input Folder")
        if d:
            self.input_dir.set(d)

    def browse_output(self):
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self.output_dir.set(d)

    def start(self):
        if self.running:
            err_dialog("Inference", "Inference already running.")
            return
        if not os.path.isfile(self.model_path.get()):
            err_dialog("Inference", "Select a valid .pth model file.")
            return
        if not os.path.isdir(self.input_dir.get()):
            err_dialog("Inference", "Select a valid input folder of images.")
            return

        outd = self.output_dir.get() or self.input_dir.get()
        os.makedirs(outd, exist_ok=True)

        self.running = True
        self.status.set("Loading model...")
        self.worker = threading.Thread(target=self._inference_loop, args=(outd,), daemon=True)
        self.worker.start()

    def stop(self):
        self.running = False
        self.status.set("Stopping...")

    def _crop_to_content(self, image_pil, mask_pil):
        mask = mask_pil.convert("L").filter(ImageFilter.MedianFilter(3))
        arr = np.array(mask)
        ys, xs = np.where(arr >= 128)
        if ys.size == 0 or xs.size == 0:
            return image_pil, mask_pil

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        pad = 2

        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(image_pil.width - 1, x1 + pad)
        y1 = min(image_pil.height - 1, y1 + pad)

        box = (x0, y0, x1 + 1, y1 + 1)
        return image_pil.crop(box), mask_pil.crop(box)

    def _inference_loop(self, outd):
        try:
            model = UNet(n_channels=3, n_classes=1).to(self.device)
            sd = torch.load(self.model_path.get(), map_location=self.device)
            model.load_state_dict(sd)
            model.eval()

            tfm = T.Compose([T.Resize(256), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

            exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            files = [f for f in os.listdir(self.input_dir.get()) if f.lower().endswith(exts)]
            if not files:
                self.ui(err_dialog, "Inference", "No images found in input folder.")
                self.ui(self.status.set, "Idle")
                self.running = False
                return

            start = time.time()
            for i, fname in enumerate(files, 1):
                if not self.running:
                    break

                fpath = os.path.join(self.input_dir.get(), fname)
                img = Image.open(fpath)
                rgb = img.convert("RGB")

                ten = tfm(rgb).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = model(ten)
                    prob = torch.sigmoid(out).cpu()[0, 0].numpy()

                mask_arr = (prob > 0.5).astype("uint8") * 255
                mask_pil = Image.fromarray(mask_arr, mode="L")

                cropped_img, cropped_mask = self._crop_to_content(img, mask_pil)

                base = os.path.splitext(fname)[0]
                if self.out_ext.get().lower() == ".png":
                    rgba = cropped_img.convert("RGBA")
                    a = np.array(cropped_mask) / 255.0
                    rgb_arr = np.array(rgba, dtype=np.float32)
                    rgb_arr[..., 3] = (a * 255.0).astype(np.uint8)
                    out_img = Image.fromarray(rgb_arr.astype(np.uint8), "RGBA")
                    out_img.save(os.path.join(outd, f"segmented_{base}.png"))
                else:
                    cropped_mask.save(os.path.join(outd, f"mask_{base}.tif"))
                    cropped_img.save(os.path.join(outd, f"cropped_{base}.tif"))

                self.ui(self.status.set, f"Processed {i}/{len(files)}: {fname}")

            dur = time.time() - start
            self.ui(self.status.set, f"Done in {dur:.1f}s")
            self.ui(info_dialog, "Inference", f"Inference finished. Output in:\n{outd}")

        except Exception as e:
            self.ui(err_dialog, "Inference Error", str(e))
        finally:
            self.running = False


# =============================
# Tab 3: GMM (fixed — no recursion, responsive)
# =============================
class GMMTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        if not (SKLEARN_OK and SKIMAGE_OK and plt is not None and Image is not None and np is not None):
            txt = "This tab requires scikit-learn, scikit-image, matplotlib, PIL, numpy."
            ttk.Label(self, text=txt, wraplength=520).pack(padx=20, pady=20)
            return

        self.img = None
        self.img_arr = None
        self.non_black = None
        self.gmm = None

        self.path = tk.StringVar()
        self.n_components = tk.IntVar(value=2)
        self.threshold_method = tk.StringVar(value="GMM")
        self.threshold_value = tk.DoubleVar(value=0.0)

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)
        ttk.Entry(top, textvariable=self.path, width=48).pack(side="left", fill="x", expand=True)
        ttk.Button(top, text="Open Image", command=self.open_image).pack(side="left", padx=6)

        opts = ttk.Frame(self)
        opts.pack(fill="x", padx=10, pady=5)
        ttk.Label(opts, text="Gaussians:").pack(side="left")
        ttk.Spinbox(opts, from_=1, to=9, textvariable=self.n_components, width=5,
                    command=self.refit_gmm).pack(side="left", padx=4)

        for m in ["GMM", "Otsu", "Triangle", "IsoData"]:
            ttk.Radiobutton(opts, text=m, value=m, variable=self.threshold_method,
                            command=self.compute_threshold).pack(side="left", padx=4)

        valf = ttk.Frame(self)
        valf.pack(fill="x", padx=10, pady=5)
        ttk.Label(valf, text="Threshold:").pack(side="left")
        self.scale = tk.Scale(
            valf, from_=0, to=255, resolution=1, orient="horizontal",
            variable=self.threshold_value, command=lambda *_: self.update_threshold_line()
        )
        self.scale.pack(side="left", fill="x", expand=True, padx=5)
        self.th_label = ttk.Label(valf, text="0.0")
        self.th_label.pack(side="left", padx=5)

        self.fig_img, self.ax_img = plt.subplots(figsize=(3, 3))
        self.ax_img.axis("off")
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=self)
        self.canvas_img.get_tk_widget().pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.fig_hist, self.ax_hist = plt.subplots(figsize=(3, 3))
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=self)
        self.canvas_hist.get_tk_widget().pack(side="left", fill="both", expand=True, padx=10, pady=10)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=5)
        ttk.Button(btns, text="Apply Segmentation", command=self.apply_segmentation).pack(side="left")

    def _show_image(self, img_pil):
        self.ax_img.clear()
        self.ax_img.imshow(np.array(img_pil), cmap="gray")
        self.ax_img.axis("off")
        self.canvas_img.draw()

    def _plot_histogram_and_gmm(self):
        if self.non_black is None or self.non_black.size == 0:
            return
        self.ax_hist.clear()
        bins = 256
        self.ax_hist.hist(self.non_black, bins=bins, color="black", edgecolor="black", alpha=0.6)

        if self.gmm is not None:
            xs = np.linspace(self.non_black.min(), self.non_black.max(), 800).reshape(-1, 1)
            lp = self.gmm.score_samples(xs)
            pdf = np.exp(lp)
            probs = self.gmm.predict_proba(xs)

            hist_max = np.max(np.histogram(self.non_black, bins=bins)[0])
            for i in range(self.gmm.n_components):
                comp_pdf = probs[:, i] * pdf
                if comp_pdf.max() > 0:
                    self.ax_hist.plot(xs, (comp_pdf / comp_pdf.max()) * hist_max, linewidth=2)

        self.ax_hist.set_xlabel("Pixel Intensity")
        self.ax_hist.set_ylabel("Frequency")
        self.canvas_hist.draw()

    def open_image(self):
        p = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.gif"), ("All", "*.*")]
        )
        if not p:
            return
        self.path.set(p)

        img = Image.open(p)
        if img.mode not in ["L", "I", "F"]:
            img = img.convert("L")
        self.img = img
        self.img_arr = np.array(self.img)

        flat = self.img_arr.flatten()
        self.non_black = flat[flat > 0]
        self._show_image(self.img)

        mn = float(self.non_black.min()) if self.non_black.size else 0.0
        mx = float(self.non_black.max()) if self.non_black.size else 255.0
        self.scale.config(from_=mn, to=mx, resolution=1)

        self.refit_gmm()
        self.compute_threshold()

    def refit_gmm(self):
        if self.non_black is None or self.non_black.size == 0:
            return
        try:
            k = int(self.n_components.get())
            self.gmm = GaussianMixture(n_components=k, random_state=42)
            self.gmm.fit(self.non_black.reshape(-1, 1))
        except Exception as e:
            err_dialog("GMM", f"Failed to fit GMM: {e}")
            self.gmm = None

        self._plot_histogram_and_gmm()
        self.update_threshold_line()

    def compute_threshold(self):
        if self.non_black is None or self.non_black.size == 0:
            return

        method = self.threshold_method.get()
        th = float(self.threshold_value.get())

        try:
            if method == "GMM" and self.gmm is not None and self.gmm.n_components >= 2:
                means = self.gmm.means_.flatten()
                covs = self.gmm.covariances_.reshape(-1)
                weights = self.gmm.weights_.flatten()

                order = np.argsort(means)
                i1, i2 = int(order[0]), int(order[1])
                m1, m2 = float(means[i1]), float(means[i2])
                s1, s2 = float(np.sqrt(covs[i1])), float(np.sqrt(covs[i2]))
                w1, w2 = float(weights[i1]), float(weights[i2])

                x = np.linspace(min(m1, m2), max(m1, m2), 5000)
                pdf1 = w1 * np.exp(-0.5 * ((x - m1) / s1) ** 2) / (s1 * np.sqrt(2 * np.pi))
                pdf2 = w2 * np.exp(-0.5 * ((x - m2) / s2) ** 2) / (s2 * np.sqrt(2 * np.pi))
                idx = int(np.argmin(np.abs(pdf1 - pdf2)))
                th = float(x[idx])

            elif method == "Otsu":
                th = float(filters.threshold_otsu(self.non_black))
            elif method == "Triangle":
                th = float(filters.threshold_triangle(self.non_black))
            elif method == "IsoData":
                th = float(filters.threshold_isodata(self.non_black))

        except Exception as e:
            err_dialog("Threshold", f"Failed to compute threshold: {e}")

        self.threshold_value.set(th)
        self.update_threshold_line()

    def update_threshold_line(self):
        if self.non_black is None or self.non_black.size == 0:
            return
        th = float(self.threshold_value.get())
        self.th_label.config(text=f"{th:.1f}")

        self._plot_histogram_and_gmm()
        self.ax_hist.axvline(th, color="green", linestyle="--")
        self.canvas_hist.draw()

    def apply_segmentation(self):
        if self.img_arr is None:
            err_dialog("GMM", "Load an image first.")
            return
        th = float(self.threshold_value.get())
        mask = (self.img_arr > th).astype(np.uint8) * 255
        vis = Image.fromarray(mask, mode="L")
        self._show_image(vis)


# =============================
# Tab 4: Mask Editing
# =======================
class MaskEditorTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        if Image is None or ImageTk is None or np is None:
            ttk.Label(self, text="Required packages missing (PIL, ImageTk, numpy).").pack(padx=20, pady=20)
            return

        self.img_pil = None
        self.disp_img = None
        self.mask_arr = None
        self.disp_mask = None
        self.tk_overlay = None

        self.img_path = tk.StringVar()
        self.mask_path = tk.StringVar()
        self.out_dir = tk.StringVar()

        self.tool = tk.StringVar(value="brush")
        self.brush_size = tk.IntVar(value=15)
        self.opacity = tk.DoubleVar(value=0.35)
        self.status = tk.StringVar(value="Idle")

        self.undo_stack = []
        self.max_undo = 20

        self.max_w = 900
        self.max_h = 600

        # NEW: canvas offsets (for centering)
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        r1 = ttk.Frame(top); r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Image:").pack(side="left")
        ttk.Entry(r1, textvariable=self.img_path, width=60).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r1, text="Browse", command=self.pick_image).pack(side="left", padx=2)
        ttk.Button(r1, text="Load", command=self.load_image).pack(side="left", padx=2)

        r2 = ttk.Frame(top); r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Mask:").pack(side="left")
        ttk.Entry(r2, textvariable=self.mask_path, width=60).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r2, text="Browse", command=self.pick_mask).pack(side="left", padx=2)
        ttk.Button(r2, text="Load", command=self.load_mask).pack(side="left", padx=2)
        ttk.Button(r2, text="New Blank Mask", command=self.new_blank_mask).pack(side="left", padx=6)

        r3 = ttk.Frame(top); r3.pack(fill="x", pady=2)
        ttk.Label(r3, text="Output Folder:").pack(side="left")
        ttk.Entry(r3, textvariable=self.out_dir, width=60).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r3, text="Browse", command=self.pick_out).pack(side="left", padx=2)
        ttk.Button(r3, text="Save Mask PNG", command=self.save_mask_png).pack(side="left", padx=6)
        ttk.Button(r3, text="Save Transparent PNG", command=self.save_transparent_png).pack(side="left", padx=2)

        tools = ttk.Frame(self)
        tools.pack(fill="x", padx=10, pady=(0, 8))

        ttk.Label(tools, text="Tool:").pack(side="left")
        ttk.Radiobutton(tools, text="Brush", value="brush", variable=self.tool).pack(side="left", padx=5)
        ttk.Radiobutton(tools, text="Eraser", value="eraser", variable=self.tool).pack(side="left", padx=5)

        ttk.Label(tools, text="Brush size:").pack(side="left", padx=(15, 5))
        ttk.Scale(tools, from_=1, to=80, orient="horizontal", variable=self.brush_size).pack(side="left", fill="x", expand=True)

        ttk.Label(tools, text="Overlay opacity:").pack(side="left", padx=(15, 5))
        ttk.Scale(tools, from_=0.0, to=0.9, orient="horizontal", variable=self.opacity,
                  command=lambda *_: self.render()).pack(side="left", fill="x", expand=True)

        ttk.Button(tools, text="Undo", command=self.undo).pack(side="left", padx=8)
        ttk.Button(tools, text="Clear Mask", command=self.clear_mask).pack(side="left", padx=2)

        ttk.Label(self, textvariable=self.status).pack(anchor="w", padx=12)

        self.canvas = tk.Canvas(self, bg="#222222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", lambda _e: self.render())  # re-center on resize

        self._is_drawing = False
        self._last_xy = None

    def pick_image(self):
        p = filedialog.askopenfilename(
            title="Pick image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.gif"), ("All", "*.*")]
        )
        if p:
            self.img_path.set(p)

    def pick_mask(self):
        p = filedialog.askopenfilename(
            title="Pick mask (binary/grayscale)",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.gif"), ("All", "*.*")]
        )
        if p:
            self.mask_path.set(p)

    def pick_out(self):
        d = filedialog.askdirectory(title="Pick output folder")
        if d:
            self.out_dir.set(d)

    def load_image(self):
        p = self.img_path.get().strip()
        if not p or not os.path.isfile(p):
            err_dialog("Mask Editor", "Please select a valid image file.")
            return

        self.img_pil = Image.open(p).convert("RGB")
        self.status.set(f"Loaded image: {os.path.basename(p)}")

        if self.mask_arr is not None and self.mask_arr.shape[:2] != (self.img_pil.height, self.img_pil.width):
            self.mask_arr = None
            self.undo_stack.clear()

        if self.mask_arr is None:
            self.new_blank_mask(push_undo=False)
        else:
            self._prepare_display_assets(mask_only=False)

        self.render()

    def load_mask(self):
        if self.img_pil is None:
            err_dialog("Mask Editor", "Load an image first (mask needs matching size).")
            return

        p = self.mask_path.get().strip()
        if not p or not os.path.isfile(p):
            err_dialog("Mask Editor", "Please select a valid mask file.")
            return

        m = Image.open(p).convert("L")
        if (m.width, m.height) != (self.img_pil.width, self.img_pil.height):
            m = m.resize((self.img_pil.width, self.img_pil.height), Image.NEAREST)

        arr = np.array(m, dtype=np.uint8)
        arr = (arr >= 128).astype(np.uint8) * 255

        self._push_undo()
        self.mask_arr = arr
        self.status.set(f"Loaded mask: {os.path.basename(p)}")

        self._prepare_display_assets(mask_only=False)
        self.render()

    def new_blank_mask(self, push_undo=True):
        if self.img_pil is None:
            self.status.set("Load an image to create a blank mask.")
            return
        if push_undo:
            self._push_undo()
        self.mask_arr = np.zeros((self.img_pil.height, self.img_pil.width), dtype=np.uint8)
        self.status.set("Created new blank mask.")
        self._prepare_display_assets(mask_only=False)
        self.render()

    def clear_mask(self):
        if self.mask_arr is None:
            return
        self._push_undo()
        self.mask_arr[:, :] = 0
        self.status.set("Mask cleared.")
        self._prepare_display_assets(mask_only=True)
        self.render()

    def _push_undo(self):
        if self.mask_arr is None:
            return
        self.undo_stack.append(self.mask_arr.copy())
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            self.status.set("Undo stack empty.")
            return
        self.mask_arr = self.undo_stack.pop()
        self.status.set("Undo applied.")
        self._prepare_display_assets(mask_only=True)
        self.render()

    def on_mouse_down(self, event):
        if self.img_pil is None or self.mask_arr is None or self.disp_img is None:
            return
        self._is_drawing = True
        self._push_undo()
        self._last_xy = (event.x, event.y)
        self._apply_brush(event.x, event.y)
        self.render()

    def on_mouse_drag(self, event):
        if not self._is_drawing:
            return
        self._apply_brush(event.x, event.y, from_xy=self._last_xy)
        self._last_xy = (event.x, event.y)
        self.render()

    def on_mouse_up(self, _event):
        self._is_drawing = False
        self._last_xy = None

    def _apply_brush(self, x, y, from_xy=None):
        if self.disp_img is None:
            return

        # Convert canvas coords -> displayed image coords by subtracting offsets
        x_img = x - self.canvas_offset_x
        y_img = y - self.canvas_offset_y

        # Ignore drawing outside the displayed image
        if x_img < 0 or y_img < 0 or x_img >= self.disp_img.width or y_img >= self.disp_img.height:
            return

        sx = self.img_pil.width / self.disp_img.width
        sy = self.img_pil.height / self.disp_img.height
        ox = int(x_img * sx)
        oy = int(y_img * sy)

        r_disp = int(self.brush_size.get())
        r_ox = max(1, int(r_disp * sx))
        val = 255 if self.tool.get() == "brush" else 0

        if from_xy is not None:
            fx, fy = from_xy
            fx_img = fx - self.canvas_offset_x
            fy_img = fy - self.canvas_offset_y

            if fx_img < 0 or fy_img < 0 or fx_img >= self.disp_img.width or fy_img >= self.disp_img.height:
                return

            fox = int(fx_img * sx)
            foy = int(fy_img * sy)
            self._draw_line_circles(fox, foy, ox, oy, r_ox, val)
        else:
            self._draw_circle(ox, oy, r_ox, val)

        self._prepare_display_assets(mask_only=True)

    def _draw_circle(self, cx, cy, r, val):
        h, w = self.mask_arr.shape
        x0 = max(0, cx - r); x1 = min(w - 1, cx + r)
        y0 = max(0, cy - r); y1 = min(h - 1, cy + r)
        yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        self.mask_arr[y0:y1 + 1, x0:x1 + 1][m] = val

    def _draw_line_circles(self, x0, y0, x1, y1, r, val):
        dx = x1 - x0
        dy = y1 - y0
        steps = int(max(abs(dx), abs(dy), 1))
        for i in range(steps + 1):
            t = i / steps
            cx = int(x0 + dx * t)
            cy = int(y0 + dy * t)
            self._draw_circle(cx, cy, r, val)

    def _prepare_display_assets(self, mask_only=False):
        if self.img_pil is None or self.mask_arr is None:
            return

        if not mask_only or self.disp_img is None:
            w, h = self.img_pil.width, self.img_pil.height
            scale = min(self.max_w / max(w, 1), self.max_h / max(h, 1), 1.0)
            dw, dh = int(w * scale), int(h * scale)
            self.disp_img = self.img_pil.resize((dw, dh), Image.BILINEAR)

        dw, dh = self.disp_img.size
        m_pil = Image.fromarray(self.mask_arr, mode="L").resize((dw, dh), Image.NEAREST)
        self.disp_mask = np.array(m_pil, dtype=np.uint8)

    def render(self):
        if self.img_pil is None or self.mask_arr is None or self.disp_img is None:
            self.canvas.delete("all")
            return

        base = self.disp_img.copy().convert("RGBA")
        alpha = int(float(self.opacity.get()) * 255)

        m = self.disp_mask
        overlay = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
        overlay[..., 0] = 255
        overlay[..., 3] = ((m > 0).astype(np.uint8) * alpha)

        comp = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))
        self.tk_overlay = ImageTk.PhotoImage(comp)

        self.canvas.delete("all")

        # Center image on canvas
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.canvas_offset_x = max(0, (cw - comp.width) // 2)
        self.canvas_offset_y = max(0, (ch - comp.height) // 2)

        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y, anchor="nw", image=self.tk_overlay)

    def _get_out_dir(self):
        outd = self.out_dir.get().strip()
        if not outd:
            if self.img_path.get().strip():
                outd = os.path.dirname(self.img_path.get().strip())
            else:
                outd = os.getcwd()
        os.makedirs(outd, exist_ok=True)
        return outd

    def save_mask_png(self):
        if self.mask_arr is None:
            err_dialog("Mask Editor", "No mask to save.")
            return
        outd = self._get_out_dir()
        base = "edited_mask"
        if self.img_path.get().strip():
            base = os.path.splitext(os.path.basename(self.img_path.get().strip()))[0] + "_mask_edited"
        out_path = os.path.join(outd, f"{base}.png")
        Image.fromarray(self.mask_arr, mode="L").save(out_path)
        info_dialog("Mask Editor", f"Saved mask PNG:\n{out_path}")

    def save_transparent_png(self):
        if self.img_pil is None or self.mask_arr is None:
            err_dialog("Mask Editor", "Load an image and mask first.")
            return
        outd = self._get_out_dir()
        base = "transparent"
        if self.img_path.get().strip():
            base = os.path.splitext(os.path.basename(self.img_path.get().strip()))[0] + "_transparent"
        out_path = os.path.join(outd, f"{base}.png")

        rgba = self.img_pil.convert("RGBA")
        arr = np.array(rgba, dtype=np.uint8)
        arr[..., 3] = self.mask_arr
        Image.fromarray(arr, mode="RGBA").save(out_path)
        info_dialog("Mask Editor", f"Saved transparent PNG:\n{out_path}")



# =============================
# Main App
# =============================
def main():
    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("1200x750")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    nb.add(TrainingTab(nb), text="1) Training (UNet)")
    nb.add(InferenceTab(nb), text="2) Inference")
    nb.add(GMMTab(nb), text="3) GMM")
    nb.add(MaskEditorTab(nb), text="4) Mask Editing")

    root.mainloop()


if __name__ == "__main__":
    main()
