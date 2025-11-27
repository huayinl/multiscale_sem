import sys
import os
import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
import tempfile
import shutil
try:
    import h5py
    H5PY_AVAILABLE = True
except Exception:
    h5py = None
    H5PY_AVAILABLE = False
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QStackedWidget, QSlider, QMessageBox, QProgressBar, QComboBox, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from queue import Queue, Empty
from collections import OrderedDict

# --- Check for SAM 2 ---
try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM 2 library not found. Tracking will not function.")

# --- Logic from your script (Refactored) ---

def process_image_and_scale_centers(img, downsample_resolution=8, min_area=1000, max_area=50000, min_hole_area=100000, num_skeleton_points=10):
    # [Exactly as provided in your prompt]
    original_height, original_width = img.shape[:2]
    new_width = original_width // downsample_resolution
    new_height = original_height // downsample_resolution
    
    downsampled_img = cv2.resize(img, (new_width, new_height))
    
    if len(downsampled_img.shape) == 3:
        gray_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = downsampled_img

    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, initial_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    inv_mask = 255 - initial_mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv_mask, connectivity=8)
    filtered_mask = initial_mask.copy()
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            filtered_mask[labels == i] = 255
        elif area > max_area:
            filtered_mask[labels == i] = 255

    filled_mask = filtered_mask.copy()
    num_labels_white, labels_white, stats_white, centroids_white = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)
    
    for i in range(1, num_labels_white):
        if stats_white[i, cv2.CC_STAT_AREA] < min_hole_area:
            filled_mask[labels_white == i] = 0

    mask = filled_mask
    num_labels_blobs, labels_blobs, stats_blobs, centroids_blobs = cv2.connectedComponentsWithStats(255 - mask, connectivity=8)
    blob_centers = []
    
    for i in range(1, num_labels_blobs):
        area = stats_blobs[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            blob_mask = (labels_blobs == i).astype(np.uint8)
            skeleton = skeletonize(blob_mask)
            skeleton_points = np.where(skeleton)
            
            if len(skeleton_points[0]) > 0:
                all_points = [(skeleton_points[1][j], skeleton_points[0][j]) for j in range(len(skeleton_points[0]))]
                N = len(all_points)
                if N <= num_skeleton_points:
                    blob_skeleton_points = all_points
                else:
                    indices = np.linspace(0, N-1, num_skeleton_points, dtype=int)
                    blob_skeleton_points = [all_points[j] for j in indices]
                blob_centers.append(blob_skeleton_points)

    scaled_blob_centers = []
    for blob_points in blob_centers:
        scaled_points = [(int(cx * downsample_resolution), int(cy * downsample_resolution)) for cx, cy in blob_points]
        scaled_blob_centers.append(scaled_points)

    return scaled_blob_centers

# --- Worker Thread for Heavy SAM 2 Processing ---
class TrackerWorker(QThread):
    finished = pyqtSignal(str) # Emits the path to the mask folder
    progress = pyqtSignal(int) # Optional: to show progress if you added a bar
    error = pyqtSignal(str)

    def __init__(self, video_path, device, scaled_blob_centers, colors=None, model_size='base'):
        super().__init__()
        self.video_path = video_path
        self.device = device
        self.scaled_blob_centers = scaled_blob_centers
        self.model_size = model_size
        # Colors can be passed in; if not, fall back to random colors
        if colors is None:
            self.colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
        else:
            self.colors = colors # We need colors here now to bake them into the images

    def run(self):
        try:
            # 1. Setup Folders
            video_dir_name = os.path.basename(os.path.normpath(self.video_path))
            parent_dir = os.path.dirname(os.path.normpath(self.video_path))
            mask_output_dir = os.path.join(parent_dir, f"{video_dir_name}_masks")
            os.makedirs(mask_output_dir, exist_ok=True)

            # 2. Initialize Model based on selected model_size
            model_size = getattr(self, 'model_size', 'base')
            if model_size == 'tiny':
                sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
            elif model_size == 'base':
                sam2_checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            elif model_size == 'large':
                sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            else:
                self.error.emit(f"Unknown model size: {model_size}")
                return

            if not os.path.exists(sam2_checkpoint):
                self.error.emit(f"Checkpoint not found: {sam2_checkpoint}")
                return

            predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
            inference_state = predictor.init_state(video_path=self.video_path)
            
            # 3. Add Prompts
            for i, prompts in enumerate(self.scaled_blob_centers):
                ann_obj_id = i + 1
                points = np.array(prompts, dtype=np.float32)
                labels = np.ones(points.shape[0], dtype=np.int32)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )

            # 4. Propagate and Save to Disk Immediately
            print("Propagating and saving masks...")
            
            # Get image dimensions from the first frame to ensure mask matches
            first_frame_path = os.path.join(self.video_path, sorted(os.listdir(self.video_path))[0])
            ref_img = cv2.imread(first_frame_path)
            h, w = ref_img.shape[:2]

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                
                # Create a black canvas for the color mask
                color_mask = np.zeros((h, w, 3), dtype=np.uint8)

                for i, out_obj_id in enumerate(out_obj_ids):
                    # Get binary mask for this object
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    
                    if mask.sum() > 0:
                        # Resize if necessary (SAM output safety)
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        # Get color for this object ID
                        color = self.colors[out_obj_id % len(self.colors)]
                        
                        # Paint the color on the mask
                        color_mask[mask > 0] = color

                # Save the pre-colored mask to the new folder as a compressed JPG
                # Filename matches frame index: 0.jpg, 1.jpg, etc.
                save_path = os.path.join(mask_output_dir, f"{out_frame_idx}.jpg")
                # Use reasonable JPEG quality to keep files small but good-looking
                cv2.imwrite(save_path, color_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            
            self.finished.emit(mask_output_dir)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))# --- GUI Components ---


class ImageLoader(QThread):
    """Background loader that reads frames and (optionally) pre-colored masks,
    composes them (mask blending) and emits RGB numpy arrays to the GUI thread.
    The GUI thread converts to QPixmap and caches the scaled pixmap.
    """
    frameLoaded = pyqtSignal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._req_q = Queue()
        self._running = True
        # references (set by GUI) to lists/dicts owned by GUI
        self.image_files = None
        self.mask_files = None
        # optional HDF5 reference
        self.h5_path = None
        self.h5_dataset = None
        # internal HDF5 handle opened in this thread (open once for efficiency)
        self._h5f = None
        self._h5_n_frames = None

    def set_references(self, image_files=None, mask_files=None, h5_path=None, h5_dataset=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.h5_path = h5_path
        self.h5_dataset = h5_dataset
        # reset per-thread HDF5 info; actual file handle will be opened inside run()
        self._h5_n_frames = None
        # if image_files given and it's a list, we keep it; otherwise None

    def request(self, idx):
        try:
            self._req_q.put_nowait(idx)
        except Exception:
            try:
                self._req_q.put(idx)
            except Exception:
                pass

    def run(self):
        while self._running:
            try:
                idx = self._req_q.get(timeout=0.1)
            except Empty:
                continue
            if idx is None:
                continue
            try:
                # Allow either file-based image list or HDF5-backed dataset
                if self.image_files is None and self.h5_path is None:
                    # nothing to load
                    continue

                # If using HDF5, ensure we have dataset length and an open handle
                if self.h5_path is not None:
                    if not H5PY_AVAILABLE:
                        continue
                    # open HDF5 file once in this thread for efficiency
                    if self._h5f is None:
                        try:
                            self._h5f = h5py.File(self.h5_path, 'r')
                            ds_name = self.h5_dataset
                            if ds_name is None:
                                # pick the first dataset
                                ds_name = next(iter(self._h5f.keys()))
                            self._h5_dataset_name_internal = ds_name
                            self._h5_n_frames = self._h5f[ds_name].shape[0]
                        except Exception:
                            # failed to open HDF5 in loader thread
                            try:
                                if self._h5f is not None:
                                    self._h5f.close()
                                self._h5f = None
                            except Exception:
                                pass
                            continue

                    # bounds check using HDF5 length
                    if idx < 0 or (self._h5_n_frames is not None and idx >= self._h5_n_frames):
                        continue

                    try:
                        ds = self._h5f[self._h5_dataset_name_internal]
                        arr = np.asarray(ds[idx])
                        # if grayscale expand to 3 channels
                        if arr.ndim == 2:
                            img_rgb = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                        elif arr.ndim == 3 and arr.shape[2] == 3:
                            if arr.dtype != np.uint8:
                                # normalize to 0-255
                                arrf = arr.astype(np.float32)
                                m = arrf.max()
                                if m == 0:
                                    img_rgb = (arrf * 0).astype(np.uint8)
                                else:
                                    img_rgb = (255.0 * (arrf / m)).astype(np.uint8)
                            else:
                                img_rgb = arr.astype(np.uint8)
                        else:
                            # unsupported shape
                            continue
                    except Exception:
                        continue
                else:
                    # file-based images
                    if idx < 0 or idx >= len(self.image_files):
                        continue
                    path = self.image_files[idx]
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # compose mask if available
                if self.mask_files and idx in self.mask_files:
                    mask_path = self.mask_files[idx]
                    mask_bgr = cv2.imread(mask_path)
                    if mask_bgr is not None:
                        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
                        if mask_rgb.shape[:2] != img_rgb.shape[:2]:
                            mask_rgb = cv2.resize(mask_rgb, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask_indices = np.any(mask_rgb != [0, 0, 0], axis=-1)
                        # ensure copy before modifying
                        img_rgb = img_rgb.copy()
                        img_rgb[mask_indices] = cv2.addWeighted(img_rgb[mask_indices], 0.5, mask_rgb[mask_indices], 0.5, 0)

                # emit the composed RGB image (numpy array)
                self.frameLoaded.emit(idx, img_rgb)
            except Exception:
                # ignore per-frame errors
                continue

    def stop(self):
        self._running = False
        try:
            self._req_q.put(None)
        except Exception:
            pass
        # close any open HDF5 handle
        try:
            if getattr(self, '_h5f', None) is not None:
                try:
                    self._h5f.close()
                except Exception:
                    pass
                self._h5f = None
        except Exception:
            pass
        self.wait()


class ExportH5Worker(QThread):
    """Export frames from an HDF5 dataset to a temporary folder as JPEGs.
    Emits `finished(temp_folder)` when done, `progress(int)` updates, and `error(str)` on failure.
    """
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, h5_path, dataset_name=None):
        super().__init__()
        self.h5_path = h5_path
        self.dataset_name = dataset_name

    def run(self):
        if not H5PY_AVAILABLE:
            self.error.emit("h5py not available")
            return
        try:
            with h5py.File(self.h5_path, 'r') as f:
                ds_name = self.dataset_name
                if ds_name is None:
                    # pick preferred dataset names or first dataset
                    for pref in ('images', 'frames', 'data'):
                        if pref in f:
                            ds_name = pref
                            break
                if ds_name is None:
                    for k in f.keys():
                        if isinstance(f[k], h5py.Dataset):
                            ds_name = k
                            break
                if ds_name is None:
                    self.error.emit('No dataset found in HDF5')
                    return

                ds = f[ds_name]
                n_frames = ds.shape[0]
                temp_dir = tempfile.mkdtemp(prefix='h5_frames_')
                for i in range(n_frames):
                    try:
                        arr = np.asarray(ds[i])
                        if arr.ndim == 2:
                            out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                        elif arr.ndim == 3 and arr.shape[2] == 3:
                            if arr.dtype != np.uint8:
                                # normalize to 0-255
                                arrf = arr.astype(np.float32)
                                m = arrf.max()
                                if m == 0:
                                    out = (arrf * 0).astype(np.uint8)
                                else:
                                    out = (255.0 * (arrf / m)).astype(np.uint8)
                            else:
                                out = arr.astype(np.uint8)
                            # convert RGB->BGR if needed (assume RGB)
                            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                        else:
                            # unsupported dim
                            continue

                        save_path = os.path.join(temp_dir, f"{i}.jpg")
                        cv2.imwrite(save_path, out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        if i % max(1, n_frames // 100) == 0:
                            pct = int(100.0 * (i+1) / n_frames)
                            self.progress.emit(pct)
                    except Exception:
                        # skip problematic frames
                        continue

            self.finished.emit(temp_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class C_Elegans_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("C. Elegans Tracker (SAM 2)")
        self.resize(1000, 800)

        # State Variables
        self.video_path = None
        self.image_files = [] # sorted list of full paths
        self.current_frame_idx = 0
        # store last loaded RGB full-resolution frame for coordinate mapping
        self._last_loaded_rgb = None
        # hovered prompt point state: (obj_idx, point_idx) or None
        self._hovered_point = None
        self.blob_centers = None # Stores result of Autosegment
        self.autoseg_frame_idx = None
        self.tracking_results = None # Stores result of SAM 2
        self.colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8) # Pre-generate 100 random colors
        # Mapping from frame_idx -> mask file path (pre-colored JPGs)
        self.tracking_mask_files = {}
        # Simple in-memory cache for recently-used color masks to speed up display
        self._mask_cache = {}
        self._mask_cache_max = 64
        # Pixmap cache (scaled QPixmaps) for fast display
        # Use OrderedDict for LRU behavior (most-recently-used at end)
        self.pixmap_cache = OrderedDict()
        self.pixmap_cache_max = 512
        # Prefetch radius (number of frames ahead/behind to load)
        self.prefetch_radius = 3
        # Playback state
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(100)  # default 100ms -> 10 FPS; adjust as needed
        self.play_timer.timeout.connect(self._on_play_timeout)
        self.playing = False

        # Ensure main window can receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Setup Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # UI Setup
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.setup_selection_screen()
        self.setup_tracking_screen()

        # Background image loader thread
        self._image_loader = ImageLoader(self)
        self._image_loader.frameLoaded.connect(self._on_frame_loaded)
        self._image_loader.start()
        # initial references (empty) - will be updated when folder selected / tracking finished
        self._image_loader.set_references(self.image_files, self.tracking_mask_files)
        # temp dir used when exporting HDF5 frames for tracking
        self._h5_export_tempdir = None

    def setup_selection_screen(self):
        self.selection_widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        label = QLabel("Please select the folder containing your video frames (0.jpg, 1.jpg...)")
        label.setStyleSheet("font-size: 18px; margin-bottom: 20px;")
        
        btn_select = QPushButton("Select Video Folder")
        btn_select.setFixedSize(200, 50)
        btn_select.clicked.connect(self.choose_folder)

        btn_import_h5 = QPushButton("Import HDF5 File")
        btn_import_h5.setFixedSize(200, 50)
        btn_import_h5.clicked.connect(self.choose_hdf5)

        layout.addWidget(label)
        layout.addWidget(btn_select, alignment=Qt.AlignCenter)
        layout.addWidget(btn_import_h5, alignment=Qt.AlignCenter)
        self.selection_widget.setLayout(layout)
        self.stacked_widget.addWidget(self.selection_widget)

    def choose_hdf5(self):
        if not H5PY_AVAILABLE:
            QMessageBox.critical(self, "Error", "h5py is not installed. Please install h5py to import HDF5 files.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select HDF5 file", filter="HDF5 Files (*.h5 *.hdf5);;All Files (*)")
        if not path:
            return

        # Attempt to inspect the file to find a suitable dataset
        try:
            with h5py.File(path, 'r') as f:
                # prefer dataset named 'images' or 'frames', otherwise take first dataset
                ds_name = None
                for pref in ('images', 'frames', 'data'):
                    if pref in f:
                        ds_name = pref
                        break
                if ds_name is None:
                    # pick first dataset
                    for k in f.keys():
                        if isinstance(f[k], h5py.Dataset):
                            ds_name = k
                            break
                if ds_name is None:
                    QMessageBox.critical(self, "Error", "No dataset found in HDF5 file.")
                    return
                n_frames = f[ds_name].shape[0]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read HDF5 file: {e}")
            return

        # set up HDF5-backed dataset
        self.video_path = None
        self.image_files = None
        self.h5_file_path = path
        self.h5_dataset_name = ds_name

        # configure slider and UI
        self.slider.setMaximum(max(0, n_frames - 1))
        self.slider.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.stacked_widget.setCurrentIndex(1)

        # update image loader references and clear caches
        try:
            self._image_loader.set_references(image_files=None, mask_files=self.tracking_mask_files, h5_path=self.h5_file_path, h5_dataset=self.h5_dataset_name)
        except Exception:
            pass
        self.clear_pixmap_cache()
        self.update_display()
        # disable tracking (TrackerWorker expects a folder of frames)
        try:
            self.btn_track.setEnabled(False)
        except Exception:
            pass
        self.status_label.setText(f"Loaded HDF5: {os.path.basename(path)} ({n_frames} frames)")
        # Prefetch first frames
        try:
            self._image_loader.request(self.current_frame_idx)
            self._prefetch_neighbors(self.current_frame_idx)
        except Exception:
            pass

    def setup_tracking_screen(self):
        self.tracking_widget = QWidget()
        layout = QVBoxLayout()

        # Image Display Area
        # Left: Image display
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #222;")

        # Right: Sidebar for object buttons
        self.sidebar_area = QScrollArea()
        self.sidebar_area.setWidgetResizable(True)
        self.sidebar_content = QWidget()
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_layout.setAlignment(Qt.AlignTop)
        self.sidebar_content.setLayout(self.sidebar_layout)
        self.sidebar_area.setWidget(self.sidebar_content)
        self.sidebar_area.setFixedWidth(240)

        # Container for object buttons (will be populated after autoseg)
        self.object_button_container = QWidget()
        self.object_button_layout = QVBoxLayout()
        self.object_button_layout.setAlignment(Qt.AlignTop)
        self.object_button_container.setLayout(self.object_button_layout)
        # put a title label at top
        self.sidebar_layout.addWidget(QLabel("Objects:"))
        self.sidebar_layout.addWidget(self.object_button_container)

        # enable mouse events on the image label and install event filter for clicks
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)

        # Place image and sidebar side-by-side
        top_hlayout = QHBoxLayout()
        top_hlayout.addWidget(self.image_label, stretch=1)
        top_hlayout.addWidget(self.sidebar_area)
        layout.addLayout(top_hlayout)

        # Controls
        controls_layout = QHBoxLayout()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_change)
        controls_layout.addWidget(QLabel("Frame:"))
        controls_layout.addWidget(self.slider)

        # Play / Pause button next to the slider
        self.btn_play = QPushButton("Play")
        self.btn_play.setFixedSize(80, 24)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        controls_layout.addWidget(self.btn_play)

        # Model size selector
        controls_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "large"])
        # default to 'base'
        self.model_combo.setCurrentText("base")
        self.model_combo.currentTextChanged.connect(lambda v: setattr(self, 'model_size', v))
        controls_layout.addWidget(self.model_combo)

        layout.addLayout(controls_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_autosegment = QPushButton("Autosegment worms (Frame 1)")
        self.btn_autosegment.clicked.connect(self.run_autosegmentation)
        
        self.btn_track = QPushButton("Track")
        self.btn_track.clicked.connect(self.run_tracking)
        self.btn_track.setEnabled(False) # Disabled until autosegment is done

        btn_layout.addWidget(self.btn_autosegment)
        btn_layout.addWidget(self.btn_track)
        layout.addLayout(btn_layout)

        # Status Bar / Progress
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # object buttons state
        self.object_buttons = []
        self.selected_object_idx = None

        self.tracking_widget.setLayout(layout)
        self.stacked_widget.addWidget(self.tracking_widget)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Directory")
        if folder:
            self.video_path = folder
            # Load images
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            # Try to sort numerically if filenames are integers
            try:
                files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except ValueError:
                files.sort()
            
            self.image_files = [os.path.join(folder, f) for f in files]

            if not self.image_files:
                QMessageBox.warning(self, "Error", "No image files found in directory.")
                return

            self.slider.setMaximum(len(self.image_files) - 1)
            self.slider.setEnabled(True)
            # enable play button now that frames are available
            try:
                self.btn_play.setEnabled(True)
            except Exception:
                pass
            self.stacked_widget.setCurrentIndex(1)
            # update loader references and clear caches
            try:
                self._image_loader.set_references(self.image_files, self.tracking_mask_files)
            except Exception:
                pass
            self.clear_pixmap_cache()
            self.update_display()
            # clear caches when new folder chosen
            self.clear_pixmap_cache()

    def on_slider_change(self, value):
        self.current_frame_idx = value
        self.update_display()
        # request prefetch of neighbors
        self._prefetch_neighbors(self.current_frame_idx)

    def clear_pixmap_cache(self):
        try:
            self.pixmap_cache.clear()
        except Exception:
            self.pixmap_cache = {}

    def run_autosegmentation(self):
        if not self.image_files:
            return

        self.status_label.setText("Autosegmenting...")
        QApplication.processEvents() # Force UI update

        # Run autoseg on the currently selected frame (so prompts appear where you are)
        target_idx = self.current_frame_idx
        img_path = self.image_files[target_idx]
        img = cv2.imread(img_path)
        
        # Run user's logic
        try:
            self.blob_centers = process_image_and_scale_centers(img, downsample_resolution=8, num_skeleton_points=10)
            # mark which frame these prompts belong to
            self.autoseg_frame_idx = target_idx

            # Update the object sidebar with buttons for each detected object
            try:
                self.update_object_sidebar()
            except Exception:
                pass

            self.status_label.setText(f"Found {len(self.blob_centers)} worms. Ready to Track.")
            self.btn_track.setEnabled(True)

            # Remove any cached pixmap for that frame (it may have been created before autoseg)
            try:
                if target_idx in self.pixmap_cache:
                    self.pixmap_cache.pop(target_idx, None)
            except Exception:
                pass

            # --- IMMEDIATE DRAW: render prompt points onto the just-segmented frame and display it
            try:
                # draw on a copy (BGR) so we can show markers immediately without waiting for the loader
                img_bgr = img.copy()
                circle_radius = 15
                # bright red in BGR for selection highlight
                gold_bgr = (0, 0, 255)
                text_bgr = (255, 255, 255)
                outline_bgr = (0, 0, 0)

                for i_obj, skeleton_pts in enumerate(self.blob_centers):
                    color_rgb = tuple(map(int, self.colors[(i_obj + 1) % len(self.colors)]))
                    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
                    is_selected = (self.selected_object_idx is not None and self.selected_object_idx == i_obj)
                    for (cx, cy) in skeleton_pts:
                        cx_i = int(cx)
                        cy_i = int(cy)
                        if is_selected:
                            sel_radius = circle_radius + 12
                            inner_radius = circle_radius + 4
                            cv2.circle(img_bgr, (cx_i, cy_i), sel_radius, gold_bgr, -1)
                            cv2.circle(img_bgr, (cx_i, cy_i), inner_radius, color_bgr, -1)
                            cv2.circle(img_bgr, (cx_i, cy_i), inner_radius, outline_bgr, 4)
                            cv2.putText(img_bgr, str(i_obj+1), (cx_i+inner_radius+2, cy_i-inner_radius-2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_bgr, 3, cv2.LINE_AA)
                        else:
                            cv2.circle(img_bgr, (cx_i, cy_i), circle_radius, color_bgr, -1)
                            cv2.circle(img_bgr, (cx_i, cy_i), circle_radius, outline_bgr, 1)
                            cv2.putText(img_bgr, str(i_obj+1), (cx_i+8, cy_i-8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_bgr, 1, cv2.LINE_AA)

                # convert back to RGB for Qt
                img_rgb_now = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb_now = np.ascontiguousarray(img_rgb_now)
                h, w, ch = img_rgb_now.shape
                bytes_per_line = ch * w
                q_img_now = QImage(img_rgb_now.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap_now = QPixmap.fromImage(q_img_now)
                scaled_now = pixmap_now.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # cache and show immediately
                try:
                    if len(self.pixmap_cache) >= self.pixmap_cache_max:
                        try:
                            self.pixmap_cache.popitem(last=False)
                        except Exception:
                            first_key = next(iter(self.pixmap_cache))
                            self.pixmap_cache.pop(first_key, None)
                except Exception:
                    pass
                self.pixmap_cache[target_idx] = scaled_now
                try:
                    self.pixmap_cache.move_to_end(target_idx, last=True)
                except Exception:
                    pass
                self.image_label.setPixmap(scaled_now)
            except Exception:
                # fallback: request loader as before
                try:
                    self._image_loader.request(target_idx)
                except Exception:
                    pass

            # ensure slider shows that frame and request the loader to (re)load it (for masks/prefetch)
            try:
                self.slider.setValue(target_idx)
                self._image_loader.request(target_idx)
                self._prefetch_neighbors(target_idx)
            except Exception:
                pass

            # update display (will either use cache or wait for loader)
            self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Segmentation failed: {str(e)}")
            self.status_label.setText("Error in segmentation.")

    def run_tracking(self):
        if not SAM2_AVAILABLE:
            QMessageBox.critical(self, "Error", "SAM 2 library is not installed/importable.")
            return

        # If dataset was loaded from HDF5, export frames to a temporary folder first
        if getattr(self, 'h5_file_path', None) is not None:
            # start export worker
            try:
                self.btn_track.setEnabled(False)
                self.btn_autosegment.setEnabled(False)
                self.status_label.setText("Exporting HDF5 frames to temporary folder...")
                self._export_worker = ExportH5Worker(self.h5_file_path, getattr(self, 'h5_dataset_name', None))
                self._export_worker.progress.connect(lambda p: self.status_label.setText(f"Exporting frames... {p}%"))
                self._export_worker.finished.connect(self._on_h5_export_finished)
                self._export_worker.error.connect(self.on_tracking_error)
                self._export_worker.start()
                return
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start HDF5 export: {e}")
                return

        # normal folder-based tracking
        self._start_tracker_worker(self.video_path)

    def _start_tracker_worker(self, video_folder):
        """Start the TrackerWorker using the given video_folder path."""
        try:
            self.btn_track.setEnabled(False)
            self.btn_autosegment.setEnabled(False)
            self.status_label.setText("Tracking in progress... This may take a moment.")
            self.worker = TrackerWorker(video_folder, self.device, self.blob_centers, colors=self.colors, model_size=getattr(self, 'model_size', 'base'))
            self.worker.finished.connect(self.on_tracking_finished)
            self.worker.error.connect(self.on_tracking_error)
            self.worker.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start tracker: {e}")
            self.btn_track.setEnabled(True)
            self.btn_autosegment.setEnabled(True)

    def _prefetch_neighbors(self, idx):
        # Request loading of current frame and neighbors
        start = max(0, idx - self.prefetch_radius)
        end = min(self.slider.maximum(), idx + self.prefetch_radius)
        for i in range(start, end + 1):
            if i not in self.pixmap_cache:
                self._image_loader.request(i)

    def _on_frame_loaded(self, idx, img_rgb):
        # Called in GUI thread when image loader has RGB numpy image ready
        try:
            img_rgb = np.ascontiguousarray(img_rgb)
            # keep a copy of the full-resolution RGB for coordinate mapping (when showing current frame)
            if idx == self.current_frame_idx:
                try:
                    self._last_loaded_rgb = img_rgb.copy()
                except Exception:
                    self._last_loaded_rgb = None
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Draw skeleton markers for the autosegmentation frame if available
            if self.autoseg_frame_idx is not None and idx == self.autoseg_frame_idx and self.blob_centers:
                # Draw filled circles and labels so markers are visible over masks
                # Convert RGB->BGR for OpenCV drawing, then back to RGB
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                circle_radius = 15
                # bright red in BGR for selection highlight
                gold_bgr = (0, 0, 255)
                text_bgr = (255, 255, 255)
                outline_bgr = (0, 0, 0)

                for i_obj, skeleton_pts in enumerate(self.blob_centers):
                    try:
                        is_selected = (self.selected_object_idx is not None and self.selected_object_idx == i_obj)
                        color_rgb = tuple(map(int, self.colors[(i_obj + 1) % len(self.colors)]))
                        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
                        for (cx, cy) in skeleton_pts:
                            cx_i = int(cx)
                            cy_i = int(cy)
                            if is_selected:
                                # draw red outer ring then inner filled color to emphasize selection
                                    sel_radius = circle_radius + 12
                                    inner_radius = circle_radius + 4
                                    cv2.circle(img_bgr, (cx_i, cy_i), sel_radius, gold_bgr, -1)
                                    cv2.circle(img_bgr, (cx_i, cy_i), inner_radius, color_bgr, -1)
                                    cv2.circle(img_bgr, (cx_i, cy_i), inner_radius, outline_bgr, 4)
                                    try:
                                        cv2.putText(img_bgr, str(i_obj+1), (cx_i+inner_radius+2, cy_i-inner_radius-2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_bgr, 3, cv2.LINE_AA)
                                    except Exception:
                                        pass
                            else:
                                cv2.circle(img_bgr, (cx_i, cy_i), circle_radius, color_bgr, -1)
                                cv2.circle(img_bgr, (cx_i, cy_i), circle_radius, outline_bgr, 1)
                                try:
                                    cv2.putText(img_bgr, str(i_obj+1), (cx_i+8, cy_i-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_bgr, 1, cv2.LINE_AA)
                                except Exception:
                                    pass
                    except Exception:
                        continue
                # convert back to RGB for Qt
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb = np.ascontiguousarray(img_rgb)
                q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # cache scaled pixmap
            try:
                    if len(self.pixmap_cache) >= self.pixmap_cache_max:
                        try:
                            # pop least-recently-used (first item)
                            self.pixmap_cache.popitem(last=False)
                        except Exception:
                            # fallback: clear one item
                            first_key = next(iter(self.pixmap_cache))
                            self.pixmap_cache.pop(first_key, None)
                    self.pixmap_cache[idx] = scaled_pixmap
                    try:
                        # mark as recently used
                        self.pixmap_cache.move_to_end(idx, last=True)
                    except Exception:
                        pass
            except Exception:
                self.pixmap_cache[idx] = scaled_pixmap

            # if this is the currently visible frame, display it
            if idx == self.current_frame_idx:
                self.image_label.setPixmap(scaled_pixmap)
        except Exception:
            pass

    def toggle_play(self):
        """Toggle playback: start or stop the play timer."""
        if not self.slider.isEnabled():
            return

        if self.playing:
            # Stop playback
            self.play_timer.stop()
            self.playing = False
            try:
                self.btn_play.setText("Play")
            except Exception:
                pass
        else:
            # Start playback
            # If already at last frame, jump to current to end
            if self.current_frame_idx >= self.slider.maximum():
                self.slider.setValue(self.slider.minimum())
            self.playing = True
            try:
                self.btn_play.setText("Pause")
            except Exception:
                pass
            self.play_timer.start()

    def _on_play_timeout(self):
        """Advance one frame on each timer timeout; stop at the end."""
        if self.current_frame_idx < self.slider.maximum():
            # advance by 1 using slider so signals/update_display happen
            self.slider.setValue(self.current_frame_idx + 1)
        else:
            # reached end; stop playback
            self.play_timer.stop()
            self.playing = False
            try:
                self.btn_play.setText("Play")
            except Exception:
                pass

    def keyPressEvent(self, event):
        """Handle left/right arrow keys to navigate frames."""
        key = event.key()
        if key == Qt.Key_Right:
            # move forward one frame
            if self.current_frame_idx < self.slider.maximum():
                self.slider.setValue(self.current_frame_idx + 1)
        elif key == Qt.Key_Left:
            # move back one frame
            if self.current_frame_idx > self.slider.minimum():
                self.slider.setValue(self.current_frame_idx - 1)
        else:
            # pass other keys to base class
            super().keyPressEvent(event)

    def on_tracking_finished(self, results):
        """`results` is the mask output folder path emitted by the worker.
        Build an index of available mask JPGs and prepare cache structures for fast loading.
        """
        mask_folder = results
        # Reset index/cache
        self.tracking_mask_files = {}
        self._mask_cache.clear()

        if os.path.isdir(mask_folder):
            for fname in os.listdir(mask_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name, _ = os.path.splitext(fname)
                    try:
                        idx = int(name)
                    except ValueError:
                        continue
                    self.tracking_mask_files[idx] = os.path.join(mask_folder, fname)

        # update image loader references to include mask files
        try:
            self._image_loader.set_references(image_files=self.image_files, mask_files=self.tracking_mask_files, h5_path=getattr(self, 'h5_file_path', None), h5_dataset=getattr(self, 'h5_dataset_name', None))
        except Exception:
            pass

        # Mark results available
        self.tracking_results = True
        self.status_label.setText("Tracking Complete.")
        self.btn_track.setEnabled(True)
        self.btn_autosegment.setEnabled(True)
        # Clear any cached scaled pixmaps (they were created without masks)
        try:
            self.pixmap_cache.clear()
        except Exception:
            self.pixmap_cache = {}

        # Prefetch current frame (so the masked version loads) and update display
        try:
            self._image_loader.request(self.current_frame_idx)
            self._prefetch_neighbors(self.current_frame_idx)
        except Exception:
            pass

        self.update_display()

    def eventFilter(self, obj, event):
        # handle hover and clicks on the image label
        try:
            if obj == self.image_label:
                # Mouse move: detect hover over any plotted prompt point
                if event.type() == QEvent.MouseMove:
                    # reset hover
                    self._hovered_point = None
                    pixmap = self.image_label.pixmap()
                    if pixmap is None or not self.blob_centers:
                        self.image_label.setCursor(Qt.ArrowCursor)
                        return False

                    label_w = self.image_label.width()
                    label_h = self.image_label.height()
                    pixmap_w = pixmap.width()
                    pixmap_h = pixmap.height()
                    offset_x = max(0, (label_w - pixmap_w) // 2)
                    offset_y = max(0, (label_h - pixmap_h) // 2)

                    pos = event.pos()
                    x_in = pos.x() - offset_x
                    y_in = pos.y() - offset_y
                    if x_in < 0 or y_in < 0 or x_in >= pixmap_w or y_in >= pixmap_h:
                        self.image_label.setCursor(Qt.ArrowCursor)
                        return False

                    # get original image size
                    try:
                        if self._last_loaded_rgb is not None:
                            orig_h, orig_w = self._last_loaded_rgb.shape[:2]
                        elif getattr(self, 'h5_file_path', None) is not None and H5PY_AVAILABLE:
                            try:
                                with h5py.File(self.h5_file_path, 'r') as f:
                                    ds = self.h5_dataset_name if getattr(self, 'h5_dataset_name', None) is not None else next(iter(f.keys()))
                                    arr = f[ds][self.current_frame_idx]
                                    tmp = np.asarray(arr)
                                    orig_h, orig_w = tmp.shape[:2]
                            except Exception:
                                self.image_label.setCursor(Qt.ArrowCursor)
                                return False
                        else:
                            orig = cv2.imread(self.image_files[self.current_frame_idx])
                            orig_h, orig_w = orig.shape[:2]
                    except Exception:
                        self.image_label.setCursor(Qt.ArrowCursor)
                        return False

                    # compute scale from original->pixmap (preserve aspect ratio so scales equal)
                    scale_x = pixmap_w / float(orig_w)
                    scale_y = pixmap_h / float(orig_h)
                    scale = min(scale_x, scale_y)

                    # threshold in display pixels for selecting a point
                    thr = 12
                    found = False
                    # iterate over all points to find nearest
                    for obj_idx, pts in enumerate(self.blob_centers):
                        for pt_idx, (cx, cy) in enumerate(pts):
                            disp_x = int(cx * scale) + offset_x
                            disp_y = int(cy * scale) + offset_y
                            dx = disp_x - pos.x()
                            dy = disp_y - pos.y()
                            if dx*dx + dy*dy <= thr*thr:
                                self._hovered_point = (obj_idx, pt_idx)
                                self.image_label.setCursor(Qt.PointingHandCursor)
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        self.image_label.setCursor(Qt.ArrowCursor)
                    return False

                # Mouse button press: handle different click behaviors
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    # Ctrl+Left: add new object at click (existing behavior)
                    if (event.modifiers() & Qt.ControlModifier):
                        # only allow adding on the first frame
                        if self.current_frame_idx != 0:
                            self.status_label.setText("Ctrl+Click additions only allowed on first frame.")
                            return True

                        pixmap = self.image_label.pixmap()
                        if pixmap is None:
                            return False
                        label_w = self.image_label.width()
                        label_h = self.image_label.height()
                        pixmap_w = pixmap.width()
                        pixmap_h = pixmap.height()
                        offset_x = max(0, (label_w - pixmap_w) // 2)
                        offset_y = max(0, (label_h - pixmap_h) // 2)

                        pos = event.pos()
                        x_in = pos.x() - offset_x
                        y_in = pos.y() - offset_y
                        if x_in < 0 or y_in < 0 or x_in >= pixmap_w or y_in >= pixmap_h:
                            return True

                        if self._last_loaded_rgb is None:
                            try:
                                if getattr(self, 'h5_file_path', None) is not None and H5PY_AVAILABLE:
                                    with h5py.File(self.h5_file_path, 'r') as f:
                                        ds = self.h5_dataset_name if getattr(self, 'h5_dataset_name', None) is not None else next(iter(f.keys()))
                                        arr = f[ds][0]
                                        tmp = np.asarray(arr)
                                        orig_h, orig_w = tmp.shape[:2]
                                else:
                                    orig = cv2.imread(self.image_files[0])
                                    if orig is None:
                                        return True
                                    orig_h, orig_w = orig.shape[:2]
                            except Exception:
                                return True
                        else:
                            orig_h, orig_w = self._last_loaded_rgb.shape[:2]

                        x_orig = int((x_in * orig_w) / pixmap_w)
                        y_orig = int((y_in * orig_h) / pixmap_h)

                        if self.blob_centers is None:
                            self.blob_centers = []
                        self.blob_centers.append([(x_orig, y_orig)])
                        self.autoseg_frame_idx = 0
                        try:
                            self.update_object_sidebar()
                            self._image_loader.request(0)
                            self._prefetch_neighbors(0)
                            self.slider.setValue(0)
                            self.update_display()
                            self.status_label.setText(f"Added object at ({x_orig},{y_orig})")
                        except Exception:
                            pass
                        return True

                    # Left click without Ctrl: if hovering on a point -> delete that point
                    if self._hovered_point is not None:
                        obj_idx, pt_idx = self._hovered_point
                        try:
                            if 0 <= obj_idx < len(self.blob_centers) and 0 <= pt_idx < len(self.blob_centers[obj_idx]):
                                self.blob_centers[obj_idx].pop(pt_idx)
                                # if no points left for object, remove object entirely
                                if len(self.blob_centers[obj_idx]) == 0:
                                    self.blob_centers.pop(obj_idx)
                                    # adjust selected index
                                    if self.selected_object_idx is not None:
                                        if self.selected_object_idx == obj_idx:
                                            self.selected_object_idx = None
                                        elif self.selected_object_idx > obj_idx:
                                            self.selected_object_idx -= 1
                        except Exception:
                            pass

                        # refresh UI (invalidate cache and request recomposition)
                        try:
                            self.update_object_sidebar()
                            if self.current_frame_idx in self.pixmap_cache:
                                self.pixmap_cache.pop(self.current_frame_idx, None)
                            self._image_loader.request(self.current_frame_idx)
                            self._prefetch_neighbors(self.current_frame_idx)
                            self.update_display()
                            self.status_label.setText("Deleted prompt point")
                        except Exception:
                            pass
                        return True

                    # Left click without Ctrl and not on a point: if an object is selected, add a prompt point to that object
                    if self.selected_object_idx is not None and self.blob_centers is not None and self.current_frame_idx == (self.autoseg_frame_idx or self.current_frame_idx):
                        pixmap = self.image_label.pixmap()
                        if pixmap is None:
                            return False
                        label_w = self.image_label.width()
                        label_h = self.image_label.height()
                        pixmap_w = pixmap.width()
                        pixmap_h = pixmap.height()
                        offset_x = max(0, (label_w - pixmap_w) // 2)
                        offset_y = max(0, (label_h - pixmap_h) // 2)

                        pos = event.pos()
                        x_in = pos.x() - offset_x
                        y_in = pos.y() - offset_y
                        if x_in < 0 or y_in < 0 or x_in >= pixmap_w or y_in >= pixmap_h:
                            return True

                        # original image size
                        try:
                            if self._last_loaded_rgb is not None and self.current_frame_idx == self.current_frame_idx:
                                orig_h, orig_w = self._last_loaded_rgb.shape[:2]
                            elif getattr(self, 'h5_file_path', None) is not None and H5PY_AVAILABLE:
                                with h5py.File(self.h5_file_path, 'r') as f:
                                    ds = self.h5_dataset_name if getattr(self, 'h5_dataset_name', None) is not None else next(iter(f.keys()))
                                    arr = f[ds][self.current_frame_idx]
                                    tmp = np.asarray(arr)
                                    orig_h, orig_w = tmp.shape[:2]
                            else:
                                orig = cv2.imread(self.image_files[self.current_frame_idx])
                                orig_h, orig_w = orig.shape[:2]
                        except Exception:
                            return True

                        x_orig = int((x_in * orig_w) / pixmap_w)
                        y_orig = int((y_in * orig_h) / pixmap_h)

                        try:
                            # append to selected object's prompt list
                            self.blob_centers[self.selected_object_idx].append((x_orig, y_orig))
                        except Exception:
                            # if selected index invalid, ignore
                            return True

                        # refresh UI (invalidate cache and request recomposition)
                        try:
                            self.update_object_sidebar()
                            if self.current_frame_idx in self.pixmap_cache:
                                self.pixmap_cache.pop(self.current_frame_idx, None)
                            self._image_loader.request(self.current_frame_idx)
                            self._prefetch_neighbors(self.current_frame_idx)
                            self.update_display()
                            self.status_label.setText(f"Added prompt to object {self.selected_object_idx+1}")
                        except Exception:
                            pass
                        return True

                # end mouse press handling
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        # Stop background threads cleanly
        try:
            if hasattr(self, '_image_loader') and self._image_loader is not None:
                self._image_loader.stop()
        except Exception:
            pass
        try:
            if hasattr(self, 'worker') and getattr(self, 'worker') is not None:
                try:
                    self.worker.terminate()
                except Exception:
                    pass
        except Exception:
            pass
        # remove temporary exported HDF5 frames if present
        try:
            if getattr(self, '_h5_export_tempdir', None):
                try:
                    shutil.rmtree(self._h5_export_tempdir)
                except Exception:
                    pass
        except Exception:
            pass
        super().closeEvent(event)

    def on_tracking_error(self, err_msg):
        QMessageBox.critical(self, "Tracking Error", err_msg)
        self.status_label.setText("Tracking Failed.")
        self.btn_track.setEnabled(True)
        self.btn_autosegment.setEnabled(True)

    def _on_h5_export_finished(self, temp_folder):
        """Called when ExportH5Worker finishes exporting frames to `temp_folder`."""
        try:
            self._h5_export_tempdir = temp_folder
            # start tracker with the exported folder
            self._start_tracker_worker(temp_folder)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start tracker after export: {e}")
            self.btn_track.setEnabled(True)
            self.btn_autosegment.setEnabled(True)

    def clear_layout(self, layout):
        """Remove all widgets from a layout."""
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                try:
                    widget.deleteLater()
                except Exception:
                    pass
            else:
                child_layout = item.layout()
                if child_layout is not None:
                    self.clear_layout(child_layout)

    # overlay helper was removed per user request

    def update_object_sidebar(self):
        """Populate the right-hand sidebar with one colored button per detected object."""
        # Clear existing buttons
        try:
            self.clear_layout(self.object_button_layout)
        except Exception:
            pass

        self.object_buttons = []

        if not self.blob_centers:
            lbl = QLabel("No objects detected")
            self.object_button_layout.addWidget(lbl)
            return

        # create a row for each object: [Object btn] [Delete btn]
        for i_obj, skeleton_pts in enumerate(self.blob_centers):
            row_widget = QWidget()
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)

            btn = QPushButton(f"Worm {i_obj+1}")
            btn.setFixedHeight(30)
            btn.setMinimumWidth(140)

            # color chosen in same way as drawing code (use object id +1)
            color_rgb = tuple(map(int, self.colors[(i_obj + 1) % len(self.colors)]))
            r, g, b = color_rgb

            # choose text color based on luminance for readability
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = '#000000' if luminance > 180 else '#FFFFFF'

            base_style = f"background-color: rgb({r},{g},{b}); color: {text_color}; border-radius: 6px;"
            btn.setStyleSheet(base_style)
            btn.clicked.connect(lambda _, idx=i_obj: self.on_object_button_clicked(idx))

            # delete button
            del_btn = QPushButton("✕")
            del_btn.setFixedSize(28, 28)
            del_btn.setStyleSheet("background-color: #ff4d4d; color: #fff; border-radius: 4px;")
            del_btn.clicked.connect(lambda _, idx=i_obj: self.delete_object(idx))

            row_layout.addWidget(btn)
            row_layout.addWidget(del_btn)
            row_widget.setLayout(row_layout)

            self.object_buttons.append(btn)
            self.object_button_layout.addWidget(row_widget)

        # add stretch to push buttons to top
        self.object_button_layout.addStretch(1)

    def on_object_button_clicked(self, idx):
        """Handle object button click: mark selected and update UI."""
        self.selected_object_idx = idx
        self.status_label.setText(f"Selected object {idx+1}")

        # update visual highlight for buttons
        for i, btn in enumerate(self.object_buttons):
            try:
                # recompute base color
                color_rgb = tuple(map(int, self.colors[(i + 1) % len(self.colors)]))
                r, g, b = color_rgb
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = '#000000' if luminance > 180 else '#FFFFFF'
                if i == idx:
                    # highlighted border
                    btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {text_color}; border: 3px solid #FFD700; border-radius: 15px;")
                else:
                    btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {text_color}; border-radius: 6px;")
            except Exception:
                pass

        # Determine which frame to refresh (autoseg frame preferred)
        target = self.autoseg_frame_idx if self.autoseg_frame_idx is not None else self.current_frame_idx

        # Invalidate any existing scaled pixmap for that frame so update_display won't reuse stale image
        try:
            if target in self.pixmap_cache:
                self.pixmap_cache.pop(target, None)
        except Exception:
            pass

        # Try to draw selection immediately from the last full-resolution RGB we have
        drawn_immediately = False
        try:
            if self._last_loaded_rgb is not None and target == self.current_frame_idx:
                # Work on a copy (BGR) and draw selection markers as in _on_frame_loaded/run_autosegmentation
                try:
                    img_rgb = self._last_loaded_rgb.copy()
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    circle_radius = 15
                    # bright red in BGR for selection highlight
                    gold_bgr = (0, 0, 255)
                    text_bgr = (255, 255, 255)
                    outline_bgr = (0, 0, 0)

                    for i_obj, skeleton_pts in enumerate(self.blob_centers or []):
                        is_selected = (self.selected_object_idx is not None and self.selected_object_idx == i_obj)
                        color_rgb = tuple(map(int, self.colors[(i_obj + 1) % len(self.colors)]))
                        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
                        for (cx, cy) in skeleton_pts:
                            cx_i = int(cx)
                            cy_i = int(cy)
                            if is_selected:
                                sel_radius = circle_radius + 12
                                inner_radius = circle_radius + 4
                                cv2.circle(img_bgr, (cx_i, cy_i), sel_radius, gold_bgr, -1)
                                cv2.circle(img_bgr, (cx_i, cy_i), inner_radius, color_bgr, -1)
                                cv2.circle(img_bgr, (cx_i, cy_i), inner_radius, outline_bgr, 4)
                                cv2.putText(img_bgr, str(i_obj+1), (cx_i+inner_radius+2, cy_i-inner_radius-2),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_bgr, 3, cv2.LINE_AA)
                            else:
                                cv2.circle(img_bgr, (cx_i, cy_i), circle_radius, color_bgr, -1)
                                cv2.circle(img_bgr, (cx_i, cy_i), circle_radius, outline_bgr, 1)
                                cv2.putText(img_bgr, str(i_obj+1), (cx_i+8, cy_i-8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_bgr, 1, cv2.LINE_AA)

                    img_rgb_now = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_rgb_now = np.ascontiguousarray(img_rgb_now)
                    h, w, ch = img_rgb_now.shape
                    bytes_per_line = ch * w
                    q_img_now = QImage(img_rgb_now.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap_now = QPixmap.fromImage(q_img_now)
                    scaled_now = pixmap_now.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                    # cache and show immediately
                    try:
                        if len(self.pixmap_cache) >= self.pixmap_cache_max:
                            try:
                                self.pixmap_cache.popitem(last=False)
                            except Exception:
                                first_key = next(iter(self.pixmap_cache))
                                self.pixmap_cache.pop(first_key, None)
                    except Exception:
                        pass
                    self.pixmap_cache[target] = scaled_now
                    try:
                        self.pixmap_cache.move_to_end(target, last=True)
                    except Exception:
                        pass
                    if target == self.current_frame_idx:
                        self.image_label.setPixmap(scaled_now)
                    drawn_immediately = True
                except Exception:
                    drawn_immediately = False
        except Exception:
            drawn_immediately = False

        # If immediate draw failed, request the background loader to recompose this frame (with masks) and update when done
        try:
            if not drawn_immediately:
                self._image_loader.request(target)
            self._prefetch_neighbors(target)
            # ensure update_display will not immediately reuse old cached pixmap (we popped it above)
            self.update_display()
        except Exception:
            pass

    def delete_object(self, idx):
        """Delete object at index `idx` from blob_centers and refresh UI."""
        if not self.blob_centers:
            return
        try:
            # Remove the selected blob
            if 0 <= idx < len(self.blob_centers):
                self.blob_centers.pop(idx)
        except Exception:
            pass

        # reset selected index if out of range
        if self.selected_object_idx is not None:
            if self.selected_object_idx >= len(self.blob_centers):
                self.selected_object_idx = None

        # Rebuild sidebar buttons (they will be re-numbered)
        try:
            self.update_object_sidebar()
        except Exception:
            pass

        # Refresh display so markers reflect deletion
        try:
            # ensure autoseg frame redraw
            if self.autoseg_frame_idx is None:
                self.autoseg_frame_idx = self.current_frame_idx
            # invalidate cached scaled pixmap so update_display won't reuse stale image
            try:
                if self.current_frame_idx in self.pixmap_cache:
                    self.pixmap_cache.pop(self.current_frame_idx, None)
            except Exception:
                pass
            self._image_loader.request(self.autoseg_frame_idx)
            self._prefetch_neighbors(self.autoseg_frame_idx)
            self.update_display()
        except Exception:
            pass

    def update_display(self):
        # allow HDF5-backed datasets as well as file lists
        if not self.image_files and not getattr(self, 'h5_file_path', None):
            return
        # If we have a cached scaled pixmap for this frame, use it immediately
        if self.current_frame_idx in self.pixmap_cache:
            try:
                pm = self.pixmap_cache[self.current_frame_idx]
                # update LRU ordering
                try:
                    self.pixmap_cache.move_to_end(self.current_frame_idx, last=True)
                except Exception:
                    pass
                self.image_label.setPixmap(pm)
            except Exception:
                pass
            return

        # Otherwise request the image loader to load this frame (and neighbors via prefetch)
        self._image_loader.request(self.current_frame_idx)
        self._prefetch_neighbors(self.current_frame_idx)

        # Show a lightweight placeholder while loading
        self.image_label.setText("Loading...")
        # Note: actual composition/blending happens in background loader to minimize UI thread work


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = C_Elegans_GUI()
    window.show()
    sys.exit(app.exec_())