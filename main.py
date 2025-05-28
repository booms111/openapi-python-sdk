import sys
from pathlib import Path
import os
import multiprocessing

# Ensure the script's directory is in sys.path for module imports
# This assumes main.py and pipeline_module.py are in the same directory when run.
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.append(str(SCRIPT_DIR))
        # Optional: Log this change if logging is already configured.
        # However, logging might not be set up this early.
        # print(f"Appended to sys.path: {SCRIPT_DIR}") 
except NameError:
    # __file__ might not be defined in some environments (e.g. certain notebook contexts directly)
    # Fallback to CWD in that case, though less reliable if script isn't in CWD.
    SCRIPT_DIR = Path.cwd()
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.append(str(SCRIPT_DIR))
        # print(f"Appended CWD to sys.path: {SCRIPT_DIR}")

# Set multiprocessing start method for CUDA safety FIRST
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # In some environments, it might already be set or not allowed to be forced.
    # Pass silently or add a print statement if logging is not yet available.
    print("Warning: Could not force multiprocessing start method to 'spawn'.", file=sys.stderr)

import torch # For torch.cuda.is_available() and version
import logging # For configuring logging in main
import pandas as pd # For reading potential ground truth
import numpy as np # Retained, though HybridConfig (which uses it) is now in pipeline_module
from dataclasses import asdict # For logging asdict(cfg)
from typing import List, Optional, Tuple, Dict, Set, Any # Retained for general use
from pipeline_module import HybridConfig # For instantiating cfg

# Configure logging for the main script as well
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Main script logging configured.")
logging.info("Multiprocessing start method set to 'spawn' (or attempted).") # Log after basicConfig

# --- PyTorch and CUDA Check ---
logging.info(f"PyTorch version: {torch.version}")
logging.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"CUDA version PyTorch is using: {torch.version.cuda}")
    logging.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
    logging.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    logging.info(f"Device compute capability: {capability[0]}.{capability[1]}")
else:
    logging.warning("WARNING: CUDA IS NOT AVAILABLE TO PYTORCH. GPU ACCELERATION WILL NOT WORK.")

# --- Corrected Paths to your model code datasets (VGGT and DUSt3R source code) ---
# These paths are derived directly from the 'ls -R' output you provided.
# NOTE: These are used by pipeline_module.py to add to sys.path and load models.
# They are included here in main as they are crucial for setting up the environment correctly
# before any potential imports of the actual models in pipeline_module.
VGGT_DATASET_ROOT_MOUNT = "/kaggle/input/vggt"
DUST3R_DATASET_ROOT_MOUNT = "/kaggle/input/dust3r"

# --- Define HybridConfig Dataclass in main ---
# This will be imported by pipeline_module.py via "from main import HybridConfig"
# HYBRIDCONFIG DATACLASS HAS BEEN MOVED TO PIPELINE_MODULE.PY

# logging.info("HybridConfig dataclass defined.") # Commented out as it's moved

# --- Path Setup ---
# Assume pipeline_module.py will be in /kaggle/input/pipeline or current working directory
PIPELINE_MODULE_PATH = '/kaggle/input/pipeline'

current_script_dir = Path.cwd()
if not (Path(PIPELINE_MODULE_PATH) / "pipeline_module.py").exists():
    logging.warning(f"Warning: '{PIPELINE_MODULE_PATH}/pipeline_module.py' not found (Kaggle path).")
    if (current_script_dir / "pipeline_module.py").exists(): # Check current working directory
        logging.info(f"Assuming 'pipeline_module.py' is in CWD: {current_script_dir} for local testing.")
        PIPELINE_MODULE_PATH = str(current_script_dir)
    else:
        logging.error(f"Error: 'pipeline_module.py' not found in {PIPELINE_MODULE_PATH} or CWD {current_script_dir}.")
        sys.exit("Pipeline module not found. Exiting.")
else:
    logging.info(f"Found 'pipeline_module.py' in {PIPELINE_MODULE_PATH}")

if PIPELINE_MODULE_PATH not in sys.path:
    sys.path.append(PIPELINE_MODULE_PATH)
    logging.info(f"Appended '{PIPELINE_MODULE_PATH}' to sys.path")


# --- Directory Setup (Simulating Kaggle structure for local testing if needed) ---
BASE_DIR = Path.cwd()

# Using /kaggle/working for output as is standard on Kaggle for actual runs
# For local simulation, we create these paths
KAGGLE_INPUT_DIR_SIM = BASE_DIR / "kaggle_input_cell1_sim"
KAGGLE_WORKING_DIR_SIM = BASE_DIR / "kaggle_working_cell1_sim"

# Paths for the dummy data to be created by this script
# LOCAL_TEST_DATA_ROOT will be the 'IMAGE_DIR' for cfg
LOCAL_TEST_DATA_ROOT = KAGGLE_INPUT_DIR_SIM / "hybrid_sfm_test_data_local"
LOCAL_OUTPUT_ROOT = KAGGLE_WORKING_DIR_SIM / "hybrid_sfm_output_local"

LOCAL_TEST_DATA_ROOT.mkdir(parents=True, exist_ok=True)
LOCAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

logging.info(f"LOCAL_TEST_DATA_ROOT (for dummy data): {LOCAL_TEST_DATA_ROOT.resolve()}")
logging.info(f"LOCAL_OUTPUT_ROOT (for dummy output): {LOCAL_OUTPUT_ROOT.resolve()}")

# --- Create Dummy Test Data with Scene/Images structure ---
dummy_scene_name = "scene_01_cell1_dummy"
# The pipeline expects IMAGE_DIR/scene_name/images/image.jpg
dummy_scene_images_path = LOCAL_TEST_DATA_ROOT / dummy_scene_name / "images"
dummy_scene_images_path.mkdir(parents=True, exist_ok=True)

try:
    (dummy_scene_images_path / "img001.jpg").touch(exist_ok=True)
    (dummy_scene_images_path / "img002.jpg").touch(exist_ok=True)
    (dummy_scene_images_path / "img003.png").touch(exist_ok=True)
    logging.info(f"Created dummy images in: {dummy_scene_images_path.resolve()}")
except Exception as e:
    logging.error(f"Could not create dummy image files: {e}")

# Create dummy model weight files if they don't exist for local testing
dummy_vggt_weights_path = KAGGLE_INPUT_DIR_SIM / "models" / "vggt_model.pt"
dummy_dust3r_weights_path = KAGGLE_INPUT_DIR_SIM / "models" / "dust3r_model.pth"
dummy_vggt_weights_path.parent.mkdir(parents=True, exist_ok=True)
dummy_dust3r_weights_path.parent.mkdir(parents=True, exist_ok=True)
dummy_vggt_weights_path.touch(exist_ok=True)
dummy_dust3r_weights_path.touch(exist_ok=True)
logging.info(f"Created dummy weight files: {dummy_vggt_weights_path}, {dummy_dust3r_weights_path}")


# --- Configuration for the Pipeline ---
# 'cfg' will be in the main scope and used by Cell 2 (below)
cfg = HybridConfig(
    # Override default Kaggle paths for local testing with dummy data
    # VGGT_WEIGHTS=str(dummy_vggt_weights_path),
    # DUST3R_WEIGHTS=str(dummy_dust3r_weights_path),
    # IMAGE_DIR=str(LOCAL_TEST_DATA_ROOT),       # Root containing scene folders
    # OUTPUT_DIR=str(LOCAL_OUTPUT_ROOT),        # Output directory for results

    VISUALIZE=False, # Set to True to enable GSPLAT renders (if gsplat is correctly installed)
    VERBOSE=True,
    VALIDATE=False, # Set to True if you have ground truth train_labels.csv

    # SIFT parameters (matching the field names in your HybridConfig)
    nfeatures=0,
    nOctaveLayers=3,
    contrastThreshold=0.02,
    edgeThreshold=12.0,
    sigma=1.6,

    # Other parameters that match your HybridConfig fields (override if needed)
    min_sift_matches=10,
    ransac_threshold_px=0.7,
    ransac_confidence=0.99,
    MAX_WORKERS=min(4, (os.cpu_count() or 1) - 1 if (os.cpu_count() or 0) > 1 else 1),
    BATCH_SIZE=4, # Smaller batch for dummy data/local testing
    DUST3R_CONFIDENCE_THRESHOLD=0.6,
    DUST3R_PAIR_THRESHOLD=10.0 # Increased for dummy data
)

logging.info("\n--- Pipeline Configuration (cfg object in main) ---")
logging.info(asdict(cfg))
logging.info(f"Torch version: {torch.version}, Device for pipeline: {cfg.DEVICE}")
logging.info("--- Initial Setup Complete ---")

# --- Notebook Cell 2: Execute the Pipeline ---
logging.info("--- Starting Pipeline Execution ---")

# Import the main pipeline class from the module
# The module itself will import HybridConfig from main (defined above)
try:
    from pipeline_module import Hybrid_VGGT_DUSt3R_Pipeline
    logging.info("Successfully imported Hybrid_VGGT_DUSt3R_Pipeline from pipeline_module.")
except ImportError as e:
    logging.error(f"Error importing Hybrid_VGGT_DUSt3R_Pipeline from pipeline_module: {e}")
    logging.error("Make sure Cell 1 (above) was executed and pipeline_module.py is in the sys.path.")
    # Exit or provide a dummy class to allow the rest of the cell to parse for debugging
    class Hybrid_VGGT_DUSt3R_Pipeline:
        def __init__(self, config):
            logging.error("Dummy Pipeline: Could not import actual pipeline. Using dummy.")
            self.config = config
        def run_pipeline(self):
            logging.error(f"Dummy Pipeline: Running with config: {self.config}")
            logging.error("Dummy Pipeline: Please ensure pipeline_module.py is correct and accessible.")
    sys.exit("Pipeline module import failed. Exiting.")


# Check if 'cfg' (pipeline configuration) exists from Cell 1
if 'cfg' in globals() or 'cfg' in locals():
    logging.info(f"Found 'cfg' in globals. Type: {type(cfg)}")

    # Instantiate the pipeline with the configuration from Cell 1
    logging.info("Instantiating Hybrid_VGGT_DUSt3R_Pipeline...")
    try:
        pipeline_instance = Hybrid_VGGT_DUSt3R_Pipeline(cfg) # cfg should be from __main__
        logging.info("Hybrid_VGGT_DUSt3R_Pipeline instantiated successfully.")
    except Exception as e:
        logging.error(f"ERROR: Failed to instantiate Hybrid_VGGT_DUSt3R_Pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit("Pipeline instantiation failed. Exiting.")


    # Run the pipeline
    logging.info("Running the pipeline...")
    try:
        submission_df = pipeline_instance.run_pipeline()

        logging.info("\nPipeline execution finished.")
        if submission_df is not None:
            logging.info("Pipeline execution complete. Submission DataFrame created.")
            # --- DEBUGGING: Final check for submission file existence ---
            final_submission_path = os.path.join(cfg.OUTPUT_DIR, "submission.csv")
            logging.info(f"Checking for submission.csv at: {final_submission_path}")
            if os.path.exists(final_submission_path):
                logging.info(f"CONFIRMATION: submission.csv found at {final_submission_path}")
                logging.info("First 5 lines of submission.csv:")
                # This part is environment-dependent, assuming Kaggle/Jupyter
                if 'ipykernel' in sys.modules: # Check if running in Jupyter/IPython
                    # %pprint off # This is IPython magic, cannot be directly used in a .py file
                    # !head -n 5 {final_submission_path} # This is also IPython magic
                    # Instead, read and print with Python's file I/O
                    with open(final_submission_path, 'r') as f_head:
                        for _idx_head in range(5):
                            line_head = f_head.readline()
                            if not line_head: break
                            logging.info(line_head.strip())
                else:
                    with open(final_submission_path, 'r') as f:
                        for _ in range(5):
                            line = f.readline()
                            if not line: break
                            logging.info(line.strip())
            else:
                logging.warning(f"WARNING: submission.csv NOT FOUND at {final_submission_path} after pipeline run.")
        else:
            logging.error("Pipeline did NOT complete successfully (submission_df is None). Check logs above for errors.")
    except Exception as e:
        logging.error(f"ERROR: An unhandled exception occurred during run_pipeline(): {e}")
        import traceback
        traceback.print_exc() # Print full traceback
        logging.error("Cell 2: Pipeline execution ABORTED due to runtime error.")

else:
    logging.error("Error: 'cfg' object not found in globals. Please ensure Cell 1 has been executed.")
    logging.error("Cannot initialize or run the pipeline.")

logging.info("--- Main Script Execution Complete ---")
