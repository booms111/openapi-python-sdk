import sys
from pathlib import Path
import os
import multiprocessing

# Block B: Debug file writing code
# --- Debugging block to write path info to a file ---
DEBUG_FILE_PATH = "debug_output.txt"
try:
    with open(DEBUG_FILE_PATH, "w") as df:
        df.write(f"DEBUG: Current sys.path: {sys.path}\n")
        
        # Path.cwd()
        try:
            cwd = Path.cwd()
            df.write(f"DEBUG: Current working directory (Path.cwd()): {cwd}\n")
            try:
                df.write(f"DEBUG: Files in CWD ({cwd}): {os.listdir(cwd)}\n")
            except Exception as e_ls_cwd:
                df.write(f"DEBUG: Error listing files in CWD ({cwd}): {e_ls_cwd}\n")
        except Exception as e_cwd:
            df.write(f"DEBUG: Error getting Path.cwd(): {e_cwd}\n")

        # __file__ based directory
        try:
            script_file_path = Path(__file__).resolve()
            script_dir = script_file_path.parent
            df.write(f"DEBUG: Path(__file__).resolve().parent would be: {script_dir}\n")
            try:
                df.write(f"DEBUG: Files in SCRIPT_DIR ({script_dir}): {os.listdir(script_dir)}\n")
            except Exception as e_ls_script_dir:
                df.write(f"DEBUG: Error listing files in SCRIPT_DIR ({script_dir}): {e_ls_script_dir}\n")
        except NameError:
            df.write("DEBUG: __file__ is not defined in this context.\n")
        except Exception as e_file_path:
            df.write(f"DEBUG: Error with __file__ path: {e_file_path}\n")
            
except Exception as e_debug_file:
    print(f"CRITICAL DEBUG: Failed to write to {DEBUG_FILE_PATH}: {e_debug_file}", file=sys.stderr)
# --- End of Debugging block ---

# Block C: SCRIPT_DIR sys.path append code
# Ensure the script's directory is in sys.path for module imports
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.append(str(SCRIPT_DIR))
except NameError:
    SCRIPT_DIR = Path.cwd()
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.append(str(SCRIPT_DIR))

# Block D: multiprocessing.set_start_method() code
# Set multiprocessing start method for CUDA safety FIRST
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    print("Warning: Could not force multiprocessing start method to 'spawn'.", file=sys.stderr)

# Block E: Other core imports
import torch 
import logging 
import pandas as pd 
import numpy as np 
from dataclasses import asdict 
from typing import List, Optional, Tuple, Dict, Set, Any 

# Block G: logging.basicConfig() and initial log messages
# Configure logging for the main script as well
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Main script logging configured.")
logging.info("Multiprocessing start method set to 'spawn' (or attempted).")

# Block H: The "Path Setup" block
# Assume pipeline_module.py will be in /kaggle/input/pipeline or current working directory
PIPELINE_MODULE_PATH = '/kaggle/input/pipeline'

# current_script_dir is SCRIPT_DIR from Block C
# Path.cwd() is also viable if __file__ is not defined.
# The goal is to ensure PIPELINE_MODULE_PATH is correctly identified and added to sys.path
# if pipeline_module.py is not in a standard location already covered by SCRIPT_DIR/CWD.

if not (Path(PIPELINE_MODULE_PATH) / "pipeline_module.py").exists():
    logging.warning(f"Warning: '{PIPELINE_MODULE_PATH}/pipeline_module.py' not found (Kaggle input path).")
    # SCRIPT_DIR here refers to the one calculated in Block C
    if (SCRIPT_DIR / "pipeline_module.py").exists(): 
        logging.info(f"Found 'pipeline_module.py' in script's directory: {SCRIPT_DIR}.")
        PIPELINE_MODULE_PATH = str(SCRIPT_DIR) # Use script's directory if module is there
    else:
        # If not in /kaggle/input/pipeline and not in SCRIPT_DIR, log error and exit.
        logging.error(f"Error: 'pipeline_module.py' not found in {PIPELINE_MODULE_PATH} or script directory {SCRIPT_DIR}.")
        sys.exit("Pipeline module not found. Exiting.")
else:
    logging.info(f"Found 'pipeline_module.py' in {PIPELINE_MODULE_PATH}")

# Ensure the identified PIPELINE_MODULE_PATH is in sys.path
# (It might be SCRIPT_DIR, which was already added, or /kaggle/input/pipeline)
if PIPELINE_MODULE_PATH not in sys.path:
    # Check if it's a directory before appending
    if os.path.isdir(PIPELINE_MODULE_PATH):
        sys.path.append(PIPELINE_MODULE_PATH)
        logging.info(f"Appended '{PIPELINE_MODULE_PATH}' to sys.path for pipeline_module.")
    elif os.path.isfile(PIPELINE_MODULE_PATH) and str(Path(PIPELINE_MODULE_PATH).parent) not in sys.path : # If it's a file, append its parent
        # This case should ideally not happen if PIPELINE_MODULE_PATH is set correctly to a directory.
        parent_dir = str(Path(PIPELINE_MODULE_PATH).parent)
        sys.path.append(parent_dir)
        logging.info(f"Appended parent of '{PIPELINE_MODULE_PATH}' ({parent_dir}) to sys.path.")

# Block F: from pipeline_module import HybridConfig
from pipeline_module import HybridConfig 

# Block I: The rest of the script
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

VGGT_DATASET_ROOT_MOUNT = "/kaggle/input/vggt"
DUST3R_DATASET_ROOT_MOUNT = "/kaggle/input/dust3r"

# HYBRIDCONFIG DATACLASS HAS BEEN MOVED TO PIPELINE_MODULE.PY

# --- Directory Setup (Simulating Kaggle structure for local testing if needed) ---
BASE_DIR = Path.cwd()
KAGGLE_INPUT_DIR_SIM = BASE_DIR / "kaggle_input_cell1_sim"
KAGGLE_WORKING_DIR_SIM = BASE_DIR / "kaggle_working_cell1_sim"
LOCAL_TEST_DATA_ROOT = KAGGLE_INPUT_DIR_SIM / "hybrid_sfm_test_data_local"
LOCAL_OUTPUT_ROOT = KAGGLE_WORKING_DIR_SIM / "hybrid_sfm_output_local"
LOCAL_TEST_DATA_ROOT.mkdir(parents=True, exist_ok=True)
LOCAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
logging.info(f"LOCAL_TEST_DATA_ROOT (for dummy data): {LOCAL_TEST_DATA_ROOT.resolve()}")
logging.info(f"LOCAL_OUTPUT_ROOT (for dummy output): {LOCAL_OUTPUT_ROOT.resolve()}")

dummy_scene_name = "scene_01_cell1_dummy"
dummy_scene_images_path = LOCAL_TEST_DATA_ROOT / dummy_scene_name / "images"
dummy_scene_images_path.mkdir(parents=True, exist_ok=True)
try:
    (dummy_scene_images_path / "img001.jpg").touch(exist_ok=True)
    (dummy_scene_images_path / "img002.jpg").touch(exist_ok=True)
    (dummy_scene_images_path / "img003.png").touch(exist_ok=True)
    logging.info(f"Created dummy images in: {dummy_scene_images_path.resolve()}")
except Exception as e:
    logging.error(f"Could not create dummy image files: {e}")

dummy_vggt_weights_path = KAGGLE_INPUT_DIR_SIM / "models" / "vggt_model.pt"
dummy_dust3r_weights_path = KAGGLE_INPUT_DIR_SIM / "models" / "dust3r_model.pth"
dummy_vggt_weights_path.parent.mkdir(parents=True, exist_ok=True)
dummy_dust3r_weights_path.parent.mkdir(parents=True, exist_ok=True)
dummy_vggt_weights_path.touch(exist_ok=True)
dummy_dust3r_weights_path.touch(exist_ok=True)
logging.info(f"Created dummy weight files: {dummy_vggt_weights_path}, {dummy_dust3r_weights_path}")

cfg = HybridConfig(
    VISUALIZE=False, 
    VERBOSE=True,
    VALIDATE=False, 
    nfeatures=0,
    nOctaveLayers=3,
    contrastThreshold=0.02,
    edgeThreshold=12.0,
    sigma=1.6,
    min_sift_matches=10,
    ransac_threshold_px=0.7,
    ransac_confidence=0.99,
    MAX_WORKERS=min(4, (os.cpu_count() or 1) - 1 if (os.cpu_count() or 0) > 1 else 1),
    BATCH_SIZE=4, 
    DUST3R_CONFIDENCE_THRESHOLD=0.6,
    DUST3R_PAIR_THRESHOLD=10.0 
)

logging.info("\n--- Pipeline Configuration (cfg object in main) ---")
logging.info(asdict(cfg))
logging.info(f"Torch version: {torch.version}, Device for pipeline: {cfg.DEVICE}")
logging.info("--- Initial Setup Complete ---")
logging.info("--- Starting Pipeline Execution ---")

try:
    from pipeline_module import Hybrid_VGGT_DUSt3R_Pipeline
    logging.info("Successfully imported Hybrid_VGGT_DUSt3R_Pipeline from pipeline_module.")
except ImportError as e:
    logging.error(f"Error importing Hybrid_VGGT_DUSt3R_Pipeline from pipeline_module: {e}")
    logging.error(f"Current sys.path for context: {sys.path}")
    sys.exit("Pipeline module (Hybrid_VGGT_DUSt3R_Pipeline class) import failed. Exiting.")

if 'cfg' in globals() or 'cfg' in locals():
    logging.info(f"Found 'cfg' in globals. Type: {type(cfg)}")
    logging.info("Instantiating Hybrid_VGGT_DUSt3R_Pipeline...")
    try:
        pipeline_instance = Hybrid_VGGT_DUSt3R_Pipeline(cfg)
        logging.info("Hybrid_VGGT_DUSt3R_Pipeline instantiated successfully.")
    except Exception as e:
        logging.error(f"ERROR: Failed to instantiate Hybrid_VGGT_DUSt3R_Pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit("Pipeline instantiation failed. Exiting.")

    logging.info("Running the pipeline...")
    try:
        submission_df = pipeline_instance.run_pipeline()
        logging.info("\nPipeline execution finished.")
        if submission_df is not None:
            logging.info("Pipeline execution complete. Submission DataFrame created.")
            final_submission_path = os.path.join(cfg.OUTPUT_DIR, "submission.csv")
            logging.info(f"Checking for submission.csv at: {final_submission_path}")
            if os.path.exists(final_submission_path):
                logging.info(f"CONFIRMATION: submission.csv found at {final_submission_path}")
                logging.info("First 5 lines of submission.csv:")
                if 'ipykernel' in sys.modules:
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
        traceback.print_exc() 
        logging.error("Pipeline execution ABORTED due to runtime error.")
else:
    logging.error("Error: 'cfg' object not found in globals. Please ensure Cell 1 has been executed.")
    logging.error("Cannot initialize or run the pipeline.")

logging.info("--- Main Script Execution Complete ---")
