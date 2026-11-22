"""Replay real-frame trajectory action batches and add Blender image paths to the manifest."""

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(PROJECT_DIR))

from generate_benchmark.render_train_trajectories_sim_single_step import RenderOptions, run
from main_nav_batch import SCENE_CONFIG as NAV_SCENE_CONFIG

# OPENBLAS_NUM_THREADS=1 python generate_benchmark/render_train_trajectories_real_multi_step.py

TRAJECTORY_JSON = PROJECT_DIR / 'outputs' / 'benchmark_train_traj_real_multi_step.json'
DATA_ROOT = Path('/data4/haibo/data/3D_Env')
OUTPUT_DIR = DATA_ROOT / 'train_traj_real_multi_step_renders'
NUM_WORKERS = 8
MAX_TRAJECTORIES = None  # Set to an integer to render a prefix for a pilot run.
RETRY_ERRORS = True
SYNC_ONLY = False  # Backfill paths from saved results without launching Blender.
SCENE_CONFIG = {**NAV_SCENE_CONFIG, 'render_top_down': False}
POSITION_TOLERANCE_M = 0.001
ANGLE_TOLERANCE_DEGREES = 0.1


def main():
    options = RenderOptions(trajectory_json=TRAJECTORY_JSON, data_root=DATA_ROOT, output_dir=OUTPUT_DIR,
                            scene_config=SCENE_CONFIG, num_workers=NUM_WORKERS, max_trajectories=MAX_TRAJECTORIES,
                            retry_errors=RETRY_ERRORS, sync_only=SYNC_ONLY,
                            position_tolerance_m=POSITION_TOLERANCE_M,
                            angle_tolerance_degrees=ANGLE_TOLERANCE_DEGREES, action_tool='camera_actions',
                            render_image_field='render_img_path')
    return run(options)


if __name__ == '__main__':
    raise SystemExit(main())
