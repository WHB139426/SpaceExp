"""Explore an existing point-cloud scene by entering camera actions in the terminal."""

from datetime import datetime
import math
from pathlib import Path

from scene.client import CameraClient


SCENE_CONFIG = {
    "scene_dir": "/data4/haibo/data/3D_Env/scannetpp/307e3262f1",
    "blender": "/data4/haibo/software/blender/blender-4.5.13-linux-x64/blender",
    "timeout": 300,
    "resolution": 960,
    "samples": 16,
    "trajectory_resolution": 960,
    "trajectory_samples": 4,
    "trajectory_max_height": 1.8,
    "trajectory_dpi": 180,
    "trajectory_arrow_length": 0.25,
    "boundary_enforced": True,  # Reject translations whose full path leaves the inset region.
    "boundary_preview": True,  # Show the boundary on the local top-down chart.
    "boundary_cell_size": 0.05,
    "boundary_close_radius": 0.15,
    "camera_radius": 0.15,
    "point_radius": 0.015,
    "move_step": 0.3,
    "look_step": 15,
    "up": None,  # Scene up for ground movement, yaw, and the map; None estimates it from the scene.
}
OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")
ACTIONS = {"move_forward", "move_backward", "move_left", "move_right",
           "turn_left", "turn_right", "turn_up", "turn_down"}
HELP = """Enter one command per line:
  move_forward [distance]  Walk forward horizontally, e.g. move_forward 0.3.
  move_backward [distance] Walk backward horizontally without turning.
  move_left [distance]     Strafe left relative to the current horizontal heading.
  move_right [distance]    Strafe right relative to the current horizontal heading.
  turn_left [degrees]      Turn left, e.g. turn_left 30.
  turn_right [degrees]     Turn right.
  turn_up [degrees]        Look up without changing the walking direction.
  turn_down [degrees]      Look down without changing the walking direction.
  observe                 Save another observation without moving.
  /new                    Start a new output directory at the initial camera pose.
  help                    Show these commands.
  /exit                   Quit (also: exit, quit).
Omit the magnitude to use move_step or look_step from SCENE_CONFIG.
All translation paths must stay within the inset scene boundary when boundary_enforced is enabled.
Blocked moves keep the camera in place; turns remain available.
Each observation saves a first-person PNG, camera metadata, and a trajectory chart."""


def new_session_dir() -> Path:
    return (Path(OUTPUT_DIR).expanduser().resolve()
            / datetime.now().strftime("env_%Y%m%d_%H%M%S_%f"))


def show_observation(record):
    folder = Path(record["image"]).parent
    if record.get('blocked'):
        print(f"Blocked: {record['reason']} Requested: {record['magnitude']:g} m; moved: 0 m.", flush=True)
    print(f"First person: {record['image']}\nTop down: {folder / 'top_down.png'}", flush=True)


def run_environment(camera):
    print(HELP, flush=True)
    show_observation(camera.observe())
    while True:
        try:
            command = input("camera> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if command in {"/exit", "exit", "quit"}:
            break
        if not command:
            continue
        try:
            if command == "help":
                print(HELP, flush=True)
                continue
            if command == "/new":
                camera.start_session(new_session_dir())
                show_observation(camera.observe())
            elif command == "observe":
                show_observation(camera.observe())
            else:
                parts = command.split()
                if parts[0] not in ACTIONS or len(parts) > 2:
                    raise ValueError("Use an action name and an optional magnitude. Enter help for commands.")
                magnitude = float(parts[1]) if len(parts) == 2 else None
                if magnitude is not None and (not math.isfinite(magnitude) or magnitude <= 0):
                    raise ValueError("Magnitude must be a positive finite number.")
                show_observation(camera.action(parts[0], magnitude))
        except (ValueError, RuntimeError, OSError) as error:
            print(f"Error: {error}", flush=True)


def main():
    config = {
        **SCENE_CONFIG,
        "scene_dir": str(Path(SCENE_CONFIG["scene_dir"]).expanduser().resolve()),
    }
    output = new_session_dir()
    print(f"Scene: {config['scene_dir']}\nOutput: {output}", flush=True)
    with CameraClient(config["blender"], config, output, timeout=config["timeout"]) as camera:
        run_environment(camera)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
