"""Run one image-goal navigation task from a recorded frame's camera pose."""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import random
import numpy as np

from agent import Agent
from agent.models import create_model, model_config_snapshot
from agent.tools import create_tools
from scene.client import CameraClient, reference_camera_state


MODEL = "qwen3.8-flash-next"

MODEL_GENERATION_CONFIGS = {
    "glm-5.3-flash": {
        "server_config": {
            "base_url": "http://127.0.0.1:8002/v1",
            "timeout": 300,
        },
        "max_tokens": 16 * 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 0,
        "reasoning_effort": "max",  # "low", "high", "max"
        "preserve_thinking": True,
    },
    "deepseek-v4.1-flash": {
        "server_config": {
            "base_url": "http://127.0.0.1:8003/v1",
            "timeout": 300,
        },
        "max_tokens": 16 * 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 0,
        "reasoning_effort": 75,  # low=25, high=50, xhigh=75, max=100; accepts 1..100.
    },
    "qwen3.5-9b": {
        "server_config": {
            "base_url": 'http://127.0.0.1:8001/v1',
            "timeout": 180,
        },
        "max_tokens": 16 * 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "enable_thinking": True,
    },
    "qwen3.8-27b": {
        "server_config": {
            "base_url": 'http://127.0.0.1:8000/v1',
            "timeout": 180,
        },
        "max_tokens": 16 * 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "enable_thinking": True,
        "reasoning_effort": "medium",  # "low", "medium", "xhigh"
        "preserve_thinking": True,
    },
    "qwen3.8-flash-next": {
        "server_config": {
            "base_url": 'http://127.0.0.1:8000/v1',
            "timeout": 180,
        },
        "max_tokens": 16 * 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "enable_thinking": True,
        "reasoning_effort": "medium",  # "low", "medium", "xhigh"
        "preserve_thinking": True,
    },
    "gpt-6-astra": {
        "server_config": {
            "base_url": 'https://api.openai.com/v1',
            "timeout": 300,
        },
        "max_output_tokens": 32 * 1024,
        "reasoning_effort": "medium", # "low", "medium", "high", "xhigh", "max"
        "reasoning_summary": "auto",
    },
    "gpt-6.1-sol": {
        "server_config": {
            "base_url": 'https://api.openai.com/v1',
            "timeout": 300,
        },
        "max_output_tokens": 16 * 1024,
        "reasoning_effort": "medium",  # "low", "medium", "high", "xhigh", "max"
        "reasoning_summary": "auto",
    },
    "gpt-6.1-terra": {
        "server_config": {
            "base_url": 'https://api.openai.com/v1',
            "timeout": 300,
        },
        "max_output_tokens": 16 * 1024,
        "reasoning_effort": "medium", # "low", "medium", "high", "xhigh", "max"
        "reasoning_summary": "auto",
    },
}


OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")

SYSTEM_PROMPT = """You are a spatial agent exploring a reconstructed 3D point-cloud scene.
Use the available tools to gather evidence and complete the user's task.

Visual evidence
Inspect each new image before continuing, using its view label and any legend to interpret it.
Images describe the viewpoint when obtained. Point-cloud gaps may be reconstruction artifacts.
State uncertainty when the visual evidence is insufficient, and answer once you have enough evidence."""

BENCHMARK_JSON = Path(__file__).resolve().parent / 'outputs' / 'benchmark_test.json'


def init_seeds(seed=42,):
    random.seed(seed)
    np.random.seed(seed)

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

test_data = load_json(BENCHMARK_JSON)

init_seeds(42)
sample = random.choice(test_data)
print(sample)


SCENE_CONFIG = {
    "scene_dir": os.path.join('/data4/haibo/data/3D_Env', sample['scene']),
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
    "min_camera_height": 0.15,  # Camera-center height above the estimated floor, in meters.
    "max_camera_height": 2.5,
    "point_radius": 0.015,
    "move_step": 0.3,
    "look_step": 15,
    "up": None,  # Scene up for ground movement, yaw, and the map; None estimates it from the scene.
}
# Example frames. Paths may be absolute or relative to scene_dir.
INITIAL_VIEW = sample['initial_view']
GOAL_VIEW = sample['target_view']
PROMPT = """Navigate from the initial view to the location and viewing direction shown in the goal view.
The camera starts at the recorded pose of the initial view. The goal view is a reference, not a current observation.
Use the camera tools to inspect the current rendered scene and navigate toward the goal.
Match both the viewpoint and viewing direction, rather than merely finding the same object.
The reference images are video frames; rendered observations may contain point-cloud gaps and a different image crop.
When you believe you have reached the goal viewpoint, report success with visual evidence.
If you cannot reach it, report failure with a reason.
If finish_task is available, use it to report the outcome and end the task; otherwise respond in text."""
AGENT_CONFIG = {
    "max_steps": 50,
    "max_conversation_images": 7,
    "verbose": True,  # Print tool calls, OpenAI reasoning summaries and usage.
    'preserve_prompt_images': True,  # Retain both references while trimming older observations.
}
TOOL_CONFIG = {
    'enable_observe': True,
    'enable_finish_task': True,
    'enable_action': True,
    'enable_reset': True,
    'enable_turn_left': True,
    'enable_turn_right': True,
    'enable_turn_up': True,
    'enable_turn_down': True,
    'enable_move_forward': True,
    'enable_move_backward': False,
    'enable_move_left': False,
    'enable_move_right': False,
    'enable_move_up': False,
    'enable_move_down': False,
    'enable_top_down': False,  # Optional map tool for navigation experiments.
}


def reference_frame(scene_dir, image, cameras):
    path = Path(image).expanduser()
    path = (path if path.is_absolute() else scene_dir / path).resolve()
    if not path.is_relative_to((scene_dir / 'frames').resolve()):
        raise ValueError(f'Reference image must be inside the scene frames directory: {path}')
    if not path.is_file():
        raise FileNotFoundError(path)
    matches = [frame for frame in cameras
               if (scene_dir / frame['image_path']).resolve() == path]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one camera record for {path}; found {len(matches)}')
    return path, matches[0]


def image_content(path):
    return {'type': 'image_url', 'image_url': {'url': str(path.resolve())}}


def final_camera_state(output):
    """Read the last committed camera observation without requesting another render."""
    trajectory = output / 'trajectory.json'
    if not trajectory.is_file():
        return None
    observations = json.loads(trajectory.read_text())
    if not observations:
        return None
    metadata = json.loads(Path(observations[-1]['image']).with_suffix('.json').read_text())
    return {key: metadata[key] for key in (
        'observation_id', 'image', 'position', 'forward', 'up', 'heading',
        'height_above_floor', 'intrinsics',
    )}


def main():
    scene_dir = Path(SCENE_CONFIG['scene_dir']).expanduser().resolve()
    cameras = json.loads((scene_dir / 'cameras.json').read_text())
    initial_path, initial_camera = reference_frame(scene_dir, INITIAL_VIEW, cameras)
    goal_path, goal_camera = reference_frame(scene_dir, GOAL_VIEW, cameras)
    config = {**SCENE_CONFIG, 'scene_dir': str(scene_dir),
              'initial_camera': initial_camera, 'target_camera': goal_camera}
    output = (Path(OUTPUT_DIR).expanduser().resolve()
              / f'nav_{scene_dir.name}_{MODEL}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    output.mkdir(parents=True, exist_ok=False)
    initial_copy = output / f'initial_view{initial_path.suffix.lower()}'
    goal_copy = output / f'target_view{goal_path.suffix.lower()}'
    shutil.copy2(initial_path, initial_copy)
    shutil.copy2(goal_path, goal_copy)
    content = [
        {'type': 'text', 'text': '<initial_view>'},
        image_content(initial_copy),
        {'type': 'text', 'text': '</initial_view>\n<goal_view>'},
        image_content(goal_copy),
        {'type': 'text', 'text': f'</goal_view>\n{PROMPT}'},
    ]
    # Camera parameters are local metadata; the model receives images and task text only.
    (output / 'task.json').write_text(json.dumps({
        'scene_dir': str(scene_dir), 'initial_view': str(initial_copy), 'goal_view': str(goal_copy),
        'prompt': PROMPT, 'initial_camera': initial_camera, 'goal_camera': goal_camera,
        'tool_config': TOOL_CONFIG,
        'model': MODEL, 'model_config': model_config_snapshot(MODEL_GENERATION_CONFIGS[MODEL]),
        'agent_config': AGENT_CONFIG,
    }, indent=2) + '\n')

    model_config = MODEL_GENERATION_CONFIGS[MODEL]
    backend = None
    agent = None
    result = None
    target_camera = None
    try:
        target_camera = reference_camera_state(goal_camera, goal_copy)
        backend, protocol = create_model(MODEL, model_config)
        with CameraClient(config['blender'], config, output, timeout=config['timeout']) as camera:
            camera.observe()  # Record the actual starting pose and its local trajectory map.
            agent = Agent(
                backend,
                protocol,
                create_tools(camera, move_step=config['move_step'], look_step=config['look_step'], **TOOL_CONFIG),
                system_prompt=SYSTEM_PROMPT, **AGENT_CONFIG,
                conversation_path=output / 'conversation.json',
            )
            print(f'Agent model: {MODEL}\nInitial view: {initial_path}\nGoal view: {goal_path}\nOutput: {output}',
                  flush=True)
            print('Agent:', agent.run(content), flush=True)
    except (Exception, KeyboardInterrupt) as error:
        result = (agent.result if agent is not None else None) or {
            'status': None, 'reason': f'{type(error).__name__}: {error}',
            'termination': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'error',
        }
        raise
    finally:
        try:
            result = result or (agent.result if agent is not None else None)
            if result is not None:
                result = {**result, 'final_camera': final_camera_state(output), 'target_camera': target_camera}
                temporary = output / 'result.json.tmp'
                temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
                temporary.replace(output / 'result.json')
        finally:
            if backend is not None:
                backend.client.close()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
