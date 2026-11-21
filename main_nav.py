"""Run one image-goal navigation task from a recorded frame's camera pose."""

from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import random
import numpy as np

from agent import Agent
from agent.models import create_model, model_config_snapshot
from agent.tools import create_tools
from scene.client import CameraClient, reference_camera_state

DATA_ROOT = Path('/data4/haibo/data/3D_Env')
BENCHMARK_JSON = Path(__file__).resolve().parent / 'outputs' / 'sub_benchmark_test.json'
GOAL_MODE = 'both'  # 'first_person', 'top_down', or 'both'.
OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")

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
        "top_p": 1.0,
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



SYSTEM_PROMPT = """You are a spatial agent exploring a reconstructed 3D point-cloud scene.
Use the available tools to gather evidence and complete the user's task.

Visual evidence
Inspect each new image before continuing, using its view label and any legend to interpret it.
Images describe the viewpoint when obtained. Point-cloud gaps may be reconstruction artifacts.
State uncertainty when the visual evidence is insufficient, and answer once you have enough evidence."""




def select_navigation_tasks(tasks, mode):
    """Filter one modality or return both variants consecutively for every pair."""
    if mode not in {'first_person', 'top_down', 'both'}:
        raise ValueError('GOAL_MODE must be first_person, top_down, or both')
    pairs = {}
    for task in tasks:
        kind = task.get('goal_type', 'first_person')
        if kind not in {'first_person', 'top_down'}:
            raise ValueError(f'Unsupported goal_type: {kind!r}')
        key = task['scene'], task.get('pair_id', task['id'])
        variants = pairs.setdefault(key, {})
        if kind in variants:
            raise ValueError(f'Duplicate {kind} task for pair {key}')
        variants[kind] = task
    if mode == 'both':
        for key, variants in pairs.items():
            if set(variants) != {'first_person', 'top_down'}:
                raise ValueError(f'Both goal variants are required for pair {key}')
            a, b = variants['first_person'], variants['top_down']
            if any(a[field] != b[field] for field in ['initial_view', 'target_view']):
                raise ValueError(f'Sibling endpoints differ: {key}')
        selected = [variants[kind] for variants in pairs.values() for kind in ['first_person', 'top_down']]
    else:
        selected = [task for task in tasks if task.get('goal_type', 'first_person') == mode]
    if not selected:
        raise ValueError(f'No tasks available for GOAL_MODE={mode!r}')
    return selected


def navigation_goal(task, data_root, target_path):
    goal_type = task.get('goal_type', 'first_person')
    metrics = {'first_person': 'forward_angle', 'top_down': 'horizontal_heading'}
    if goal_type not in metrics:
        raise ValueError(f'Unsupported goal_type: {goal_type!r}')
    metric = metrics[goal_type]
    if task.get('direction_metric', metric) != metric:
        raise ValueError(f'Inconsistent direction_metric for {goal_type}')
    metadata = {'goal_type': goal_type, 'direction_metric': metric,
                'pair_id': task.get('pair_id', task.get('id'))}
    path = Path(target_path).resolve()
    if goal_type == 'top_down':
        if not task.get('goal_image'):
            raise ValueError('A top_down goal requires goal_image')
        path = Path(task['goal_image']).expanduser()
        path = (Path(data_root).expanduser() / path).resolve()
        up = np.asarray(task.get('scene_up'), dtype=float)
        if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-6:
            raise ValueError('A top_down goal requires a finite, nonzero scene_up vector')
        metadata['scene_up'] = (up / np.linalg.norm(up)).tolist()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, metadata


def init_seeds(seed=42,):
    random.seed(seed)
    np.random.seed(seed)

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

test_data = select_navigation_tasks(load_json(BENCHMARK_JSON), GOAL_MODE)

init_seeds(42)
pool = [task for task in test_data if task.get('goal_type', 'first_person') == 'first_person'] if GOAL_MODE == 'both' else test_data
sample = random.choice(pool)
samples = ([task for task in test_data
            if (task['scene'], task.get('pair_id', task['id'])) ==
               (sample['scene'], sample.get('pair_id', sample['id']))]
           if GOAL_MODE == 'both' else [sample])
print(sample)


SCENE_CONFIG = {
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
PROMPT = """Navigate from the initial view to the goal.
Goal type: {goal_type}.
The initial view is a reference first-person video frame. The camera starts at its recorded pose.
The goal image is a reference, not a current observation.
{goal_instruction}
Use the camera tools to inspect the current rendered first-person scene and navigate toward the goal.
Rendered observations may contain point-cloud gaps and a different image crop from the reference video.
When you believe you have reached the goal, report success with visual evidence.
If you cannot reach it, report failure with a reason.
If finish_task is available, use it to report the outcome and end the task; otherwise respond in text."""
GOAL_PROMPTS = {
    'first_person': """The goal image is a reference first-person video frame.
Match both the camera location and the full viewing direction shown in that image.
Finding the same object alone does not establish that the camera viewpoint matches.""",
    'top_down': """The goal image is a static top-down map.
The green Start marker and arrow show your initial position and horizontal heading.
The orange Target marker and arrow show the goal position and required horizontal heading.
Reach the target position and match the target arrow's horizontal heading.
The map remains fixed while you move. Use new first-person observations to track your progress.
Pitch and roll are not evaluated for this goal type.""",
}
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
    'enable_actions': True,
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


def run_task(task):
    scene_dir = (Path(DATA_ROOT).expanduser() / task['scene']).resolve()
    cameras = json.loads((scene_dir / 'cameras.json').read_text())
    initial_path, initial_camera = reference_frame(scene_dir, task['initial_view'], cameras)
    target_path, goal_camera = reference_frame(scene_dir, task['target_view'], cameras)
    goal_path, goal_metadata = navigation_goal(task, DATA_ROOT, target_path)
    goal_type = goal_metadata['goal_type']
    prompt = PROMPT.format(goal_type=goal_type, goal_instruction=GOAL_PROMPTS[goal_type])
    config = {**SCENE_CONFIG, 'scene_dir': str(scene_dir),
              'initial_camera': initial_camera, 'target_camera': goal_camera}
    output = (Path(OUTPUT_DIR).expanduser().resolve()
              / f'nav_{scene_dir.name}_{MODEL}_{goal_type}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    output.mkdir(parents=True, exist_ok=False)
    initial_copy = output / f'initial_view{initial_path.suffix.lower()}'
    target_copy = output / f'target_view{target_path.suffix.lower()}'
    goal_copy = target_copy if goal_type == 'first_person' else output / f'goal_image{goal_path.suffix.lower()}'
    shutil.copy2(initial_path, initial_copy)
    shutil.copy2(target_path, target_copy)
    if goal_copy != target_copy:
        shutil.copy2(goal_path, goal_copy)
    content = [
        {'type': 'text', 'text': '<initial_view>'},
        image_content(initial_copy),
        {'type': 'text', 'text': '</initial_view>\n<goal_view>'},
        image_content(goal_copy),
        {'type': 'text', 'text': f'</goal_view>\n{prompt}'},
    ]
    # Camera parameters are local metadata; the model receives images and task text only.
    (output / 'task.json').write_text(json.dumps({
        **goal_metadata, 'benchmark_task': task, 'scene_dir': str(scene_dir), 'initial_view': str(initial_copy),
        'goal_view': str(goal_copy), 'goal_image': str(goal_copy), 'target_view': str(target_copy),
        'prompt': prompt, 'initial_camera': initial_camera, 'goal_camera': goal_camera,
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
        target_camera = reference_camera_state(goal_camera, target_copy)
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
                result = {**result, **goal_metadata,
                          'final_camera': final_camera_state(output), 'target_camera': target_camera}
                temporary = output / 'result.json.tmp'
                temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
                temporary.replace(output / 'result.json')
        finally:
            if backend is not None:
                backend.client.close()
    return 0


def main():
    exit_code = 0
    for task in samples:
        try:
            run_task(task)
        except Exception as error:
            if len(samples) == 1:
                raise
            print(f'Navigation failed for {task["id"]}: {error}', file=sys.stderr, flush=True)
            exit_code = 1
    return exit_code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
