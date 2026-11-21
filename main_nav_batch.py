"""Run every benchmark navigation task sequentially, checkpointing each result."""

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

from tqdm import tqdm
import numpy as np

from agent import Agent
from agent.models import create_model, model_config_snapshot
from agent.protocols import MODEL_ADAPTERS
from agent.tools import create_tools
from scene.client import CameraClient, reference_camera_state


DATA_ROOT = Path('/data4/haibo/data/3D_Env')
BENCHMARK_JSON = Path(__file__).resolve().parent / 'outputs' / 'sub_benchmark_test.json'
GOAL_MODE = 'both'  # 'first_person', 'top_down', or 'both'.
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
RESUME_DIR = None  # Set to a previous nav_batch_* directory to continue it.

MODEL = "gpt-6.1-terra"

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
        "reasoning_effort": "high", # "low", "medium", "high", "xhigh", "max"
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

SYSTEM_PROMPT = """You are a spatial agent exploring a reconstructed 3D point-cloud scene.
Use the available tools to gather evidence and complete the user's task.

Visual evidence
Inspect each new image before continuing, using its view label and any legend to interpret it.
Images describe the viewpoint when obtained. Point-cloud gaps may be reconstruction artifacts.
State uncertainty when the visual evidence is insufficient, and answer once you have enough evidence."""

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


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def task_key(task):
    return task['scene'], task['id']


def load_benchmark():
    tasks = json.loads(Path(BENCHMARK_JSON).read_text())
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Benchmark must be a non-empty list of tasks')
    keys = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError('Each benchmark task must be an object')
        for field in ('id', 'scene', 'initial_view', 'target_view'):
            if not isinstance(task.get(field), str) or not task[field]:
                raise ValueError(f'Each benchmark task needs a non-empty {field}')
        if task.get('goal_type', 'first_person') not in GOAL_PROMPTS:
            raise ValueError(f'Unsupported goal_type: {task.get("goal_type")!r}')
        scene = Path(task['scene'])
        if scene.is_absolute() or len(scene.parts) != 2 or '..' in scene.parts:
            raise ValueError(f'Expected a source/scene path: {task["scene"]}')
        if Path(task['id']).name != task['id'] or task['id'] in {'.', '..'}:
            raise ValueError(f'Invalid task ID: {task["id"]}')
        if task_key(task) in keys:
            raise ValueError(f'Duplicate benchmark task: {task_key(task)}')
        keys.add(task_key(task))
    return select_navigation_tasks(tasks, GOAL_MODE)


def reference_frame(scene_dir, image, cameras):
    path = Path(image).expanduser()
    path = (path if path.is_absolute() else scene_dir / path).resolve()
    if not path.is_relative_to((scene_dir / 'frames').resolve()):
        raise ValueError(f'Reference image must be inside the scene frames directory: {path}')
    if not path.is_file():
        raise FileNotFoundError(path)
    matches = [frame for frame in cameras if (scene_dir / frame['image_path']).resolve() == path]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one camera record for {path}; found {len(matches)}')
    return path, matches[0]


def final_camera_state(output):
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


def run_task(task, output, backend, protocol):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    agent = None
    result = None
    target_camera = None
    goal_metadata = {'goal_type': task.get('goal_type', 'first_person'),
                     'pair_id': task.get('pair_id', task['id'])}
    try:
        scene_dir = (Path(DATA_ROOT).expanduser() / task['scene']).resolve()
        task_data = {
            'benchmark_task': task, 'scene_dir': str(scene_dir),
            'prompt': PROMPT.format(goal_type=goal_metadata['goal_type'],
                                    goal_instruction=GOAL_PROMPTS[goal_metadata['goal_type']]),
            'model': MODEL, 'model_config': model_config_snapshot(MODEL_GENERATION_CONFIGS[MODEL]),
            'agent_config': AGENT_CONFIG, 'tool_config': TOOL_CONFIG,
        }
        write_json(output / 'task.json', task_data)
        cameras = json.loads((scene_dir / 'cameras.json').read_text())
        initial_path, initial_camera = reference_frame(scene_dir, task['initial_view'], cameras)
        target_path, goal_camera = reference_frame(scene_dir, task['target_view'], cameras)
        target_camera = reference_camera_state(goal_camera, target_path)
        goal_path, goal_metadata = navigation_goal(task, DATA_ROOT, target_path)
        config = {**SCENE_CONFIG, 'scene_dir': str(scene_dir),
                  'initial_camera': initial_camera, 'target_camera': goal_camera}
        initial_copy = output / f'initial_view{initial_path.suffix.lower()}'
        target_copy = output / f'target_view{target_path.suffix.lower()}'
        goal_copy = target_copy if goal_metadata['goal_type'] == 'first_person' else output / f'goal_image{goal_path.suffix.lower()}'
        shutil.copy2(initial_path, initial_copy)
        shutil.copy2(target_path, target_copy)
        if goal_copy != target_copy:
            shutil.copy2(goal_path, goal_copy)
        target_camera['image'] = str(target_copy)
        write_json(output / 'task.json', {
            **task_data, **goal_metadata, 'initial_view': str(initial_copy),
            'goal_view': str(goal_copy), 'goal_image': str(goal_copy), 'target_view': str(target_copy),
            'initial_camera': initial_camera, 'goal_camera': goal_camera,
        })
        content = [
            {'type': 'text', 'text': '<initial_view>'},
            {'type': 'image_url', 'image_url': {'url': str(initial_copy)}},
            {'type': 'text', 'text': '</initial_view>\n<goal_view>'},
            {'type': 'image_url', 'image_url': {'url': str(goal_copy)}},
            {'type': 'text', 'text': f'</goal_view>\n{task_data["prompt"]}'},
        ]
        with CameraClient(config['blender'], config, output, timeout=config['timeout']) as camera:
            camera.observe()
            agent = Agent(
                backend, protocol,
                create_tools(camera, move_step=config['move_step'], look_step=config['look_step'], **TOOL_CONFIG),
                system_prompt=SYSTEM_PROMPT, **AGENT_CONFIG,
                conversation_path=output / 'conversation.json',
            )
            agent.run(content)
        result = agent.result
    except KeyboardInterrupt:
        result = {'status': None, 'reason': 'Navigation interrupted by the user.', 'termination': 'interrupted'}
        raise
    except Exception as error:
        result = {
            'status': None, 'reason': f'{type(error).__name__}: {error}', 'termination': 'error',
        }
    finally:
        if result is not None:
            result = {**result, **goal_metadata, 'id': task['id'], 'scene': task['scene'], 'model': MODEL,
                      'output_dir': str(output), 'elapsed_seconds': round(time.perf_counter() - started, 3),
                      'target_camera': target_camera}
            try:
                result['final_camera'] = final_camera_state(output)
            except Exception as error:
                result['final_camera'] = None
                result['camera_state_error'] = f'{type(error).__name__}: {error}'
            write_json(output / 'result.json', result)
    return result


def checkpoint(output, tasks, records):
    ordered = [records[task_key(task)] for task in tasks if task_key(task) in records]
    def counts(selected):
        items = [records[task_key(task)] for task in selected if task_key(task) in records]
        finished = [record for record in items if record['termination'] != 'interrupted']
        successes = sum(record['status'] == 'success' for record in finished)
        return {
            'total': len(selected), 'finished': len(finished),
            'pending': len(selected) - len(finished), 'complete': len(finished) == len(selected),
            'status_counts': dict(Counter(record['status'] or 'unreported' for record in finished)),
            'termination_counts': dict(Counter(record['termination'] for record in items)),
            'self_reported_success_rate': successes / len(finished) if finished else None,
        }
    summary = {
        'model': MODEL, 'goal_mode': GOAL_MODE, **counts(tasks),
        'by_goal_type': {kind: counts([task for task in tasks if task.get('goal_type', 'first_person') == kind])
                         for kind in ['first_person', 'top_down']
                         if any(task.get('goal_type', 'first_person') == kind for task in tasks)},
    }
    write_json(output / 'results.json', ordered)
    write_json(output / 'summary.json', summary)
    return summary


def main():
    tasks = load_benchmark()
    settings = {
        'model': MODEL, 'goal_mode': GOAL_MODE,
        'model_config': model_config_snapshot(MODEL_GENERATION_CONFIGS[MODEL]),
        'data_root': str(Path(DATA_ROOT).expanduser().resolve()),
        'scene_config': SCENE_CONFIG, 'agent_config': AGENT_CONFIG, 'tool_config': TOOL_CONFIG,
        'system_prompt': SYSTEM_PROMPT, 'prompt': PROMPT, 'goal_prompts': GOAL_PROMPTS,
    }
    settings = json.loads(json.dumps(settings, allow_nan=False))
    records = {}
    if RESUME_DIR is None:
        output = Path(OUTPUT_DIR).expanduser().resolve() / f'nav_batch_{MODEL}_{GOAL_MODE}_{datetime.now():%Y%m%d_%H%M%S_%f}'
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'run.json', {'benchmark': str(Path(BENCHMARK_JSON).resolve()), 'settings': settings})
        write_json(output / 'benchmark.json', tasks)
    else:
        output = Path(RESUME_DIR).expanduser().resolve()
        saved_settings = json.loads((output / 'run.json').read_text())['settings']
        saved_tasks = json.loads((output / 'benchmark.json').read_text())
        if 'goal_mode' not in saved_settings:
            kinds = {task.get('goal_type', 'first_person') for task in saved_tasks}
            saved_settings['goal_mode'] = next(iter(kinds)) if len(kinds) == 1 else 'both'
        if 'server_config' in saved_settings:  # Accept snapshots from before per-model server settings.
            backend_name = MODEL_ADAPTERS[saved_settings['model']]['backend']
            saved_settings['model_config'] = {**saved_settings['model_config'],
                                             'server_config': saved_settings.pop('server_config')[backend_name]}
        saved_settings['tool_config'].setdefault('enable_actions', False)
        if saved_settings != settings:
            raise ValueError('Resume settings differ from run.json; restore the original configuration')
        if saved_tasks != tasks:
            raise ValueError('Benchmark differs from the saved benchmark.json')
        keys = {task_key(task) for task in tasks}
        # Recover even if a process stopped after saving a task but before checkpointing.
        paths = (output / 'tasks').glob('*/*/*/result.json')
        for path in sorted(paths, key=lambda path: path.stat().st_mtime_ns):
            record = json.loads(path.read_text())
            if task_key(record) in keys:
                if 'target_camera' not in record:
                    task_data = json.loads((path.parent / 'task.json').read_text())
                    goal = task_data.get('goal_camera')
                    target_image = task_data.get('target_view', task_data['goal_view'])
                    record['target_camera'] = reference_camera_state(goal, target_image) if goal else None
                    write_json(path, record)
                records[task_key(record)] = record
    summary = checkpoint(output, tasks, records)
    print(f'Batch model: {MODEL}\nBenchmark: {BENCHMARK_JSON}\nOutput: {output}', flush=True)
    if summary['complete']:
        print('All tasks already finished.', flush=True)
        return 0

    backend = None
    try:
        backend, protocol = create_model(MODEL, MODEL_GENERATION_CONFIGS[MODEL])
        with tqdm(total=len(tasks), initial=summary['finished'], desc='Navigation tasks', unit='task') as progress:
            for task in tasks:
                previous = records.get(task_key(task))
                if previous is not None and previous['termination'] != 'interrupted':
                    continue
                task_output = (output / 'tasks' / task['scene']
                               / f'{task["id"]}_{MODEL}_{datetime.now():%Y%m%d_%H%M%S_%f}')
                tqdm.write(f'Running {task["scene"]} / {task["id"]}')
                try:
                    result = run_task(task, task_output, backend, protocol)
                except KeyboardInterrupt:
                    result_path = task_output / 'result.json'
                    if result_path.is_file():
                        records[task_key(task)] = json.loads(result_path.read_text())
                    raise
                records[task_key(task)] = result
                summary = checkpoint(output, tasks, records)
                progress.update(1)
                tqdm.write(f'{task["id"]}: {result["status"] or "unreported"} ({result["termination"]})')
    except KeyboardInterrupt:
        checkpoint(output, tasks, records)
        print(f'Interrupted. Set RESUME_DIR to {str(output)!r} to continue.', flush=True)
        return 130
    finally:
        if backend is not None:
            backend.client.close()
    print(f'Finished {summary["finished"]}/{summary["total"]}; summary: {output / "summary.json"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
