"""Run every benchmark navigation task sequentially, checkpointing each result."""

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

from tqdm import tqdm

from agent import Agent
from agent.models import create_model
from agent.tools import create_tools
from scene.client import CameraClient, reference_camera_state


DATA_ROOT = Path('/data4/haibo/data/3D_Env')
BENCHMARK_JSON = Path(__file__).resolve().parent / 'outputs' / 'sub_benchmark_test.json'
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
RESUME_DIR = None  # Set to a previous nav_batch_* directory to continue it.

MODEL = "gpt-6.1-terra"

MODEL_GENERATION_CONFIGS = {
    "qwen3.8-27b": {
        "max_tokens": 16 * 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "enable_thinking": True,
        "reasoning_effort": "medium",  # "low", "medium", "xhigh"
        "preserve_thinking": True,
    },
    "qwen3.8-flash-next": {
        "max_tokens": 16 * 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "enable_thinking": True,
        "reasoning_effort": "medium",  # "low", "medium", "xhigh"
        "preserve_thinking": True,
    },
    "gpt-6-astra": {
        "max_output_tokens": 32 * 1024,
        "reasoning_effort": "high", # "low", "medium", "high", "xhigh", "max"
        "reasoning_summary": "auto",
    },
    "gpt-6.1-sol": {
        "max_output_tokens": 16 * 1024,
        "reasoning_effort": "medium",  # "low", "medium", "high", "xhigh", "max"
        "reasoning_summary": "auto",
    },
    "gpt-6.1-terra": {
        "max_output_tokens": 16 * 1024,
        "reasoning_effort": "medium", # "low", "medium", "high", "xhigh", "max"
        "reasoning_summary": "auto",
    },
}

MODEL_SERVER_CONFIG = {
    "vllm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "timeout": 180,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "timeout": 300,
        # api_key is read from OPENAI_API_KEY by the SDK.
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

PROMPT = """Navigate from the initial view to the location and viewing direction shown in the goal view.
The camera starts at the recorded pose of the initial view. The goal view is a reference, not a current observation.
Use the camera tools to inspect the current rendered scene and navigate toward the goal.
Match both the viewpoint and viewing direction, rather than merely finding the same object.
The reference images are video frames; rendered observations may contain point-cloud gaps and a different image crop.
When you believe you have reached the goal viewpoint, report success with visual evidence.
If you cannot reach it, report failure with a reason.
If finish_task is available, use it to report the outcome and end the task; otherwise respond in text."""


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
        scene = Path(task['scene'])
        if scene.is_absolute() or len(scene.parts) != 2 or '..' in scene.parts:
            raise ValueError(f'Expected a source/scene path: {task["scene"]}')
        if Path(task['id']).name != task['id'] or task['id'] in {'.', '..'}:
            raise ValueError(f'Invalid task ID: {task["id"]}')
        if task_key(task) in keys:
            raise ValueError(f'Duplicate benchmark task: {task_key(task)}')
        keys.add(task_key(task))
    return tasks


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
    try:
        scene_dir = (Path(DATA_ROOT).expanduser() / task['scene']).resolve()
        task_data = {
            'benchmark_task': task, 'scene_dir': str(scene_dir), 'prompt': PROMPT,
            'model': MODEL, 'model_config': MODEL_GENERATION_CONFIGS[MODEL],
            'agent_config': AGENT_CONFIG, 'tool_config': TOOL_CONFIG,
        }
        write_json(output / 'task.json', task_data)
        cameras = json.loads((scene_dir / 'cameras.json').read_text())
        initial_path, initial_camera = reference_frame(scene_dir, task['initial_view'], cameras)
        goal_path, goal_camera = reference_frame(scene_dir, task['target_view'], cameras)
        target_camera = reference_camera_state(goal_camera, goal_path)
        config = {**SCENE_CONFIG, 'scene_dir': str(scene_dir),
                  'initial_camera': initial_camera, 'target_camera': goal_camera}
        initial_copy = output / f'initial_view{initial_path.suffix.lower()}'
        goal_copy = output / f'target_view{goal_path.suffix.lower()}'
        shutil.copy2(initial_path, initial_copy)
        shutil.copy2(goal_path, goal_copy)
        target_camera['image'] = str(goal_copy)
        write_json(output / 'task.json', {
            **task_data, 'initial_view': str(initial_copy), 'goal_view': str(goal_copy),
            'initial_camera': initial_camera, 'goal_camera': goal_camera,
        })
        content = [
            {'type': 'text', 'text': '<initial_view>'},
            {'type': 'image_url', 'image_url': {'url': str(initial_copy)}},
            {'type': 'text', 'text': '</initial_view>\n<goal_view>'},
            {'type': 'image_url', 'image_url': {'url': str(goal_copy)}},
            {'type': 'text', 'text': f'</goal_view>\n{PROMPT}'},
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
            result = {**result, 'id': task['id'], 'scene': task['scene'], 'model': MODEL,
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
    finished = [record for record in ordered if record['termination'] != 'interrupted']
    successes = sum(record['status'] == 'success' for record in finished)
    summary = {
        'model': MODEL, 'total': len(tasks), 'finished': len(finished),
        'pending': len(tasks) - len(finished), 'complete': len(finished) == len(tasks),
        'status_counts': dict(Counter(record['status'] or 'unreported' for record in finished)),
        'termination_counts': dict(Counter(record['termination'] for record in ordered)),
        'self_reported_success_rate': successes / len(finished) if finished else None,
    }
    write_json(output / 'results.json', ordered)
    write_json(output / 'summary.json', summary)
    return summary


def main():
    tasks = load_benchmark()
    settings = {
        'model': MODEL, 'model_config': MODEL_GENERATION_CONFIGS[MODEL],
        'server_config': {name: {key: value for key, value in config.items() if key != 'api_key'}
                          for name, config in MODEL_SERVER_CONFIG.items()},
        'data_root': str(Path(DATA_ROOT).expanduser().resolve()),
        'scene_config': SCENE_CONFIG, 'agent_config': AGENT_CONFIG, 'tool_config': TOOL_CONFIG,
        'system_prompt': SYSTEM_PROMPT, 'prompt': PROMPT,
    }
    settings = json.loads(json.dumps(settings, allow_nan=False))
    records = {}
    if RESUME_DIR is None:
        output = Path(OUTPUT_DIR).expanduser().resolve() / f'nav_batch_{MODEL}_{datetime.now():%Y%m%d_%H%M%S_%f}'
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'run.json', {'benchmark': str(Path(BENCHMARK_JSON).resolve()), 'settings': settings})
        write_json(output / 'benchmark.json', tasks)
    else:
        output = Path(RESUME_DIR).expanduser().resolve()
        if json.loads((output / 'run.json').read_text())['settings'] != settings:
            raise ValueError('Resume settings differ from run.json; restore the original configuration')
        if json.loads((output / 'benchmark.json').read_text()) != tasks:
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
                    record['target_camera'] = reference_camera_state(goal, task_data['goal_view']) if goal else None
                    write_json(path, record)
                records[task_key(record)] = record
    summary = checkpoint(output, tasks, records)
    print(f'Batch model: {MODEL}\nBenchmark: {BENCHMARK_JSON}\nOutput: {output}', flush=True)
    if summary['complete']:
        print('All tasks already finished.', flush=True)
        return 0

    backend = None
    try:
        backend, protocol = create_model(MODEL, MODEL_GENERATION_CONFIGS[MODEL], MODEL_SERVER_CONFIG)
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
