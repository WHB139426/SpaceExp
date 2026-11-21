"""Generate batched first-person navigation labels paired with original video frames."""

import json
import math
from pathlib import Path
import random
import sys
from tempfile import NamedTemporaryFile

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(PROJECT_DIR))

from scene.client import reference_camera_state


BENCHMARK_JSON = PROJECT_DIR / 'outputs' / 'benchmark_train.json'
DATA_ROOT = Path('/data4/haibo/data/3D_Env')
OUTPUT_JSON = BENCHMARK_JSON.with_name('benchmark_train_traj_frame.json')
SEED = 42
WAYPOINT_COUNT_RANGE = (5, 10)  # Inclusive; counts both endpoints.


def sample_indices(length, requested, rng):
    if length <= requested:
        return list(range(length))
    return [0, *sorted(rng.sample(range(1, length - 1), requested - 2)), length - 1]


def horizontal(vector, up):
    return vector - np.dot(vector, up) * up


def plan_actions(frames, scene_up):
    up = np.asarray(scene_up, dtype=float)
    norm = np.linalg.norm(up)
    if up.shape != (3,) or not math.isfinite(norm) or norm < 1e-8:
        raise ValueError('Scene up must be a finite, nonzero 3D vector')
    up = up / norm

    initial = frames[0]
    position = np.asarray(initial['position'])
    forward = np.asarray(initial['forward'])
    right = np.cross(forward, initial['up'])
    right /= np.linalg.norm(right)
    camera_axes = np.stack((forward, np.cross(right, forward)))
    heading = horizontal(forward, up)
    if np.linalg.norm(heading) < 1e-6:
        heading = np.cross(up, right)
    heading /= np.linalg.norm(heading)
    route_actions = []

    def record(name, arguments, frame):
        height_error = abs(np.dot(position - np.asarray(frame['position']), up))
        route_actions.append({
            'name': name,
            'arguments': arguments,
            'frame_path': frame['image_path'],
            'reference_index': frame['reference_index'],
            'video_frame_index': frame['video_frame_index'],
            'pose': {'position': position.tolist(), 'forward': camera_axes[0].tolist(),
                     'up': camera_axes[1].tolist(), 'heading': heading.tolist()},
            'height_error_m': float(height_error),
        })

    def rotate_view(axis, angle):
        nonlocal camera_axes
        axis = axis / np.linalg.norm(axis)
        cosine, sine = math.cos(angle), math.sin(angle)
        camera_axes = (camera_axes * cosine + np.cross(axis, camera_axes) * sine
                       + np.outer(camera_axes @ axis, axis) * (1 - cosine))

    def turn_heading(desired, actions):
        nonlocal heading
        angle = math.atan2(np.dot(up, np.cross(heading, desired)), np.dot(heading, desired))
        degrees = math.degrees(angle)
        if abs(degrees) > 1e-7:
            rotate_view(up, angle)
            heading = desired
            actions.append({'key': 'turn_left' if degrees > 0 else 'turn_right', 'magnitude': abs(degrees)})

    record('camera_observe', {}, initial)
    for frame in frames[1:]:
        actions = []
        target_position = np.asarray(frame['position'])
        target_forward = np.asarray(frame['forward'])
        delta = horizontal(target_position - position, up)
        distance = np.linalg.norm(delta)
        if distance > 1e-8:
            turn_heading(delta / distance, actions)
            position = position + heading * distance
            actions.append({'key': 'move_forward', 'magnitude': float(distance)})

        target_heading = horizontal(target_forward, up)
        heading_length = np.linalg.norm(target_heading)
        if heading_length > 1e-8:
            turn_heading(target_heading / heading_length, actions)
        pitch = math.atan2(np.dot(camera_axes[0], up), np.dot(camera_axes[0], heading))
        target_pitch = math.atan2(np.dot(target_forward, up), heading_length)
        change = math.degrees(target_pitch - pitch)
        if abs(change) > 1e-7:
            rotate_view(np.cross(heading, up), math.radians(change))
            actions.append({'key': 'turn_up' if change > 0 else 'turn_down', 'magnitude': abs(change)})

        if (np.linalg.norm(horizontal(position - target_position, up)) > 1e-7
                or np.linalg.norm(camera_axes[0] - target_forward) > 1e-7):
            raise ValueError(f'Planned pose does not match the ground position and forward of {frame["image_path"]}')
        if actions:
            record('camera_actions', {'actions': actions}, frame)
        else:
            record('camera_observe', {}, frame)

    record('finish_task', {'status': 'success',
                           'reason': 'Reached the target location and matched its viewing direction.'}, frames[-1])
    return route_actions


def generate(benchmark_json, data_root, output_json):
    if output_json.resolve() == benchmark_json.resolve():
        raise ValueError('The output path must differ from the input benchmark path')
    tasks = json.loads(benchmark_json.read_text(encoding='utf-8'))
    scene_ups = {}
    scenes = {}
    for task in tasks:
        key = task['scene'], task.get('pair_id', task['id'])
        if task.get('scene_up') is not None:
            scene_ups[key] = task['scene_up']
        if task.get('goal_type', 'first_person') == 'first_person':
            pairs = scenes.setdefault(key[0], {})
            if key[1] in pairs:
                raise ValueError(f'Duplicate first-person task for {key[0]}/{key[1]}')
            pairs[key[1]] = task

    rng = random.Random(SEED)
    trajectories = []
    for scene, pairs in scenes.items():
        scene_dir = data_root / scene
        camera_records = json.loads((scene_dir / 'cameras.json').read_text(encoding='utf-8'))
        cameras = {camera['image_path']: camera for camera in camera_records}
        if len(cameras) != len(camera_records):
            raise ValueError(f'Duplicate camera image paths in {scene}/cameras.json')
        poses = {}
        for pair_id, task in pairs.items():
            route = task['reference_route']
            if len(route) < 2 or (route[0], route[-1]) != (task['initial_view'], task['target_view']):
                raise ValueError(f'Reference route must include both endpoints for {scene}/{pair_id}')
            scene_up = scene_ups.get((scene, pair_id))
            if scene_up is None:
                raise ValueError(f'No scene_up recorded for {scene}/{pair_id}')

            requested = rng.randint(*WAYPOINT_COUNT_RANGE)
            frames = []
            for index in sample_indices(len(route), requested, rng):
                frame = route[index]
                if frame not in poses:
                    if frame not in cameras:
                        raise ValueError(f'No camera record for {scene}/{frame} in pair {pair_id}')
                    if not (scene_dir / frame).is_file():
                        raise FileNotFoundError(scene_dir / frame)
                    camera = reference_camera_state(cameras[frame], scene_dir / frame)
                    poses[frame] = {k: camera[k] for k in ('image_path', 'video_frame_index', 'position', 'forward', 'up')}
                frames.append({'reference_index': index, **poses[frame]})

            route_actions = plan_actions(frames, scene_up)
            trajectories.append({
                'scene': scene,
                'pair_id': pair_id,
                'distance_group': task['distance_group'],
                'initial_view': task['initial_view'],
                'target_view': task['target_view'],
                'scene_up': scene_up,
                'task_ids': {'first_person': task['id']},
                'reference_frame_count': len(route),
                'requested_waypoint_count': requested,
                'route_actions': route_actions,
            })

    data = {
        'source_benchmark': benchmark_json.name,
        'sampling': {'method': 'random_reference_index', 'seed': SEED,
                     'waypoint_count_range': list(WAYPOINT_COUNT_RANGE), 'include_endpoints': True},
        'action_plan': {'goal_type': 'first_person', 'observation_source': 'original_video_frame',
                        'frame_path_base': 'scene_directory', 'pose_space': 'world',
                        'height': 'initial_camera_height', 'roll': 'initial_camera_roll'},
        'trajectories': trajectories,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(mode='w', encoding='utf-8', dir=output_json.parent,
                                  prefix=f'.{output_json.name}.', suffix='.tmp', delete=False)
    path = Path(temporary.name)
    try:
        with temporary:
            json.dump(data, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.write('\n')
        path.replace(output_json)
    finally:
        path.unlink(missing_ok=True)
    return len(trajectories)


if __name__ == '__main__':
    count = generate(BENCHMARK_JSON, DATA_ROOT, OUTPUT_JSON)
    print(f'Saved {count:,} frame trajectories to {OUTPUT_JSON}')
