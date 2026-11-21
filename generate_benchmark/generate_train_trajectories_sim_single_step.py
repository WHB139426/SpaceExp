"""Sample reference waypoints and generate first-person actions and camera poses."""

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
OUTPUT_JSON = BENCHMARK_JSON.with_name('benchmark_train_traj.json')
SEED = 42
WAYPOINT_COUNT_RANGE = (5, 10)  # Inclusive; counts both endpoints.
START_LEVEL_PITCH_THRESHOLD_DEGREES = 30.0
SPLIT_YAW_MIN_DEGREES = 120.0
SPLIT_YAW_MIN_FOLLOWING_MOVE_M = 0.2
SPLIT_FORWARD_MIN_M = 1.6
SPLIT_FORWARD_MAX_PART_M = 1.0


def sample_indices(length, requested):
    count = min(length, requested)
    if count == length:
        return list(range(length))
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def camera_action(key, magnitude):
    return {'name': 'camera_action', 'arguments': {'key': key, 'magnitude': float(magnitude)}}


def horizontal(vector, up):
    return vector - np.dot(vector, up) * up


def yaw_action(heading, desired, up):
    # The renderer uses positive rotation around scene up for turn_left.
    angle = math.degrees(math.atan2(np.dot(up, np.cross(heading, desired)), np.dot(heading, desired)))
    if abs(angle) <= 1e-7:
        return None
    return camera_action('turn_left' if angle > 0 else 'turn_right', abs(angle))


def plan_actions(waypoints, scene_up):
    up = np.asarray(scene_up, dtype=float)
    length = np.linalg.norm(up)
    if up.shape != (3,) or not math.isfinite(length) or length < 1e-8:
        raise ValueError('Scene up must be a finite, nonzero 3D vector')
    up /= length
    initial, target = waypoints[0], waypoints[-1]
    start_forward = np.asarray(initial['forward'])
    right = np.cross(start_forward, initial['up'])
    right /= np.linalg.norm(right)
    camera_axes = np.stack((start_forward, np.cross(right, start_forward)))
    heading = horizontal(start_forward, up)
    if np.linalg.norm(heading) < 1e-6:
        heading = np.cross(up, right)
    heading /= np.linalg.norm(heading)
    start_pitch = math.atan2(np.dot(start_forward, up), np.dot(start_forward, heading))

    position = np.asarray(initial['position'])
    route_actions = []

    def record(call, waypoint_index=None):
        if waypoint_index is not None:
            call['waypoint_index'] = waypoint_index
        call['pose'] = {'position': position.tolist(), 'forward': camera_axes[0].tolist(),
                        'up': camera_axes[1].tolist(), 'heading': heading.tolist()}
        route_actions.append(call)

    def rotate_view(axis, angle):
        nonlocal camera_axes
        axis = axis / np.linalg.norm(axis)
        cosine, sine = math.cos(angle), math.sin(angle)
        camera_axes = (camera_axes * cosine + np.cross(axis, camera_axes) * sine
                       + np.outer(camera_axes @ axis, axis) * (1 - cosine))

    def turn_heading(desired, waypoint_index=None, following_distance=0.0):
        nonlocal heading
        turn = yaw_action(heading, desired, up)
        if turn is not None:
            args = turn['arguments']
            sign = 1 if args['key'] == 'turn_left' else -1
            split = (args['magnitude'] >= SPLIT_YAW_MIN_DEGREES
                     and following_distance >= SPLIT_YAW_MIN_FOLLOWING_MOVE_M)
            if split:
                args['magnitude'] /= 2
            angle = math.radians(sign * args['magnitude'])
            if split:
                rotate_view(up, angle)
                heading = heading * math.cos(angle) + np.cross(up, heading) * math.sin(angle)
                record(camera_action(args['key'], args['magnitude']), waypoint_index)
            rotate_view(up, angle)
            heading = desired
            record(turn, waypoint_index)

    def turn_pitch(change, waypoint_index=None):
        degrees = math.degrees(change)
        if abs(degrees) > 1e-7:
            rotate_view(np.cross(heading, up), change)
            key = 'turn_up' if degrees > 0 else 'turn_down'
            record(camera_action(key, abs(degrees)), waypoint_index)

    record({'name': 'camera_observe', 'arguments': {}})
    travel_pitch = start_pitch
    if abs(math.degrees(start_pitch)) > START_LEVEL_PITCH_THRESHOLD_DEGREES:
        turn_pitch(-start_pitch, waypoint_index=0)
        travel_pitch = 0.0

    # Forward moves preserve camera height. Visit each waypoint's horizontal
    # position, keeping the recorded initial height for the entire route.
    for index, point in enumerate(waypoints[1:], start=1):
        delta = horizontal(np.asarray(point['position']) - position, up)
        distance = np.linalg.norm(delta)
        if distance <= 1e-8:
            continue
        turn_heading(delta / distance, index, distance)
        parts = math.ceil(distance / SPLIT_FORWARD_MAX_PART_M) if distance >= SPLIT_FORWARD_MIN_M else 1
        start = position
        traveled = 0.0
        for part in range(1, parts + 1):
            next_distance = distance if part == parts else distance * part / parts
            position = start + heading * next_distance
            record(camera_action('move_forward', next_distance - traveled), index)
            traveled = next_distance

    target_forward = np.asarray(target['forward'])
    target_heading = horizontal(target_forward, up)
    target_length = np.linalg.norm(target_heading)
    if target_length >= 1e-6:
        turn_heading(target_heading / target_length)
    target_pitch = math.atan2(np.dot(target_forward, up), target_length)
    turn_pitch(target_pitch - travel_pitch)
    record({'name': 'finish_task', 'arguments': {
        'status': 'success', 'reason': 'Reached the target location and matched its viewing direction.'
    }})
    return route_actions


def generate(benchmark_json, data_root, output_json):
    if output_json.resolve() == benchmark_json.resolve():
        raise ValueError('The output path must differ from the input benchmark path')
    tasks = json.loads(benchmark_json.read_text(encoding='utf-8'))
    scenes = {}
    for task in tasks:
        scene = task['scene']
        pair_id = task.get('pair_id', task['id'])
        goal_type = task.get('goal_type', 'first_person')
        if goal_type not in ('first_person', 'top_down'):
            raise ValueError(f'Unsupported goal type {goal_type!r} for {scene}/{pair_id}')
        siblings = scenes.setdefault(scene, {}).setdefault(pair_id, {})
        if goal_type in siblings:
            raise ValueError(f'Duplicate {goal_type} task for {scene}/{pair_id}')
        siblings[goal_type] = task

    rng = random.Random(SEED)
    trajectories = []
    for scene, pairs in scenes.items():
        scene_dir = data_root / scene
        cameras = json.loads((scene_dir / 'cameras.json').read_text(encoding='utf-8'))
        cameras_by_path = {camera['image_path']: camera for camera in cameras}
        if len(cameras_by_path) != len(cameras):
            raise ValueError(f'Duplicate camera image paths in {scene}/cameras.json')
        poses = {}
        for pair_id, siblings in pairs.items():
            task = siblings.get('first_person')
            if task is None:
                continue
            route = task['reference_route']
            endpoint_fields = ('initial_view', 'target_view')
            if len(route) < 2 or (route[0], route[-1]) != tuple(task[k] for k in endpoint_fields):
                raise ValueError(f'Reference route must include both endpoints for {scene}/{pair_id}')
            shared_fields = (*endpoint_fields, 'reference_route', 'distance_group')
            if any(sibling[k] != task[k] for sibling in siblings.values() for k in shared_fields):
                raise ValueError(f'Sibling tasks disagree on the route for {scene}/{pair_id}')

            requested = rng.randint(*WAYPOINT_COUNT_RANGE)
            waypoints = []
            for index in sample_indices(len(route), requested):
                frame = route[index]
                if frame not in poses:
                    if frame not in cameras_by_path:
                        raise ValueError(f'No camera record for {scene}/{frame} in pair {pair_id}')
                    camera = reference_camera_state(cameras_by_path[frame], scene_dir / frame)
                    poses[frame] = {k: camera[k] for k in ('video_frame_index', 'position', 'forward', 'up')}
                waypoints.append({'reference_index': index, 'image_path': frame, **poses[frame]})

            scene_up = siblings.get('top_down', {}).get('scene_up', task.get('scene_up'))
            if scene_up is None:
                # Legacy first-person benchmarks can still be sampled without scene geometry.
                route_actions = None
            else:
                try:
                    route_actions = plan_actions(waypoints, scene_up)
                except ValueError as exc:
                    raise ValueError(f'{scene}/{pair_id}: {exc}') from exc
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
                'waypoints': waypoints,
                'route_actions': route_actions,
            })

    data = {
        'source_benchmark': benchmark_json.name,
        'sampling': {
            'method': 'uniform_reference_index',
            'seed': SEED,
            'waypoint_count_range': list(WAYPOINT_COUNT_RANGE),
        },
        'action_plan': {
            'method': 'turn_then_forward',
            'goal_type': 'first_person',
            'height': 'initial_camera_height',
            'initial_observation': 'camera_observe',
            'pose_space': 'world',
            'start_leveling': {'threshold_degrees': START_LEVEL_PITCH_THRESHOLD_DEGREES,
                               'target_pitch_degrees': 0.0},
            'splitting': {
                'yaw': {'minimum_degrees': SPLIT_YAW_MIN_DEGREES,
                        'minimum_following_move_m': SPLIT_YAW_MIN_FOLLOWING_MOVE_M, 'parts': 2},
                'forward': {'minimum_m': SPLIT_FORWARD_MIN_M, 'maximum_part_m': SPLIT_FORWARD_MAX_PART_M},
            },
        },
        'trajectories': trajectories,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(mode='w', encoding='utf-8', dir=output_json.parent,
                                  prefix=f'.{output_json.name}.', suffix='.tmp', delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            json.dump(data, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.write('\n')
        temporary_path.replace(output_json)
    finally:
        temporary_path.unlink(missing_ok=True)
    return len(trajectories)


if __name__ == '__main__':
    count = generate(BENCHMARK_JSON, DATA_ROOT, OUTPUT_JSON)
    print(f'Saved {count:,} sampled routes to {OUTPUT_JSON}')
