"""Replay single-step SFT trajectories and save their first-person observations."""

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import stat
import sys
from tempfile import mkdtemp

import numpy as np
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(PROJECT_DIR))

from main_nav_batch import SCENE_CONFIG as NAV_SCENE_CONFIG
from scene.client import CameraClient

# OPENBLAS_NUM_THREADS=1 python generate_benchmark/render_train_trajectories_sim_single_step.py

TRAJECTORY_JSON = PROJECT_DIR / 'outputs' / 'benchmark_train_traj_sim_single_step.json'
DATA_ROOT = Path('/data4/haibo/data/3D_Env')
OUTPUT_DIR = PROJECT_DIR / 'outputs' / 'train_traj_sim_single_step_renders'
NUM_WORKERS = 16
MAX_TRAJECTORIES = None  # Set to an integer to render a prefix for a pilot run.
RETRY_ERRORS = True
SYNC_ONLY = False  # Backfill paths from saved results without launching Blender.
SCENE_CONFIG = {**NAV_SCENE_CONFIG, 'render_top_down': False}
POSITION_TOLERANCE_M = 0.001
ANGLE_TOLERANCE_DEGREES = 0.1


@dataclass(frozen=True)
class RenderOptions:
    trajectory_json: Path
    data_root: Path
    output_dir: Path
    scene_config: dict
    num_workers: int
    max_trajectories: int | None
    retry_errors: bool
    sync_only: bool
    position_tolerance_m: float
    angle_tolerance_degrees: float
    action_tool: str = 'camera_action'
    render_image_field: str = 'image_path'


def file_version(metadata):
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns


def write_json(path, data, *, expected_file=None):
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        if expected_file is not None:
            if file_version(path.stat()) != file_version(expected_file):
                raise RuntimeError(f'Input changed during rendering; leaving {path} untouched')
            temporary.chmod(stat.S_IMODE(expected_file.st_mode))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def plan_fingerprint(data):
    """Identify the input plan independently of the render paths added below."""
    digest = hashlib.sha256()

    def add(value):
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        digest.update(encoded.encode('utf-8') + b'\n')

    add({key: value for key, value in data.items() if key != 'trajectories'})
    for row in data['trajectories']:
        steps = [{key: value for key, value in step.items() if key not in ('image_path', 'render_img_path')}
                 for step in row['route_actions']]
        add({**row, 'route_actions': steps})
    return digest.hexdigest()


def route_key(row):
    return row['scene'], row['pair_id']


def route_dir(root, row):
    scene, pair_id = route_key(row)
    directory = Path(scene)
    if directory.is_absolute() or len(directory.parts) != 2 or '..' in directory.parts:
        raise ValueError(f'Expected a dataset/scene path: {scene}')
    if not pair_id or Path(pair_id).name != pair_id or pair_id in ('.', '..'):
        raise ValueError(f'Invalid pair ID: {pair_id!r}')
    return root / 'tasks' / directory / pair_id


def pose_errors(actual, planned):
    angles = []
    for key in ('forward', 'up', 'heading'):
        a, b = np.asarray(actual[key]), np.asarray(planned[key])
        angles.append(math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))
    return math.dist(actual['position'], planned['position']), max(angles)


def render_route(row, root, data_root, scene_config, position_tolerance, angle_tolerance,
                 action_tool='camera_action'):
    folder = route_dir(root, row)
    folder.mkdir(parents=True, exist_ok=True)
    attempt = Path(mkdtemp(prefix='attempt_', dir=folder))
    result = {'scene': row['scene'], 'pair_id': row['pair_id'], 'distance_group': row['distance_group'],
              'initial_view': row['initial_view'], 'target_view': row['target_view'],
              'attempt_dir': str(attempt.relative_to(root)), 'status': 'error'}
    images = []
    try:
        steps = row['route_actions']
        intermediate_tools = {action_tool}
        if action_tool == 'camera_actions':
            intermediate_tools.add('camera_observe')  # An identical-pose video frame has no movement.
        if (len(steps) < 2 or steps[0]['name'] != 'camera_observe' or steps[-1]['name'] != 'finish_task'
                or any(s['name'] not in intermediate_tools for s in steps[1:-1])):
            raise ValueError(f'Expected camera_observe, {action_tool} steps, and finish_task')
        scene_dir = data_root / row['scene']
        cameras = json.loads((scene_dir / 'cameras.json').read_text(encoding='utf-8'))
        matches = [c for c in cameras if c['image_path'] == row['initial_view']]
        if len(matches) != 1:
            raise ValueError(f'Expected one camera record for {row["initial_view"]}')
        config = {**scene_config, 'scene_dir': str(scene_dir), 'initial_camera': matches[0]}
        with CameraClient(config['blender'], config, attempt, timeout=config['timeout']) as camera:
            for index, step in enumerate(steps, start=1):
                name = step['name']
                if name == 'finish_task':
                    # Finishing retains the last viewpoint and therefore the last image.
                    image = root / images[-1]
                else:
                    if name == 'camera_observe':
                        observation = camera.observe()
                    elif name == 'camera_actions':
                        observation = camera.actions(**step['arguments'])
                        if len(observation['actions']) != len(step['arguments']['actions']):
                            raise RuntimeError('Blender returned an incomplete action batch')
                        blocked = next(((i, a) for i, a in enumerate(observation['actions'], 1) if a['blocked']), None)
                        if blocked is not None:
                            sub_action, action_result = blocked
                            result.update(status='blocked', step=index, sub_action=sub_action,
                                          reason=action_result.get('reason'))
                            break
                    else:
                        observation = camera.action(**step['arguments'])
                    image = Path(observation['image'])
                    if observation.get('blocked'):
                        result.update(status='blocked', step=index, reason=observation.get('reason'))
                        break
                if not image.is_file() or image.stat().st_size == 0:
                    raise FileNotFoundError(f'Missing rendered image: {image}')
                actual = json.loads(image.with_suffix('.json').read_text(encoding='utf-8'))
                position_error, angle_error = pose_errors(actual, step['pose'])
                if position_error > position_tolerance or angle_error > angle_tolerance:
                    result.update(status='pose_mismatch', step=index, position_error_m=position_error,
                                  angle_error_degrees=angle_error)
                    break
                images.append(str(image.relative_to(root)))
            else:
                result.update(status='rendered', image_paths=images)
    except Exception as error:
        result.update(status='error', reason=f'{type(error).__name__}: {error}')
        result.pop('image_paths', None)

    write_json(folder / 'result.json', result)
    return result


def resume_result(root, row, retry_errors):
    path = route_dir(root, row) / 'result.json'
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    if result['status'] == 'error' and retry_errors:
        return None
    if result['status'] == 'rendered':
        images = result['image_paths']
        if len(images) != len(row['route_actions']) or not all((root / image).is_file() for image in images):
            return None
    return result


def render_pending(jobs, results, root, options):
    with tqdm(total=len(jobs), initial=len(results), desc='Blender trajectories', unit='route') as bar:
        queue = (row for row in jobs if route_key(row) not in results)
        if len(results) == len(jobs):
            return
        with ProcessPoolExecutor(max_workers=options.num_workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = set()

            def submit():
                row = next(queue, None)
                if row is not None:
                    pending.add(pool.submit(render_route, row, root, options.data_root, options.scene_config,
                                            options.position_tolerance_m, options.angle_tolerance_degrees,
                                            options.action_tool))

            counts = Counter(r['status'] for r in results.values())
            for _ in range(options.num_workers):
                submit()
            try:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        results[route_key(result)] = result
                        counts[result['status']] += 1
                        bar.update()
                        bar.set_postfix(ok=counts['rendered'], blocked=counts['blocked'],
                                        mismatch=counts['pose_mismatch'], error=counts['error'])
                        submit()
            finally:
                for future in pending:
                    future.cancel()


def write_index(root, jobs, results):
    temporary = root / 'rendered.jsonl.tmp'
    with temporary.open('w', encoding='utf-8') as f:
        for row in jobs:
            result = results.get(route_key(row))
            if result and result['status'] == 'rendered':
                f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(root / 'rendered.jsonl')
    counts = Counter(r['status'] for r in results.values())
    write_json(root / 'summary.json', {'total': len(jobs), 'pending': len(jobs) - len(results),
                                      'by_status': dict(counts)})


def backfill_image_paths(path, data, jobs, root, results, expected_file, image_field):
    changed = False
    fields = {image_field, 'image_path'}
    for row in jobs:
        result = results.get(route_key(row))
        steps = row['route_actions']
        if result and result['status'] == 'rendered':
            for step, image in zip(steps, result['image_paths'], strict=True):
                reference = os.path.relpath(root / image, path.parent)
                if step.get(image_field) != reference:
                    step[image_field] = reference
                    changed = True
                if image_field != 'image_path' and 'image_path' in step:
                    del step['image_path']
                    changed = True
        else:
            for step in steps:
                for field in fields:
                    image = step.get(field)
                    if image and (path.parent / image).resolve().is_relative_to(root):
                        del step[field]
                        changed = True
    if changed:
        write_json(path, data, expected_file=expected_file)
        print(f'Updated step image paths in {path}', flush=True)


def run(options):
    if options.num_workers < 1 or (options.max_trajectories is not None and options.max_trajectories < 1):
        raise ValueError('NUM_WORKERS and MAX_TRAJECTORIES must be positive')
    root = options.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = options.trajectory_json.resolve()
    manifest_lock = manifest.with_name(f'.{manifest.name}.render.lock')
    with (root / '.render.lock').open('a') as lock, manifest_lock.open('a') as input_lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(input_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another renderer is using {root} or {manifest}') from error

        with manifest.open('rb') as f:
            input_file = os.fstat(f.fileno())
            raw = f.read()
        legacy_hash = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        del raw
        jobs = data['trajectories'][:options.max_trajectories]
        if len(jobs) != len({route_key(r) for r in jobs}):
            raise ValueError('Input contains duplicate scene/pair IDs')
        settings = {'input': str(manifest), 'input_plan_sha256': plan_fingerprint(data),
                    'data_root': str(options.data_root.resolve()), 'scene_config': options.scene_config,
                    'position_tolerance_m': options.position_tolerance_m,
                    'angle_tolerance_degrees': options.angle_tolerance_degrees}
        if options.action_tool != 'camera_action':
            settings['action_tool'] = options.action_tool
        run_path = root / 'run.json'
        if run_path.is_file():
            saved = json.loads(run_path.read_text(encoding='utf-8'))
            previous = saved.copy()
            if previous.get('input_sha256') == legacy_hash and 'input_plan_sha256' not in previous:
                del previous['input_sha256']
                previous['input_plan_sha256'] = settings['input_plan_sha256']
            if previous != settings:
                raise ValueError('Input or rendering settings changed; choose a new OUTPUT_DIR')
            if saved != previous:
                write_json(run_path, settings)
        else:
            write_json(run_path, settings)

        results = {}
        for row in jobs:
            result = resume_result(root, row, options.retry_errors)
            if result is not None:
                results[route_key(row)] = result
        try:
            if not options.sync_only:
                render_pending(jobs, results, root, options)
        except KeyboardInterrupt:
            for row in jobs:
                if route_key(row) not in results:
                    result = resume_result(root, row, options.retry_errors)
                    if result is not None:
                        results[route_key(row)] = result
            print(f'Interrupted. Run the same script to resume in {root}.', flush=True)
            return 130
        finally:
            write_index(root, jobs, results)
            backfill_image_paths(manifest, data, jobs, root, results, input_file, options.render_image_field)
    print(f'Completed. Index: {root / "rendered.jsonl"}')
    return 0


def main():
    return run(RenderOptions(TRAJECTORY_JSON, DATA_ROOT, OUTPUT_DIR, SCENE_CONFIG, NUM_WORKERS,
                             MAX_TRAJECTORIES, RETRY_ERRORS, SYNC_ONLY, POSITION_TOLERANCE_M,
                             ANGLE_TOLERANCE_DEGREES))


if __name__ == '__main__':
    raise SystemExit(main())
