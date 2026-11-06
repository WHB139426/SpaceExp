"""Generate image-goal navigation tasks from held-out ScanNet++ scenes."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from main_agent import SCENE_CONFIG
from scene.boundary import build_boundary, movement_block_reason


DATA_ROOT = Path('/data4/haibo/data/3D_Env')
TEST_SPLIT = DATA_ROOT / 'test.json'
OUTPUT_JSON = Path(__file__).resolve().parent / 'benchmark_test.json'
PREVIEW_DIR = Path(__file__).resolve().parent / 'benchmark_test_previews'
SEED = 42
NUM_SCENES = 30
TASKS_PER_SCENE = 10
DISTANCE_QUOTAS = {'near': 3, 'medium': 4, 'far': 3}
POSITION_BIN = 0.5  # Meters in the reconstructed scene.
HEADING_BIN_DEGREES = 45
PITCH_BIN_DEGREES = 30
HEIGHT_BIN = 0.15
MIN_DISTANCE = 0.75
MAX_HEIGHT_DIFFERENCE = 0.15
MAX_ABS_ROLL_DEGREES = 20
MAX_ROLL_DIFFERENCE_DEGREES = 10
MAX_ABS_PITCH_DEGREES = 60
MIN_CONTRAST = 8
MIN_SHARPNESS = 10  # Laplacian variance after thumbnailing.
MAX_POSE_STEP = 1.5
MAX_POSE_SPEED = 3.0
MAX_REFERENCE_DETOUR = 6.0
MAX_INITIAL_ATTEMPTS = 20
MAX_TASKS_PER_INITIAL = 2
MAX_TARGET_REUSE = 2
SUCCESS_DISTANCE = 0.3
SUCCESS_VIEW_ANGLE_DEGREES = 15
MAP_CONFIG = {
    **SCENE_CONFIG,
    'resolution': 128,
    'samples': 1,
    'trajectory_resolution': 800,
    'trajectory_samples': 2,
    'trajectory_dpi': 80,
    'boundary_enforced': True,
    'boundary_preview': True,
}
PREVIEW_DPI = 90
MAX_PREVIEW_POINTS = 750_000  # Rendering only; geometry checks use the full cloud.


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def eligible_scenes():
    scenes = sorted({name for name in json.loads(TEST_SPLIT.read_text())
                     if len(Path(name).parts) == 2 and Path(name).parts[0] == 'scannetpp'})
    return [name for name in scenes if all((DATA_ROOT / name / file).is_file()
                                          for file in ('cameras.json', 'reconstruction.ply'))]


def image_quality(path):
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert('L')
        image.thumbnail((320, 240))
        gray = np.asarray(image, dtype=float)
    if min(gray.shape) < 3 or gray.std() < MIN_CONTRAST:
        return 0.0
    laplacian = (gray[2:, 1:-1] + gray[:-2, 1:-1] + gray[1:-1, 2:] + gray[1:-1, :-2]
                 - 4 * gray[1:-1, 1:-1])
    return float(laplacian.var())


def candidates(directory, cameras, poses, up, frame):
    positions, forward = poses[:, :3, 3], poses[:, :3, 2]
    heading = forward - (forward @ up)[:, None] * up
    heading /= np.maximum(np.linalg.norm(heading, axis=1, keepdims=True), 1e-12)
    right = np.cross(heading, up)
    ideal_up = np.cross(right, forward)
    camera_up = -poses[:, :3, 1]
    roll = np.degrees(np.arctan2(np.einsum('ij,ij->i', camera_up, right),
                                 np.einsum('ij,ij->i', camera_up, ideal_up)))
    pitch = np.degrees(np.arcsin(np.clip(forward @ up, -1, 1)))
    basis = np.column_stack((frame['right'], frame['ahead']))
    xy = (positions - frame['origin']) @ basis
    heights = positions @ up
    yaw = np.degrees(np.arctan2(heading @ basis[:, 0], heading @ basis[:, 1])) % 360
    groups = {}
    for i, camera in enumerate(cameras):
        if abs(roll[i]) > MAX_ABS_ROLL_DEGREES or abs(pitch[i]) > MAX_ABS_PITCH_DEGREES:
            continue
        path = (directory / camera['image_path']).resolve()
        if not path.is_relative_to((directory / 'frames').resolve()) or not path.is_file():
            continue
        try:
            quality = image_quality(path)
        except (OSError, ValueError):
            continue
        if quality < MIN_SHARPNESS:
            continue
        key = (*np.floor(xy[i] / POSITION_BIN).astype(int), int(np.floor(heights[i] / HEIGHT_BIN)),
               int(yaw[i] // HEADING_BIN_DEGREES), int((pitch[i] + 90) // PITCH_BIN_DEGREES))
        if key not in groups or quality > groups[key][0]:
            groups[key] = (quality, i)
    return sorted(i for _, i in groups.values()), positions, heading, roll, heights, xy


def episode_boundary(points, floor, up, pose):
    # Match TrajectoryMap's axes for this episode's chosen initial camera.
    right = pose[:3, 0] - up * np.dot(pose[:3, 0], up)
    right /= np.linalg.norm(right)
    return build_boundary(
        points, origin=pose[:3, 3], right=right, ahead=np.cross(up, right), up=up, floor=floor,
        max_height=MAP_CONFIG['trajectory_max_height'],
        cell_size=MAP_CONFIG['boundary_cell_size'], close_radius=MAP_CONFIG['boundary_close_radius'],
        camera_radius=MAP_CONFIG['camera_radius'],
    )


def route_prefix(boundary, positions, up, initial, valid_steps):
    # The agent walks at its initial height. Project the observed path to that plane.
    projected = positions - ((positions - positions[initial]) @ up)[:, None] * up
    valid = valid_steps.copy()
    for k in np.flatnonzero(valid):
        delta = projected[k + 1] - projected[k]
        length = float(np.linalg.norm(delta))
        direction = delta / length if length > 1e-9 else np.asarray(boundary['ahead'])
        valid[k] = movement_block_reason(boundary, projected[k], direction, length) is None
    return np.r_[0, np.cumsum(~valid)]


def choose_pairs(directory, cameras, poses, frame, points, rng):
    up = np.asarray(frame['up'], dtype=float)
    up /= np.linalg.norm(up)
    indices, positions, heading, roll, heights, xy = candidates(directory, cameras, poses, up, frame)
    with np.load(frame['boundary']) as saved:
        boundary = dict(saved)
    indices = [i for i in indices if movement_block_reason(boundary, positions[i], heading[i], 0) is None]
    if len(indices) < 5:
        raise ValueError(f'Only {len(indices)} usable position/orientation representatives')
    ids = np.asarray(indices)
    frame_indices = np.asarray([camera['video_frame_index'] for camera in cameras])
    distances = np.linalg.norm(xy[ids, None] - xy[ids][None, :], axis=2)
    compatible = ((distances >= MIN_DISTANCE)
                  & (frame_indices[ids, None] < frame_indices[ids][None, :])
                  & (np.abs(heights[ids, None] - heights[ids][None, :]) <= MAX_HEIGHT_DIFFERENCE)
                  & (np.abs(roll[ids, None] - roll[ids][None, :]) <= MAX_ROLL_DIFFERENCE_DEGREES))
    if not compatible.any():
        raise ValueError('No forward-in-time pairs satisfy distance, height, and roll constraints')
    cuts = np.quantile(distances[compatible], [1 / 3, 2 / 3])
    distance_groups = np.searchsorted(cuts, distances, side='left')
    labels = ['near', 'medium', 'far']
    times = np.asarray([c['time_seconds'] for c in cameras])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dt = np.diff(times)
    valid_steps = ((dt > 0) & (steps <= MAX_POSE_STEP)
                   & (steps / np.maximum(dt, 1e-9) <= MAX_POSE_SPEED))
    path_lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    group_counts, bearing_counts, angle_counts, target_uses = Counter(), Counter(), Counter(), Counter()
    remaining, attempted, selected = np.flatnonzero(compatible.any(axis=1)).tolist(), [], []
    while remaining and len(selected) < TASKS_PER_SCENE and len(attempted) < MAX_INITIAL_ATTEMPTS:
        if not attempted:
            a = int(rng.choice(remaining))
        else:
            spread = np.min(np.linalg.norm(xy[ids[remaining], None] - xy[ids[attempted]][None, :], axis=2), axis=1)
            a = remaining[int(np.argmax(spread + rng.uniform(0, 1e-8, len(remaining))))]
        remaining.remove(a)
        attempted.append(a)
        i = int(ids[a])
        possible = np.flatnonzero(compatible[a]).tolist()
        if not possible:
            continue
        own_boundary = episode_boundary(points, np.asarray(frame['center']), up, poses[i])
        if not own_boundary['start_allowed']:
            continue
        invalid_prefix = route_prefix(own_boundary, positions, up, i, valid_steps)
        choices = []
        for b in possible:
            j = int(ids[b])
            length = float(path_lengths[j] - path_lengths[i])
            if invalid_prefix[j] != invalid_prefix[i] or length > distances[a, b] * MAX_REFERENCE_DETOUR:
                continue
            direction = positions[j] - positions[i]
            bearing = float(np.degrees(np.arctan2(np.dot(direction, np.cross(heading[i], up)),
                                                  np.dot(direction, heading[i]))))
            side = ('front' if abs(bearing) <= 45 else 'back' if abs(bearing) >= 135
                    else 'right' if bearing > 0 else 'left')
            angle = float(np.degrees(np.arccos(np.clip(np.dot(poses[i, :3, 2], poses[j, :3, 2]), -1, 1))))
            turn = 'small_turn' if angle < 45 else 'medium_turn' if angle < 120 else 'large_turn'
            choices.append((b, j, length, side, angle, turn))
        for _ in range(MAX_TASKS_PER_INITIAL):
            available = [choice for choice in choices if target_uses[choice[1]] < MAX_TARGET_REUSE]
            if not available or len(selected) >= TASKS_PER_SCENE:
                break
            def priority(choice):
                b, j, _, side, _, turn = choice
                group = labels[distance_groups[a, b]]
                return (max(DISTANCE_QUOTAS[group] - group_counts[group], 0),
                        -bearing_counts[side], -angle_counts[turn], -target_uses[j], rng.random())
            choice = max(available, key=priority)
            choices.remove(choice)
            b, j, length, side, angle, turn = choice
            group = labels[distance_groups[a, b]]
            selected.append({
                'initial_index': i, 'target_index': j,
                'initial_frame_index': int(frame_indices[i]), 'target_frame_index': int(frame_indices[j]),
                'distance_group': group, 'target_bearing': side, 'orientation_group': turn,
                'ground_distance_m': round(float(distances[a, b]), 4),
                'height_difference_m': round(float(abs(heights[i] - heights[j])), 4),
                'view_angle_degrees': round(angle, 3), 'reference_path_length_m': round(length, 4),
                'route_indices': list(range(i, j + 1)),
            })
            group_counts[group] += 1
            bearing_counts[side] += 1
            angle_counts[turn] += 1
            target_uses[j] += 1
    if len(selected) != TASKS_PER_SCENE:
        raise ValueError(f'Only {len(selected)}/{TASKS_PER_SCENE} valid tasks after {len(attempted)} initial poses')
    return selected, {'representatives': len(ids), 'distance_cuts_m': cuts.tolist(),
                      'distance_groups': dict(group_counts), 'bearings': dict(bearing_counts),
                      'orientations': dict(angle_counts), 'scene_up': up.tolist(),
                      'floor_estimate': frame['floor_estimate']}


def save_preview(directory, cameras, poses, frame, pair, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1, 1.2]})
    try:
        for ax, key, title in zip(axes[:2], ('initial_index', 'target_index'), ('Initial view', 'Target view')):
            camera = cameras[pair[key]]
            with Image.open(directory / camera['image_path']) as image:
                ax.imshow(image)
            ax.set_title(f"{title}\n{Path(camera['image_path']).name}")
            ax.axis('off')
        ax = axes[2]
        basis = np.column_stack((frame['right'], frame['ahead']))
        xy = (poses[pair['route_indices'], :3, 3] - frame['origin']) @ basis
        ax.imshow(plt.imread(frame['background']), extent=frame['image_extent'], origin='upper')
        ax.plot(xy[:, 0], xy[:, 1], color='white', linewidth=4)
        ax.plot(xy[:, 0], xy[:, 1], color='#2285c3', linewidth=2, label='Reference route')
        for index, point, color, marker, label in (
            (pair['initial_index'], xy[0], '#20c56b', 'o', 'Initial'),
            (pair['target_index'], xy[-1], '#ff8c00', '*', 'Target'),
        ):
            ax.scatter(*point, c=color, marker=marker, s=140, edgecolors='white', zorder=5, label=label)
            direction = poses[index, :3, 2] @ basis
            length = np.linalg.norm(direction)
            if length > 1e-6:
                arrow = direction / length * .3
                ax.quiver(*point, *arrow, angles='xy', scale_units='xy', scale=1,
                          color=color, edgecolors='white', linewidths=.5, width=.01, zorder=6)
        ax.set(xlim=frame['xlim'], ylim=frame['ylim'], aspect='equal',
               title=f"{pair['distance_group']} | {pair['ground_distance_m']:.1f} m apart\n"
                     f"{pair['reference_path_length_m']:.1f} m reference route")
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        fig.savefig(output, dpi=PREVIEW_DPI, facecolor='white')
    finally:
        plt.close(fig)


def prepare_map(job_path):
    """Run inside Blender; keep complete geometry for floor and boundary estimation."""
    import bpy
    from scene.renderer import SceneRenderer

    job = json.loads(job_path.read_text())
    renderer = SceneRenderer(job['config'], job['output'])
    total = sum(len(obj.data.vertices) for obj in bpy.data.objects if obj.type == 'MESH')
    stride = max(1, int(np.ceil(total / MAX_PREVIEW_POINTS)))
    for group in bpy.data.node_groups:
        for points in list(group.nodes):
            if points.bl_idname != 'GeometryNodeMeshToPoints':
                continue
            points.inputs['Radius'].default_value *= min(np.sqrt(stride), 2)
            if stride > 1:
                index = group.nodes.new('GeometryNodeInputIndex')
                modulo = group.nodes.new('ShaderNodeMath')
                modulo.operation = 'MODULO'
                modulo.inputs[1].default_value = stride
                select = group.nodes.new('ShaderNodeMath')
                select.operation = 'LESS_THAN'
                select.inputs[1].default_value = .5
                group.links.new(index.outputs[0], modulo.inputs[0])
                group.links.new(modulo.outputs[0], select.inputs[0])
                group.links.new(select.outputs[0], points.inputs['Selection'])
    frame = renderer.trajectory_map.render(Path(job['output']), [{
        'position': list(renderer.initial_position), 'heading': list(renderer.initial_heading),
        'observation_id': 1,
    }])
    write_json(Path(job['output']) / 'frame.json', frame)


def process_scene(name):
    import open3d as o3d

    directory = DATA_ROOT / name
    cameras = sorted(json.loads((directory / 'cameras.json').read_text()), key=lambda c: c['video_frame_index'])
    if len(cameras) < 2 or np.any(np.diff([c['video_frame_index'] for c in cameras]) <= 0):
        raise ValueError('Camera frame indices must be unique and strictly increasing')
    poses = np.linalg.inv(np.asarray([c['extrinsic'] for c in cameras], dtype=float))
    if not np.isfinite(poses).all():
        raise ValueError('Non-finite camera poses')
    output = PREVIEW_DIR / Path(name).name
    output.mkdir(parents=True, exist_ok=True)
    config = {**MAP_CONFIG, 'scene_dir': str(directory.resolve())}
    config.pop('initial_camera', None)
    config.pop('target_camera', None)
    # Obtain the environment's ground frame, with bounded preview rendering cost.
    map_output = output / '.map'
    map_output.mkdir(exist_ok=True)
    job = map_output / 'job.json'
    write_json(job, {'config': config, 'output': str(map_output)})
    with (map_output / 'render.log').open('w') as log:
        subprocess.run([
            config['blender'], '--background', '--factory-startup', '--python-exit-code', '1',
            '--python', str(Path(__file__).resolve()), '--', '--prepare-map', str(job),
        ], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=config['timeout'])
    frame = json.loads((map_output / 'frame.json').read_text())
    cloud = o3d.io.read_point_cloud(str(directory / 'reconstruction.ply'))
    points = np.asarray(cloud.points)
    scene_seed = SEED + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    pairs, summary = choose_pairs(directory, cameras, poses, frame, points, np.random.default_rng(scene_seed))
    tasks = []
    for n, pair in enumerate(pairs, 1):
        task_id = f'{Path(name).name}_{n:04d}'
        preview = output / f'{task_id}.jpg'
        save_preview(directory, cameras, poses, frame, pair, preview)
        task = {'id': task_id, 'scene': name,
                'initial_view': cameras[pair['initial_index']]['image_path'],
                'target_view': cameras[pair['target_index']]['image_path']}
        task.update({key: value for key, value in pair.items()
                     if key not in ('initial_index', 'target_index', 'route_indices')})
        task['reference_route'] = [cameras[i]['image_path'] for i in pair['route_indices']]
        task['preview'] = str(preview.relative_to(OUTPUT_JSON.parent))
        tasks.append(task)
    return tasks, summary


def main():
    from tqdm import tqdm

    if sum(DISTANCE_QUOTAS.values()) != TASKS_PER_SCENE:
        raise ValueError('DISTANCE_QUOTAS must sum to TASKS_PER_SCENE')
    pool = eligible_scenes()
    if len(pool) < NUM_SCENES:
        raise ValueError(f'Only {len(pool)} reconstructed test ScanNet++ scenes are available')
    order = np.random.default_rng(SEED).permutation(pool).tolist()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'seed': SEED, 'test_split': str(TEST_SPLIT), 'candidate_order': order,
        'settings': {key: value for key, value in globals().items()
                     if key.isupper() and isinstance(value, (int, float, dict))},
        'route_validation': 'Observed video path projected to initial height; episode-specific outer boundary only. '
                            'Interior obstacles are not collision-tested. Reference routes are not shortest paths.',
        'success_distance_m': SUCCESS_DISTANCE, 'success_view_angle_degrees': SUCCESS_VIEW_ANGLE_DEGREES,
        'frame_order': 'initial_frame_index < target_frame_index; reference routes advance in video order',
        'scenes': {}, 'rejected_scenes': {}, 'complete': False,
    }
    tasks = []
    with tqdm(total=NUM_SCENES, desc='Benchmark scenes', unit='scene') as progress:
        for name in order:
            tqdm.write(f'Generating: {name}')
            try:
                selected, summary = process_scene(name)
            except ValueError as error:
                report['rejected_scenes'][name] = f'{type(error).__name__}: {error}'
                tqdm.write(f'SKIP {name}: {error}')
            else:
                tasks.extend(selected)
                report['scenes'][name] = summary
                progress.update(1)
                tqdm.write(f'Accepted {name}: {dict(Counter(t["distance_group"] for t in selected))}')
            report['complete'] = len(report['scenes']) == NUM_SCENES
            write_json(OUTPUT_JSON, tasks)
            write_json(OUTPUT_JSON.with_suffix('.report.json'), report)
            if report['complete']:
                break
    index = ['# Navigation benchmark previews', '', 'Blue: projected reference route. Green: initial. Orange: target.', '']
    for task in tasks:
        relative = (OUTPUT_JSON.parent / task['preview']).relative_to(PREVIEW_DIR)
        index.append(f"- [{task['id']}]({relative.as_posix()}) — {task['distance_group']}, {task['ground_distance_m']:.2f} m")
    (PREVIEW_DIR / 'index.md').write_text('\n'.join(index) + '\n')
    for name in set(pool) - set(report['scenes']):
        scene_id = Path(name).name
        for stale in (PREVIEW_DIR / scene_id).glob(f'{scene_id}_????.jpg'):
            stale.unlink()
    print(f'Scenes: {len(report["scenes"])}/{NUM_SCENES}; tasks: {len(tasks)}; JSON: {OUTPUT_JSON}', flush=True)
    if not report['complete']:
        raise SystemExit('Not enough valid scenes. See the report before adjusting filters.')


if __name__ == '__main__':
    if '--prepare-map' in sys.argv:
        prepare_map(Path(sys.argv[sys.argv.index('--prepare-map') + 1]))
    else:
        main()
