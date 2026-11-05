"""Load an existing point cloud and render a movable first-person camera in Blender."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import bpy
from mathutils import Matrix, Quaternion, Vector

from .boundary import movement_block_reason
from .trajectory import TrajectoryMap


def emission_material(name, color, strength=1):
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*color, 1)
    emission.inputs['Strength'].default_value = strength
    material.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
    return material


def create_scene(source, args):
    import bpy
    from mathutils import Vector

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.ply_import(filepath=str(source), import_colors='SRGB')
    meshes = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    point_nodes = []
    for obj in meshes:
        is_points = len(obj.data.polygons) == 0
        mat = emission_material('ScanColor', (0.6, 0.6, 0.6))
        nodes = mat.node_tree.nodes
        emission = nodes['Emission']
        colors = obj.data.color_attributes
        if colors:
            color_name = (colors.active_color or colors[0]).name
            if is_points:
                color = nodes.new('ShaderNodeAttribute')
                color.attribute_name = color_name
            else:
                color = nodes.new('ShaderNodeVertexColor')
                color.layer_name = color_name
            mat.node_tree.links.new(color.outputs['Color'], emission.inputs['Color'])
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        if is_points:
            # Cycles renders point-cloud primitives directly: no per-point mesh copies.
            group = bpy.data.node_groups.new('PLY_RenderPoints', 'GeometryNodeTree')
            group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
            group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
            input_node = group.nodes.new('NodeGroupInput')
            output_node = group.nodes.new('NodeGroupOutput')
            points = group.nodes.new('GeometryNodeMeshToPoints')
            points.mode = 'VERTICES'
            material = group.nodes.new('GeometryNodeSetMaterial')
            material.inputs['Material'].default_value = mat
            group.links.new(input_node.outputs['Geometry'], points.inputs['Mesh'])
            group.links.new(points.outputs['Points'], material.inputs['Geometry'])
            group.links.new(material.outputs['Geometry'], output_node.inputs['Geometry'])
            modifier = obj.modifiers.new('Render Point Cloud', 'NODES')
            modifier.node_group = group
            point_nodes.append(points)
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    low = Vector([min(c[i] for c in corners) for i in range(3)])
    high = Vector([max(c[i] for c in corners) for i in range(3)])
    center = (low + high) / 2
    extent = (high - low).length
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = 'PERSP'
    # The renderer configures the camera from the existing cameras.json.
    camera.data.clip_start = 0.001
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 8
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.025, .025, .025, 1)
    scene.view_settings.view_transform = 'Standard'
    return meshes, point_nodes, corners, center, extent, camera, scene


def estimate_scene_up(meshes, cameras, extent, up_override=None):
    """Return up, description, and a floor point; use a low-point estimate if no plane fits."""
    import numpy as np

    extrinsics = np.asarray([c['extrinsic'] for c in cameras], dtype=np.float64)
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    camera_centers = -np.einsum('nij,nj->ni', rotations.transpose(0, 2, 1), translations)
    camera_ups = -rotations[:, 1, :]
    camera_ups /= np.linalg.norm(camera_ups, axis=1, keepdims=True)
    reference_up = np.asarray(up_override, dtype=float) if up_override is not None else np.median(camera_ups, axis=0)
    length = np.linalg.norm(reference_up)
    if not np.isfinite(reference_up).all() or length < 1e-6:
        raise ValueError('Camera orientations do not provide a stable up direction; set scene up explicitly')
    reference_up = reference_up / length
    if up_override is None and float(np.median(camera_ups @ reference_up)) < .5:
        raise ValueError('Camera up directions disagree; set scene up explicitly')

    arrays = []
    for obj in meshes:
        coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get('co', coords)
        coords = coords.reshape((-1, 3)).astype(np.float64)
        transform = np.asarray(obj.matrix_world, dtype=np.float64)
        arrays.append(coords @ transform[:3, :3].T + transform[:3, 3])
    cloud = np.concatenate(arrays)
    if len(cloud) < 3 or not np.isfinite(cloud).all():
        raise ValueError('point cloud has too few valid points')
    rng = np.random.default_rng(42)
    sample = cloud[rng.choice(len(cloud), min(len(cloud), 12000), replace=False)]
    tolerance = max(extent * 0.003, 1e-8)
    levels = sample @ reference_up
    pool = sample[levels <= np.quantile(levels, .2)]
    if len(pool) < 3:
        pool = sample
    best = None
    for _ in range(1500):
        a, b, c = pool[rng.choice(len(pool), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-10 * extent * extent:
            continue
        normal /= length
        if normal @ reference_up < 0:
            normal = -normal
        # Use the whole sequence as the vertical reference, rejecting wall planes.
        if normal @ reference_up < np.cos(np.deg2rad(30)):
            continue
        offset = -normal @ a
        heights = np.einsum('ij,j->i', sample, normal) + offset
        inliers = np.abs(heights) < tolerance
        support = int(inliers.sum())
        if support < max(60, len(sample) * 0.03):
            continue
        camera_heights = camera_centers @ normal + offset
        if np.mean(camera_heights > tolerance) < 0.95:
            continue
        if np.mean(heights >= -2 * tolerance) < 0.90:
            continue
        # Prefer a well-supported floor whose other points lie above it.
        score = support * np.mean(heights >= -2 * tolerance)
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None:
        floor_height = float(np.quantile(sample @ reference_up, 0.05))
        floor_point = reference_up * floor_height
        return (reference_up.tolist(),
                'multi-camera up; approximate floor from lower 5% of points', floor_point.tolist(), sample, cloud)

    floor = sample[best[1]]
    center = floor.mean(axis=0)
    _, _, basis = np.linalg.svd(floor - center, full_matrices=False)
    normal = basis[-1]
    if normal @ reference_up < 0:
        normal = -normal
    if normal @ reference_up < np.cos(np.deg2rad(30)):
        floor_height = float(np.quantile(sample @ reference_up, .05))
        return (reference_up.tolist(), 'multi-camera up; approximate floor after rejected refinement',
                (reference_up * floor_height).tolist(), sample, cloud)
    camera_above = float(np.mean((camera_centers - center) @ normal > tolerance))
    return normal.tolist(), (f'floor plane: {len(floor):,}/{len(sample):,} sampled points; '
                           f'{camera_above:.0%} cameras above'), center.tolist(), sample, cloud


def configure_first_person(camera, frame, panel_width):
    """Keep the video focal lengths while extending its view to a square frame."""
    import bpy

    fx_norm = frame['intrinsic'][0][0] / frame['W']
    fy_norm = frame['intrinsic'][1][1] / frame['H']
    cx_norm = frame['intrinsic'][0][2] / frame['W']
    cy_norm = frame['intrinsic'][1][2] / frame['H']

    width = height = panel_width
    video_height = max(1, round(width * frame['H'] / frame['W']))
    # Keep the original horizontal field of view and central video projection.
    # Extra rows above and below it expand the view without stretching pixels.
    fx = fx_norm * width
    fy = fy_norm * video_height
    cx = cx_norm * width
    cy = cy_norm * video_height + (height - video_height) / 2
    pixel_aspect_y = fx / fy
    bpy.context.scene.render.pixel_aspect_x = max(1.0, 1 / pixel_aspect_y)
    bpy.context.scene.render.pixel_aspect_y = max(1.0, pixel_aspect_y)

    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.sensor_width = 36.0
    camera.data.lens = camera.data.sensor_width * fx_norm
    camera.data.shift_x = .5 - cx_norm
    camera.data.shift_y = (cy - height / 2) * pixel_aspect_y / width

    return {
        'source': 'first_scene_camera',
        'width': width, 'height': height,
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'pixel_aspect_y': pixel_aspect_y,
    }


def initial_camera_pose(extrinsic):
    """Convert the recorded OpenCV camera pose to Blender's camera axes."""
    return Matrix(extrinsic).inverted() @ Matrix.Diagonal((1, -1, -1, 1))


def create_crosshair(camera, scene):
    depth = max(0.08, camera.data.clip_start * 10)
    root = bpy.data.objects.new('First-person crosshair', None)
    scene.collection.objects.link(root)
    root.parent = camera
    root.location = (0, 0, -depth)
    # Local x/y units become fractions of the rendered image width.
    scale = depth * camera.data.sensor_width / camera.data.lens
    root.scale = (scale,) * 3

    gap, end = .006, .018
    arms = [((-end, 0, 0), (-gap, 0, 0)),
            ((gap, 0, 0), (end, 0, 0)),
            ((0, -end, 0), (0, -gap, 0)),
            ((0, gap, 0), (0, end, 0))]

    def strokes(name, color, thickness):
        curve = bpy.data.curves.new(name, 'CURVE')
        curve.dimensions = '3D'
        curve.bevel_depth = thickness
        curve.bevel_resolution = 2
        for start, finish in arms:
            spline = curve.splines.new('POLY')
            spline.points.add(1)
            spline.points[0].co = (*start, 1)
            spline.points[1].co = (*finish, 1)
        curve.materials.append(emission_material(name, color))
        obj = bpy.data.objects.new(name, curve)
        scene.collection.objects.link(obj)
        obj.parent = root
        if name == 'Crosshair white strokes':
            obj.location.z = .01  # Put white closer to the camera than its outline.
        for prop in ('visible_shadow', 'visible_diffuse', 'visible_glossy',
                     'visible_transmission', 'visible_volume_scatter'):
            setattr(obj, prop, False)

    strokes('Crosshair dark outline', (0.005, 0.005, 0.005), .0027)
    strokes('Crosshair white strokes', (1, 1, 1), .0015)
    return root


class SceneRenderer:
    def __init__(self, config, output):
        self.config = config
        directory = Path(config['scene_dir'])
        cameras = json.loads((directory / 'cameras.json').read_text())
        cameras.sort(key=lambda frame: frame.get('video_frame_index', frame.get('id', 0)))
        args = SimpleNamespace(samples=config['samples'], resolution=config['resolution'])
        meshes, point_nodes, _, self.center, self.extent, self.camera, self.scene = create_scene(
            directory / 'reconstruction.ply', args)
        for points in point_nodes:
            points.inputs['Radius'].default_value = config['point_radius']
        first = cameras[0]
        initial_pose = initial_camera_pose(first['extrinsic'])
        self.initial_position = initial_pose.translation.copy()
        self.initial_rotation = initial_pose.to_quaternion().normalized()
        estimated, description, floor, sample, cloud = estimate_scene_up(
            meshes, cameras, self.extent, config['up'])
        self.floor_description = description
        self.up = Vector(config['up'] if config['up'] is not None else estimated).normalized()
        if self.up.length < 1e-8 or not all(math.isfinite(value) for value in self.up):
            raise ValueError('Scene up must be a finite, nonzero vector')
        forward = self.initial_rotation @ Vector((0, 0, -1))
        heading = forward - self.up * forward.dot(self.up)
        if heading.length < 1e-6:
            # A vertical first view has no projected forward direction. Use its
            # right axis to choose a horizontal heading without changing the pose.
            heading = self.up.cross(self.initial_rotation @ Vector((1, 0, 0)))
        self.initial_heading = heading.normalized()
        self.intrinsics = configure_first_person(self.camera, first, config['resolution'])
        self.crosshair = create_crosshair(self.camera, self.scene)
        self.trajectory_map = TrajectoryMap(self, sample, floor, cloud)
        self.start_session(output)
        print(f'Scene loaded: {directory}; scene up={list(self.up)} ({description}); '
              'camera pose from the first recorded frame', flush=True)

    def _apply_pose(self, position, rotation):
        self.camera.location = position
        self.camera.rotation_mode = 'QUATERNION'
        self.camera.rotation_quaternion = rotation.normalized()
        self.camera.data.clip_end = max(
            self.extent * 20, (self.camera.location - self.center).length + self.extent * 2)

    def start_session(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self._apply_pose(self.initial_position, self.initial_rotation)
        self.heading = self.initial_heading.copy()
        self.output = output
        self.index = 0
        self.observations = []

    def observe(self, command='observe', magnitude=None, applied_magnitude=None, blocked=False, reason=None):
        index = self.index + 1
        label = command if magnitude is None else f'{command}_{str(magnitude).removesuffix(".0")}'
        if blocked:
            label += '_blocked'
        image = self.output / f'{index:04d}_{label}.png'
        metadata = image.with_suffix('.json')
        top_down = image.with_name(image.stem + '_top_down.png')
        trajectory_path = self.output / 'trajectory.json'
        temporary_trajectory = trajectory_path.with_suffix('.json.tmp')
        try:
            bpy.context.view_layer.update()
            self.scene.render.filepath = str(image)
            bpy.ops.render.render(write_still=True)
            record = {
                'observation_id': index, 'image': str(image),
                'width': self.config['resolution'], 'height': self.config['resolution'],
            }
            if magnitude is not None:
                record.update(magnitude=magnitude, applied_magnitude=applied_magnitude, blocked=blocked, reason=reason)
            observation = {
                **record, 'command': command,
                'magnitude': magnitude, 'applied_magnitude': applied_magnitude,
                'blocked': blocked, 'reason': reason,
                'position': list(self.camera.location),
                'forward': list(self.camera.rotation_quaternion @ Vector((0, 0, -1))),
                'heading': list(self.heading),
                'up': list(self.camera.rotation_quaternion @ Vector((0, 1, 0))),
                'top_down_image': str(top_down),
            }
            observations = [*self.observations, observation]
            frame = self.trajectory_map.render(self.output, observations)
            metadata.write_text(json.dumps({
                **observation, 'intrinsics': self.intrinsics, 'top_down_frame': frame,
            }, indent=2) + '\n')
            temporary_trajectory.write_text(json.dumps(observations, indent=2) + '\n')
            temporary_trajectory.replace(trajectory_path)
        except BaseException:
            for path in (image, metadata, temporary_trajectory):
                if not path.is_dir():
                    path.unlink(missing_ok=True)
            raise
        self.index = index
        self.observations = observations
        return record

    def action(self, key, magnitude=None):
        if key not in {'move_forward', 'turn_left', 'turn_right', 'turn_up', 'turn_down'}:
            raise ValueError('Allowed actions: move_forward, turn_left, turn_right, turn_up, turn_down')
        if magnitude is None:
            magnitude = self.config['move_step' if key == 'move_forward' else 'look_step']
        if type(magnitude) not in (int, float) or not math.isfinite(magnitude) or magnitude <= 0:
            raise ValueError('Magnitude must be a positive finite number')
        position = self.camera.location.copy()
        rotation = self.camera.rotation_quaternion.copy()
        heading = self.heading.copy()
        next_position, next_rotation = position.copy(), rotation.copy()
        next_heading = heading.copy()
        applied_magnitude = magnitude
        reason = None
        if key == 'move_forward':
            if self.config.get('boundary_enforced', True):
                reason = movement_block_reason(self.trajectory_map.boundary, position, heading, magnitude)
            if reason is None:
                next_position += heading * magnitude
            else:
                applied_magnitude = 0
        elif key in {'turn_left', 'turn_right'}:
            angle = math.radians(magnitude) * (1 if key == 'turn_left' else -1)
            turn = Quaternion(self.up, angle)
            next_heading = (turn @ heading).normalized()
            next_rotation = turn @ rotation
        else:
            forward = rotation @ Vector((0, 0, -1))
            pitch = math.atan2(forward.dot(self.up), forward.dot(heading))
            angle = math.radians(magnitude) * (1 if key == 'turn_up' else -1)
            next_pitch = max(-math.pi / 2, min(math.pi / 2, pitch + angle))
            applied_magnitude = math.degrees(abs(next_pitch - pitch))
            right = heading.cross(self.up).normalized()
            next_rotation = Quaternion(right, next_pitch - pitch) @ rotation
        if reason is None:
            self._apply_pose(next_position, next_rotation)
            self.heading = next_heading
        try:
            return self.observe(key, magnitude, applied_magnitude, blocked=reason is not None, reason=reason)
        except BaseException:
            self._apply_pose(position, rotation)
            self.heading = heading
            raise
