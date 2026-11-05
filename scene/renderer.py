"""Load an existing point cloud and render a movable first-person camera in Blender."""

import json
import math
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import bpy
from mathutils import Matrix, Quaternion, Vector


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


def estimate_scene_up(meshes, cameras, extent):
    """Return up, description, and a floor point; use a low-point estimate if no plane fits."""
    import numpy as np

    extrinsics = np.asarray([c['extrinsic'] for c in cameras], dtype=np.float64)
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    camera_centers = -np.einsum('nij,nj->ni', rotations.transpose(0, 2, 1), translations)
    first_camera_up = -rotations[0, 1, :]
    first_camera_up /= np.linalg.norm(first_camera_up)

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
    best = None
    for _ in range(500):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-10 * extent * extent:
            continue
        normal /= length
        if normal @ first_camera_up < 0:
            normal = -normal
        if normal @ first_camera_up < 0.75:
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
        floor_height = float(np.quantile(sample @ first_camera_up, 0.05))
        floor_point = first_camera_up * floor_height
        return (first_camera_up.tolist(),
                'first camera; approximate floor from lower 5% of points', floor_point.tolist())

    floor = sample[best[1]]
    center = floor.mean(axis=0)
    _, _, basis = np.linalg.svd(floor - center, full_matrices=False)
    normal = basis[-1]
    if normal @ first_camera_up < 0:
        normal = -normal
    camera_above = float(np.mean((camera_centers - center) @ normal > tolerance))
    return normal.tolist(), (f'floor plane: {len(floor):,}/{len(sample):,} sampled points; '
                           f'{camera_above:.0%} cameras above'), center.tolist()


def configure_first_person(camera, cameras, panel_width):
    """Keep the video focal lengths while extending its view to a square frame."""
    import bpy

    fx_norm = median(frame['intrinsic'][0][0] / frame['W'] for frame in cameras)
    fy_norm = median(frame['intrinsic'][1][1] / frame['H'] for frame in cameras)
    cx_norm = median(frame['intrinsic'][0][2] / frame['W'] for frame in cameras)
    cy_norm = median(frame['intrinsic'][1][2] / frame['H'] for frame in cameras)

    width = height = panel_width
    video_height = max(1, round(width * cameras[0]['H'] / cameras[0]['W']))
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
        'source': 'median_scene_cameras',
        'width': width, 'height': height,
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'pixel_aspect_y': pixel_aspect_y,
    }


def initial_camera_pose(extrinsic, up, floor_normal, floor_point, eye_height):
    """Project the first camera onto the floor along up, then rise by eye_height."""
    from mathutils import Matrix, Vector

    pose = Matrix(extrinsic).inverted()
    position = pose.translation
    direction = pose.to_3x3() @ Vector((0, 0, 1))
    alignment = up.dot(floor_normal)
    if alignment <= 0:
        raise ValueError('Scene up must point away from the estimated floor')
    position += up * (eye_height - (position - floor_point).dot(floor_normal) / alignment)
    return position, direction


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
        args = SimpleNamespace(samples=config['samples'], resolution=config['resolution'])
        meshes, point_nodes, _, self.center, self.extent, self.camera, self.scene = create_scene(
            directory / 'reconstruction.ply', args)
        for points in point_nodes:
            points.inputs['Radius'].default_value = config['point_radius']
        estimated, description, floor = estimate_scene_up(meshes, cameras, self.extent)
        self.up = Vector(config['up'] if config['up'] is not None else estimated).normalized()
        position, direction = initial_camera_pose(
            cameras[0]['extrinsic'], self.up, Vector(estimated), Vector(floor), config['eye_height'])
        direction -= self.up * direction.dot(self.up)
        if direction.length < 1e-8:
            raise ValueError('Initial camera direction is vertical; cannot determine a horizontal heading')
        self.initial_position = position.copy()
        self.initial_forward = direction.normalized()
        self.intrinsics = configure_first_person(self.camera, cameras, config['resolution'])
        create_crosshair(self.camera, self.scene)
        self.start_session(output)
        print(f'Scene loaded: {directory}; up={list(self.up)} ({description})', flush=True)

    def _apply_pose(self, position, forward):
        backward = -forward.normalized()
        right = self.up.cross(backward).normalized()
        screen_up = backward.cross(right).normalized()
        self.camera.location = position
        self.camera.rotation_euler = Matrix((right, screen_up, backward)).transposed().to_euler()
        self.camera.data.clip_end = max(
            self.extent * 20, (self.camera.location - self.center).length + self.extent * 2)

    def start_session(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self._apply_pose(self.initial_position, self.initial_forward)
        self.output = output
        self.index = 0

    def observe(self, command='observe'):
        index = self.index + 1
        image = self.output / f'{index:04d}_{command}.png'
        metadata = image.with_suffix('.json')
        try:
            bpy.context.view_layer.update()
            self.scene.render.filepath = str(image)
            bpy.ops.render.render(write_still=True)
            record = {
                'observation_id': index, 'image': str(image),
                'width': self.config['resolution'], 'height': self.config['resolution'],
            }
            metadata.write_text(json.dumps({
                **record, 'command': command,
                'position': list(self.camera.location),
                'forward': list(self.camera.rotation_euler.to_matrix() @ Vector((0, 0, -1))),
                'intrinsics': self.intrinsics,
            }, indent=2) + '\n')
        except BaseException:
            image.unlink(missing_ok=True)
            metadata.unlink(missing_ok=True)
            raise
        self.index = index
        return record

    def action(self, key, magnitude=None):
        if key not in {'move_forward', 'turn_left', 'turn_right'}:
            raise ValueError('Allowed actions: move_forward, turn_left, turn_right')
        if magnitude is None:
            magnitude = self.config['move_step' if key == 'move_forward' else 'look_step']
        if type(magnitude) not in (int, float) or not math.isfinite(magnitude) or magnitude <= 0:
            raise ValueError('Magnitude must be a positive finite number')
        position = self.camera.location.copy()
        forward = (self.camera.rotation_euler.to_matrix() @ Vector((0, 0, -1))).normalized()
        next_position, next_forward = position.copy(), forward.copy()
        if key == 'move_forward':
            next_position += forward * magnitude
        else:
            angle = math.radians(magnitude) * (1 if key == 'turn_left' else -1)
            next_forward = Quaternion(self.up, angle) @ forward
        self._apply_pose(next_position, next_forward)
        try:
            return self.observe(key)
        except BaseException:
            self._apply_pose(position, forward)
            raise
