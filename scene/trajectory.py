"""Blender map backgrounds and trajectory charts, optionally exposed by a tool."""

from pathlib import Path


class TrajectoryMap:
    """Render an opaque, roof-clipped background; reuse it while bounds stay fixed."""

    def __init__(self, renderer, points, floor, cloud):
        import bpy
        import numpy as np
        from mathutils import Matrix, Vector

        self.renderer = renderer
        self.up = renderer.up
        self.floor = Vector(floor)
        self.origin = renderer.initial_position.copy()
        right = renderer.initial_rotation @ Vector((1, 0, 0))
        right -= self.up * right.dot(self.up)
        if right.length < 1e-6:
            right = self.up.orthogonal()
        self.right = right.normalized()
        self.ahead = self.up.cross(self.right).normalized()
        heights = (points - floor) @ np.asarray(self.up)
        keep = (heights >= -.1) & (heights <= renderer.config['trajectory_max_height'])
        if not keep.any():
            raise ValueError('No map points remain; increase trajectory_max_height')
        xy = (points[keep] - np.asarray(self.origin)) @ np.column_stack((self.right, self.ahead))
        self.low, self.high = np.quantile(xy, [.005, .995], axis=0)
        self.target = None
        if renderer.config.get('target_camera') is not None:
            pose = np.linalg.inv(np.asarray(renderer.config['target_camera']['extrinsic'], dtype=float))
            self.target = {'position': pose[:3, 3].tolist(), 'forward': pose[:3, 2].tolist()}
            basis = np.column_stack((self.right, self.ahead))
            target_xy = (pose[:3, 3] - np.asarray(self.origin)) @ basis
            direction = pose[:3, 2] @ basis
            length = np.linalg.norm(direction)
            tip = target_xy.copy()
            if length > 1e-6:
                tip += direction / length * renderer.config.get('trajectory_arrow_length', .25) * 1.4
            self.low = np.minimum(self.low, np.minimum(target_xy, tip))
            self.high = np.maximum(self.high, np.maximum(target_xy, tip))
        self.boundary = None
        enforced = bool(renderer.config.get('boundary_enforced', True))
        if renderer.config.get('boundary_preview', True) or enforced:
            from .boundary import build_boundary

            self.boundary = build_boundary(
                cloud, origin=self.origin, right=self.right, ahead=self.ahead, up=self.up, floor=floor,
                max_height=renderer.config['trajectory_max_height'],
                cell_size=renderer.config.get('boundary_cell_size', .05),
                close_radius=renderer.config.get('boundary_close_radius', .15),
                camera_radius=renderer.config.get('camera_radius', .15),
            )
            self.boundary['boundary_enforced'] = enforced
            self.low = np.minimum(self.low, self.boundary['bounds'][0])
            self.high = np.maximum(self.high, self.boundary['bounds'][1])
        data = bpy.data.cameras.new('Trajectory map camera')
        data.type = 'ORTHO'
        self.camera = bpy.data.objects.new('Trajectory map camera', data)
        renderer.scene.collection.objects.link(self.camera)
        self.camera.rotation_euler = Matrix((self.right, self.ahead, self.up)).transposed().to_euler()
        self.cached_frame = None

    def render(self, output, observations):
        import bpy
        import numpy as np
        from mathutils import Vector

        renderer, scene = self.renderer, self.renderer.scene
        xy = np.array([((Vector(record['position']) - self.origin).dot(self.right),
                        (Vector(record['position']) - self.origin).dot(self.ahead))
                       for record in observations])
        low, high = np.minimum(self.low, xy.min(axis=0)), np.maximum(self.high, xy.max(axis=0))
        margin = max(float((high - low).max()) * .05, .2)
        middle = (low + high) / 2
        span = float((high - low).max()) + 2 * margin
        center = self.origin + self.right * float(middle[0]) + self.ahead * float(middle[1])
        center -= self.up * (center - self.floor).dot(self.up)
        background = Path(output) / '.top_down_background.png'
        boundary = Path(output) / 'boundary.npz'
        if self.boundary is not None and not boundary.is_file():
            np.savez_compressed(boundary, **self.boundary)
        frame = {
            'background': str(background), 'center': list(center), 'span': span,
            'origin': list(self.origin), 'right': list(self.right),
            'ahead': list(self.ahead), 'up': list(self.up),
            'image_extent': [float(middle[0] - span / 2), float(middle[0] + span / 2),
                             float(middle[1] - span / 2), float(middle[1] + span / 2)],
            'xlim': [float(low[0] - margin), float(high[0] + margin)],
            'ylim': [float(low[1] - margin), float(high[1] + margin)],
            'scene': Path(renderer.config['scene_dir']).name,
            'dpi': renderer.config.get('trajectory_dpi', 180),
            'arrow_length': renderer.config.get('trajectory_arrow_length', .25),
            'boundary': str(boundary) if self.boundary is not None else None,
            'show_boundary': bool(renderer.config.get('boundary_preview', True)),
            'boundary_enforced': bool(renderer.config.get('boundary_enforced', True)),
            'floor_estimate': renderer.floor_description,
            'target': self.target,
        }
        if frame == self.cached_frame and background.is_file():
            return frame
        distance = renderer.extent * 2 + 10
        self.camera.location = center + self.up * distance
        self.camera.data.ortho_scale = span
        self.camera.data.clip_start = max(.001, distance - renderer.config['trajectory_max_height'])
        self.camera.data.clip_end = distance + .1
        temporary = background.with_name('.top_down_background.tmp.png')
        saved = (scene.camera, scene.render.resolution_x, scene.render.resolution_y,
                 scene.render.pixel_aspect_x, scene.render.pixel_aspect_y,
                 scene.cycles.samples, scene.render.filepath)
        crosshair = [renderer.crosshair, *renderer.crosshair.children_recursive]
        visibility = [obj.hide_render for obj in crosshair]
        try:
            for obj in crosshair:
                obj.hide_render = True
            scene.camera = self.camera
            scene.render.resolution_x = scene.render.resolution_y = renderer.config['trajectory_resolution']
            scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
            scene.cycles.samples = renderer.config['trajectory_samples']
            scene.render.filepath = str(temporary)
            bpy.context.view_layer.update()
            bpy.ops.render.render(write_still=True)
            temporary.replace(background)
        finally:
            if not temporary.is_dir():
                temporary.unlink(missing_ok=True)
            for obj, hidden in zip(crosshair, visibility):
                obj.hide_render = hidden
            (scene.camera, scene.render.resolution_x, scene.render.resolution_y,
             scene.render.pixel_aspect_x, scene.render.pixel_aspect_y,
             scene.cycles.samples, scene.render.filepath) = saved
            bpy.context.view_layer.update()
        self.cached_frame = frame
        return frame


def draw_trajectory(frame, observations, output):
    """Draw every observation in the reference chart style, in ordinary Python."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    output = Path(output)
    basis = np.column_stack((frame['right'], frame['ahead']))
    positions = np.array([record['position'] for record in observations])
    xy = (positions - frame['origin']) @ basis
    directions = np.array([record['heading'] for record in observations]) @ basis
    lengths = np.linalg.norm(directions, axis=1)
    usable = lengths > 1e-6
    steps = np.array([record['observation_id'] for record in observations])
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    temporary = output.with_suffix('.tmp.png')
    latest = output.parent / 'top_down.png'
    temporary_latest = latest.with_suffix('.tmp.png')
    try:
        ax.imshow(plt.imread(frame['background']), extent=frame['image_extent'],
                  origin='upper', interpolation='nearest', zorder=1)
        if frame.get('boundary') and frame.get('show_boundary', True):
            with np.load(frame['boundary']) as boundary:
                x = boundary['grid_origin'][0] + (np.arange(boundary['scene_mask'].shape[1]) + .5) * boundary['cell_size']
                y = boundary['grid_origin'][1] + (np.arange(boundary['scene_mask'].shape[0]) + .5) * boundary['cell_size']
                for name, color, label in (
                    ('scene_mask', '#ffbf00', 'Scene boundary'),
                    ('allowed_mask', '#39e75f', f"Inset ({float(boundary['camera_radius']):g} m)"),
                ):
                    if boundary[name].any():
                        ax.contour(x, y, boundary[name].astype(float), levels=[.5],
                                   colors=[color], linewidths=1.2, zorder=2)
                        ax.plot([], [], color=color, linewidth=2, label=label)
                if not bool(boundary['start_allowed']):
                    message = ('Initial position outside scene footprint' if not bool(boundary['start_inside'])
                               else 'Initial position outside inset region')
                    ax.text(.02, .02, message, transform=ax.transAxes, color='#c02020', zorder=10,
                            bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none'})
        # Resets teleport to the start; never draw their jumps as traveled paths.
        connected = np.array([record.get('command') != 'reset' for record in observations[1:]], dtype=bool)
        segments = np.stack((xy[:-1], xy[1:]), axis=1)[connected]
        ax.add_collection(LineCollection(segments, colors='white', linewidths=5, zorder=3))
        colors = matplotlib.colors.LinearSegmentedColormap.from_list(
            'trajectory_blues', plt.get_cmap('Blues')(np.linspace(.45, .95, 256)))
        line = LineCollection(segments, cmap=colors, norm=plt.Normalize(1, max(int(steps[-1]), 2)),
                              linewidths=2.5, zorder=4, label='History')
        line.set_array(((steps[:-1] + steps[1:]) / 2)[connected])
        ax.add_collection(line)
        ax.scatter(*xy[0], s=115, c='#20c56b', edgecolors='black',
                   linewidths=1.2, zorder=7, label='Start')
        if usable[0]:
            heading = directions[0] / lengths[0] * frame['arrow_length']
            ax.quiver(*xy[0], *heading, angles='xy', scale_units='xy', scale=1,
                      color='#20c56b', edgecolors='white', linewidths=.5, width=.007, zorder=8)
        target = frame.get('target')
        if target is not None:
            target_xy = (np.asarray(target['position']) - frame['origin']) @ basis
            target_direction = np.asarray(target['forward']) @ basis
            target_length = np.linalg.norm(target_direction)
            ax.scatter(*target_xy, s=240, c='#ff8c00', marker='*', edgecolors='white',
                       linewidths=1.2, zorder=10, label='Target view')
            if target_length > 1e-6:
                arrow = target_direction / target_length * frame['arrow_length'] * 1.4
                ax.quiver(*target_xy, *arrow, angles='xy', scale_units='xy', scale=1,
                          color='#ff8c00', edgecolors='white', linewidths=.5, width=.006, zorder=11)
        ax.scatter(*xy[-1], s=145, c='#ef3340', marker='X', edgecolors='white',
                   linewidths=1.2, zorder=20, label='Current')
        if usable[-1]:
            heading = directions[-1] / lengths[-1] * frame['arrow_length']
            ax.quiver(*xy[-1], *heading, angles='xy', scale_units='xy', scale=1,
                      color='#ef3340', edgecolors='white', linewidths=.5, width=.007, zorder=21)
        preview = (' | boundary limits' if frame.get('boundary_enforced') else ' | boundary preview') if frame.get('boundary') else ''
        if 'approximate' in frame.get('floor_estimate', ''):
            preview += ' | approximate floor'
        ax.set(xlim=frame['xlim'], ylim=frame['ylim'], xlabel='Ground X (approx. m)',
               ylabel='Ground Y (approx. m)', title=f"{frame['scene']} | {len(observations)} observations{preview}")
        ax.set_aspect('equal')
        ax.legend(loc='upper right', framealpha=.95)
        colorbar = fig.colorbar(line, ax=ax, label='Observation step', shrink=.8, pad=.025)
        colorbar.locator = MaxNLocator(integer=True)
        colorbar.update_ticks()
        fig.savefig(temporary, dpi=frame['dpi'], bbox_inches='tight', pad_inches=.1, transparent=False)
        temporary_latest.write_bytes(temporary.read_bytes())
        temporary.replace(output)
        temporary_latest.replace(latest)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)
        temporary_latest.unlink(missing_ok=True)
