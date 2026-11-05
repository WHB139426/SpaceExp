"""Camera tools and optional top-down observations."""

from pathlib import Path

from .types import Tool, ToolOutput


def _observation(record, view='first_person_view') -> ToolOutput:
    return ToolOutput(
        value=record,
        messages=({
            'role': 'user',
            'content': [
                {'type': 'text', 'text': f'<{view}>'},
                {'type': 'image_url', 'image_url': {'url': str(Path(record['image']).resolve())}},
                {'type': 'text', 'text': f'</{view}>'},
            ],
        },),
    )


def create_tools(camera, *, move_step, look_step, enable_move_backward=True,
                 enable_move_left=True, enable_move_right=True, enable_top_down=False) -> list[Tool]:
    actions = ['move_forward']
    movement_description = 'move_forward moves along the stored horizontal forward direction. '
    for key, enabled, description in (
        ('move_backward', enable_move_backward, 'move_backward moves in the opposite direction. '),
        ('move_left', enable_move_left, 'move_left strafes horizontally to the left without turning. '),
        ('move_right', enable_move_right, 'move_right strafes horizontally to the right without turning. '),
    ):
        if enabled:
            actions.append(key)
            movement_description += description
    actions.extend(['turn_left', 'turn_right', 'turn_up', 'turn_down'])

    def camera_observe():
        return _observation(camera.observe())

    def camera_action(key, magnitude=None):
        if key not in actions:
            raise ValueError(f'Action disabled or unknown: {key}. Allowed actions: {", ".join(actions)}')
        return _observation(camera.action(key, magnitude))

    tools = [
        Tool(
            name='camera_observe',
            description=(
                'Return one image showing the current first-person camera view, with a crosshair, without moving. '
                'Useful, for example, for inspecting the initial scene or examining the current viewpoint.'
            ),
            parameters={'type': 'object', 'properties': {}, 'additionalProperties': False},
            function=camera_observe,
        ),
        Tool(
            name='camera_action',
            description=(
                'Execute one action relative to the current first-person camera and return its updated view as one image. '
                'Useful, for example, for approaching an object or looking around to gather visual evidence. '
                'Actions are relative to the current first-person viewpoint and use the scene up direction for ground movement. '
                f'{movement_description}'
                'All enabled translations are relative to the current first-person horizontal heading, never fixed world axes. '
                'They preserve height, camera orientation, and horizontal heading. '
                'The initial forward direction is the configured initial frame\'s viewing direction projected onto the ground plane. '
                'turn_left and turn_right rotate both this forward direction and the camera around scene up, preserving position and pitch. '
                'turn_up and turn_down adjust camera pitch only, preserving position and horizontal forward direction. '
                'Pitch is limited to -90 through 90 degrees; looking vertically up or down retains the horizontal forward direction. '
                'Looking up or down does not make any translation ascend or descend. '
                'The initial camera position and complete orientation are preserved from the configured initial frame. '
                'One scene unit is one meter. Movement magnitudes are in meters; turn magnitudes are in degrees. '
                f'Default distance for each translation: {move_step:g} meters. '
                f'Default turn angle: {look_step:g} degrees. '
                'Magnitudes must be positive and finite. '
                'When scene-boundary limits are enabled, the entire path of any translation must stay inside the scene boundary with camera clearance. '
                'A blocked move keeps the camera in place and returns blocked=true, reason, and applied_magnitude=0, together with the current view. '
                'Turning remains allowed. Furniture and interior obstacles are not collision-tested.'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'key': {'type': 'string', 'enum': actions},
                    'magnitude': {
                        'type': 'number', 'exclusiveMinimum': 0,
                        'description': (
                            'Distance in meters for any move action (one scene unit equals one meter), '
                            'or angle in degrees for a turn. '
                            f'Omit to move {move_step:g} meters or turn {look_step:g} degrees.'
                        ),
                    },
                },
                'required': ['key'], 'additionalProperties': False,
            },
            function=camera_action,
        ),
    ]
    if enable_top_down:
        def camera_top_down():
            return _observation(camera.top_down(), 'top_down_view')

        tools.append(Tool(
            name='camera_top_down',
            description=(
                'Return the latest top-down map without moving the camera or adding a trajectory observation. '
                'The map shows the start position, past trajectory, current position and horizontal heading, '
                'and the target position and viewing direction when a target is configured. '
                'Useful, for example, for understanding the layout, choosing a route, recovering orientation, '
                'or checking navigation progress. Read the legend to identify each marker. '
                'The target direction is the viewing direction projected onto the ground; a vertical target view has no direction arrow. '
                'If no observation exists yet, call camera_observe first.'
            ),
            parameters={'type': 'object', 'properties': {}, 'additionalProperties': False},
            function=camera_top_down,
        ))
    return tools
