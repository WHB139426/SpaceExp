"""Camera tools, optional top-down observations and task completion."""

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
                 enable_move_left=True, enable_move_right=True, enable_top_down=False,
                 enable_finish_task=False, enable_move_up=False, enable_move_down=False,
                 enable_observe=True, enable_action=True, enable_move_forward=True,
                 enable_turn_left=True, enable_turn_right=True,
                 enable_turn_up=True, enable_turn_down=True, enable_reset=False) -> list[Tool]:
    actions = []
    movement_description = ''
    for key, enabled, description in (
        ('move_forward', enable_move_forward, 'move_forward moves along the stored horizontal forward direction. '),
        ('move_backward', enable_move_backward, 'move_backward moves in the opposite direction. '),
        ('move_left', enable_move_left, 'move_left strafes horizontally to the left without turning. '),
        ('move_right', enable_move_right, 'move_right strafes horizontally to the right without turning. '),
    ):
        if enabled:
            actions.append(key)
            movement_description += description
    vertical_description = ''
    for key, enabled, description in (
        ('move_up', enable_move_up, 'move_up raises the camera along scene up. '),
        ('move_down', enable_move_down, 'move_down lowers the camera opposite scene up. '),
    ):
        if enabled:
            actions.append(key)
            vertical_description += description
    if vertical_description:
        vertical_description += (
            'Vertical translations preserve camera orientation and horizontal heading, '
            'regardless of where the camera is looking; they do not follow the rolled image vertical axis. '
        )
    turn_description = ''
    for key, enabled, description in (
        ('turn_left', enable_turn_left,
         'turn_left rotates both the horizontal forward direction and the camera to the left around scene up, preserving position and pitch. '),
        ('turn_right', enable_turn_right,
         'turn_right rotates both the horizontal forward direction and the camera to the right around scene up, preserving position and pitch. '),
        ('turn_up', enable_turn_up,
         'turn_up adjusts camera pitch upward, preserving position and horizontal forward direction. '),
        ('turn_down', enable_turn_down,
         'turn_down adjusts camera pitch downward, preserving position and horizontal forward direction. '),
    ):
        if enabled:
            actions.append(key)
            turn_description += description
    if enable_turn_up or enable_turn_down:
        turn_description += (
            'Pitch is limited to -90 through 90 degrees; looking vertically up or down retains the horizontal forward direction. '
        )

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
                'Horizontal translations are relative to the current first-person horizontal heading, never fixed world axes. '
                'Horizontal translations preserve height, camera orientation, and horizontal heading. '
                f'{vertical_description}'
                'The initial forward direction is the configured initial frame\'s viewing direction projected onto the ground plane. '
                f'{turn_description}'
                'Looking up or down does not change the direction of any translation. '
                'The initial camera position and complete orientation are preserved from the configured initial frame. '
                'One scene unit is one meter. Movement magnitudes are in meters; turn magnitudes are in degrees. '
                f'Default distance for each translation: {move_step:g} meters. '
                f'Default turn angle: {look_step:g} degrees. '
                'Magnitudes must be positive and finite. '
                'When scene-boundary limits are enabled, the entire path of any translation must stay inside the scene boundary with camera clearance. '
                'The camera center must also stay within the configured minimum and maximum heights in meters above the estimated floor. '
                'A blocked move keeps the camera in place and returns blocked=true, reason, and applied_magnitude=0, together with the current view. '
                'Enabled turns remain available when a translation is blocked. '
                'Height limits are a configured range, not detected local ceilings; '
                'furniture and interior obstacles are not collision-tested.'
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
    enabled_tools = {'camera_observe': enable_observe, 'camera_action': enable_action and bool(actions)}
    tools = [tool for tool in tools if enabled_tools[tool.name]]
    if enable_reset:
        def camera_reset():
            return _observation(camera.reset())

        tools.append(Tool(
            name='camera_reset',
            description=(
                'Teleport the camera back to this task\'s configured initial view and return its first-person image. '
                'Restore the exact initial position, complete camera orientation (including pitch and roll), '
                'and horizontal forward direction. For navigation, this is the selected initial view, '
                'not necessarily the first frame in the scene. '
                'Useful, for example, for restarting exploration after losing orientation or taking an unhelpful route. '
                'This is a teleport, not a movement path; movement-boundary checks do not block a reset. '
                'Keep the goal, conversation history, past trajectory and step budget; '
                'append a reset observation to the existing session without ending the task.'
            ),
            parameters={'type': 'object', 'properties': {}, 'additionalProperties': False},
            function=camera_reset,
        ))
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
                'If no observation exists yet, first obtain one using an available observation or action tool.'
            ),
            parameters={'type': 'object', 'properties': {}, 'additionalProperties': False},
            function=camera_top_down,
        ))
    if enable_finish_task:
        def finish_task(status, reason):
            if status not in ('success', 'failure'):
                raise ValueError('status must be success or failure')
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError('reason must be a non-empty English explanation')
            return {'status': status, 'reason': reason.strip()}

        tools.append(Tool(
            name='finish_task',
            description=(
                'Report the task outcome and immediately end the agent loop. '
                'Call this tool when you believe the task is complete or cannot be completed. '
                'Use success only when visual evidence supports meeting the task requirements, '
                'including both goal location and viewing direction for navigation. '
                'Use failure when you cannot complete the task, explaining the obstacle or uncertainty. '
                'This is your self-reported judgment, not an independent success evaluation. '
                'Provide a brief English reason. Call this tool by itself; no further actions will run afterward.'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'status': {'type': 'string', 'enum': ['success', 'failure']},
                    'reason': {'type': 'string', 'minLength': 1,
                               'description': 'Brief English evidence for success or explanation of failure.'},
                },
                'required': ['status', 'reason'], 'additionalProperties': False,
            },
            function=finish_task,
            terminal=True,
        ))
    return tools
