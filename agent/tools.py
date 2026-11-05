"""First-person camera tools and multimodal observations."""

import base64
from pathlib import Path

from .types import Tool, ToolOutput


def _observation(record) -> ToolOutput:
    encoded = base64.b64encode(Path(record['image']).read_bytes()).decode('ascii')
    return ToolOutput(
        value=record,
        messages=({
            'role': 'user',
            'content': [
                {'type': 'text', 'text': '<first_person_view>'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encoded}'}},
                {'type': 'text', 'text': '</first_person_view>'},
            ],
        },),
    )


def create_tools(camera, *, move_step, look_step) -> list[Tool]:
    def camera_observe():
        return _observation(camera.observe())

    def camera_action(key, magnitude=None):
        return _observation(camera.action(key, magnitude))

    return [
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
                'move_forward moves along the stored horizontal forward direction, preserving height and camera orientation. '
                'The initial forward direction is the first frame\'s viewing direction projected onto the ground plane. '
                'turn_left and turn_right rotate both this forward direction and the camera around scene up, preserving position and pitch. '
                'turn_up and turn_down adjust camera pitch only, preserving position and horizontal forward direction. '
                'Pitch is limited to -90 through 90 degrees; looking vertically up or down retains the horizontal forward direction. '
                'Looking down does not make move_forward descend, and looking up does not make it ascend. '
                'The initial camera position and complete orientation are preserved from the first recorded frame. '
                'One scene unit is one meter. Movement magnitudes are in meters; turn magnitudes are in degrees. '
                f'Default forward distance: {move_step:g} meters. '
                f'Default turn angle: {look_step:g} degrees. '
                'Magnitudes must be positive and finite. '
                'When scene-boundary limits are enabled, the entire forward path must stay inside the scene boundary with camera clearance. '
                'A blocked move keeps the camera in place and returns blocked=true, reason, and applied_magnitude=0, together with the current view. '
                'Turning remains allowed. Furniture and interior obstacles are not collision-tested.'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'key': {'type': 'string', 'enum': [
                        'move_forward', 'turn_left', 'turn_right', 'turn_up', 'turn_down',
                    ]},
                    'magnitude': {
                        'type': 'number', 'exclusiveMinimum': 0,
                        'description': (
                            'Distance in meters for move_forward (one scene unit equals one meter), '
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
