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
                'move_forward moves along the current horizontal viewing direction, preserving height and orientation. '
                'turn_left and turn_right rotate the camera toward its own left or right, preserving position and pitch. '
                f'Default forward distance: {move_step:g} scene units. '
                f'Default turn angle: {look_step:g} degrees. '
                'Distances use the PLY coordinate scale; for metric-calibrated scenes they are approximately meters. '
                'Magnitudes must be positive and finite. There is no collision detection.'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'key': {'type': 'string', 'enum': ['move_forward', 'turn_left', 'turn_right']},
                    'magnitude': {
                        'type': 'number', 'exclusiveMinimum': 0,
                        'description': 'Distance in scene units for move_forward, or angle in degrees for a turn. Omit to use the default.',
                    },
                },
                'required': ['key'], 'additionalProperties': False,
            },
            function=camera_action,
        ),
    ]
