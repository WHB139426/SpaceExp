"""First-person SFT samples from single-step and multi-step trajectories."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Literal

from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools import create_tools


ACTION_TOOLS = {'multi_step': 'camera_actions', 'single_step': 'camera_action'}
OBSERVATION_FIELDS = {
    'multi_step': {'render': 'render_img_path', 'frame': 'frame_path'},
    'single_step': {'render': 'image_path'},
}
MOVE_STEP_M = 0.3
LOOK_STEP_DEGREES = 15
TOOL_CONFIG = {
    'enable_observe': True,
    'enable_finish_task': True,
    'enable_action': True,
    'enable_reset': True,
    'enable_turn_left': True,
    'enable_turn_right': True,
    'enable_turn_up': True,
    'enable_turn_down': True,
    'enable_move_forward': True,
    'enable_move_backward': False,
    'enable_move_left': False,
    'enable_move_right': False,
    'enable_move_up': False,
    'enable_move_down': False,
    'enable_top_down': False,
}

SYSTEM_PROMPT = """You are a spatial agent exploring a reconstructed 3D point-cloud scene.
Use the available tools to gather evidence and complete the user's task.

Visual evidence
Inspect each new image before continuing, using its view label and any legend to interpret it.
Images describe the viewpoint when obtained. Point-cloud gaps may be reconstruction artifacts.
State uncertainty when the visual evidence is insufficient, and answer once you have enough evidence."""

USER_PROMPT = """Navigate from the initial view to the goal.
Goal type: first_person.
The initial view is a reference first-person video frame. The camera starts at its recorded pose.
The goal image is a reference, not a current observation.
The goal image is a reference first-person video frame.
Match both the camera location and the full viewing direction shown in that image.
Finding the same object alone does not establish that the camera viewpoint matches.
Use the camera tools to inspect the current rendered first-person scene and navigate toward the goal.
Rendered observations may contain point-cloud gaps and a different image crop from the reference video.
When you believe you have reached the goal, report success with visual evidence.
If you cannot reach it, report failure with a reason.
If finish_task is available, use it to report the outcome and end the task; otherwise respond in text."""


def _text_part(text):
    return {'type': 'text', 'text': text}


def _image_part(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing navigation image: {path}')
    return {'type': 'image', 'image': str(path)}


def _step_arguments(step):
    arguments = deepcopy(step['arguments'])
    if step['name'] == 'camera_action':
        actions = (arguments,)
    elif step['name'] == 'camera_actions':
        actions = arguments['actions']
    else:
        return arguments
    for action in actions:
        action['magnitude'] = max(0.1, round(float(action['magnitude']), 1))
    return arguments


def _action_result(action):
    # Both manifests describe successfully executed expert actions.
    magnitude = action['magnitude']
    return {'magnitude': magnitude, 'applied_magnitude': magnitude, 'blocked': False, 'reason': None}


def _observation_result(name, arguments, observation_id):
    result = {'observation_id': observation_id}
    if name == 'camera_action':
        result.update(_action_result(arguments))
    elif name == 'camera_actions':
        result['actions'] = [
            {'key': action['key'], **_action_result(action)}
            for action in arguments['actions']
        ]
    return json.dumps({'ok': True, 'result': result}, ensure_ascii=False)


class NavigationTrajectoryDataset(Dataset):
    """Return one complete conversation and its tool schemas per trajectory.

    Reference paths are relative to each scene under `data_root`. Rendered
    observations are relative to `observation_root` and may include its
    directory name as a prefix. Frame observations are relative to each
    scene under that root. The trajectory type and source select the
    observation field.
    """

    def __init__(
        self,
        trajectory_json: str | Path,
        data_root: str | Path,
        observation_root: str | Path,
        trajectory_type: Literal['multi_step', 'single_step'] = 'multi_step',
        observation_source: Literal['render', 'frame'] = 'render',
    ):
        if trajectory_type not in ACTION_TOOLS:
            raise ValueError("trajectory_type must be 'multi_step' or 'single_step'")
        observation_fields = OBSERVATION_FIELDS[trajectory_type]
        if observation_source not in observation_fields:
            raise ValueError(f'{trajectory_type} observation_source must be one of {", ".join(observation_fields)}')
        observation_field = observation_fields[observation_source]

        self.trajectory_json = Path(trajectory_json).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        self.observation_root = Path(observation_root).expanduser().resolve()
        self.observation_source = observation_source
        self.trajectory_type = trajectory_type
        route_tools = ('camera_observe', ACTION_TOOLS[trajectory_type])

        self.prompt = USER_PROMPT
        self.tools = [
            tool.schema()
            for tool in create_tools(
                None, move_step=MOVE_STEP_M, look_step=LOOK_STEP_DEGREES,
                enable_actions=trajectory_type == 'multi_step', **TOOL_CONFIG
            )
        ]
        self.trajectories = []
        self.skipped_missing_observations = 0
        with self.trajectory_json.open(encoding='utf-8') as file:
            manifest = json.load(file)
        for row in manifest['trajectories']:
            steps = row['route_actions']
            if (
                len(steps) < 2
                or steps[0]['name'] != 'camera_observe'
                or steps[-1]['name'] != 'finish_task'
                or any(step['name'] not in route_tools for step in steps[1:-1])
            ):
                raise ValueError(f"Invalid {trajectory_type} trajectory: {row['scene']}/{row['pair_id']}")
            # finish_task leaves the camera in place and has no observation.
            if any(not step.get(observation_field) for step in steps[:-1]):
                self.skipped_missing_observations += 1
                continue

            self.trajectories.append({
                'scene': row['scene'],
                'pair_id': row['pair_id'],
                'distance_group': row['distance_group'],
                'initial_view': row['initial_view'],
                'target_view': row['target_view'],
                'steps': [
                    {'name': step['name'], 'arguments': step['arguments'], 'image': step.get(observation_field)}
                    for step in steps
                ],
            })
        if not self.trajectories:
            raise ValueError(f'No trajectories with observation field {observation_field!r} in {self.trajectory_json}')

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        row = self.trajectories[index]
        scene_dir = self.data_root / row['scene']
        observation_dir = self.observation_root
        if self.observation_source == 'frame':
            observation_dir = observation_dir / row['scene']
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': [
                _text_part('<initial_view>'),
                _image_part(scene_dir / row['initial_view']),
                _text_part('</initial_view>\n<goal_view>'),
                _image_part(scene_dir / row['target_view']),
                _text_part(f'</goal_view>\n{self.prompt}'),
            ]},
        ]
        # The online runner records an initial observation before the first tool call.
        for observation_id, step in enumerate(row['steps'], start=2):
            name = step['name']
            arguments = _step_arguments(step)
            call_id = f'call_{observation_id - 1}'
            messages.append({
                'role': 'assistant',
                'content': '',
                'tool_calls': [{
                    'id': call_id,
                    'type': 'function',
                    'function': {'name': name, 'arguments': arguments},
                }],
            })
            if name == 'finish_task':
                break
            image_path = Path(step['image'])
            if self.observation_source == 'render' and image_path.is_relative_to(observation_dir.name):
                image_path = image_path.relative_to(observation_dir.name)
            messages.extend([
                {
                    'role': 'tool',
                    'tool_call_id': call_id,
                    'name': name,
                    'content': _observation_result(name, arguments, observation_id),
                },
                {'role': 'user', 'content': [
                    _text_part('<first_person_view>'),
                    _image_part(observation_dir / image_path),
                    _text_part('</first_person_view>'),
                ]},
            ])
        return {
            'messages': messages,
            'tools': self.tools,
            'scene': row['scene'],
            'pair_id': row['pair_id'],
            'distance_group': row['distance_group'],
        }


def main():
    import argparse
    import random
    import re
    from itertools import groupby

    parser = argparse.ArgumentParser(description='Inspect random first-person SFT trajectories.')
    parser.add_argument('--data-root', default='/data4/haibo/data/3D_Env')

    # parser.add_argument('--trajectory-type', choices=ACTION_TOOLS, default='multi_step')
    # parser.add_argument('--observation-source', choices=('render', 'frame'), default='render')
    # parser.add_argument('--trajectory-json', default='/data4/haibo/code/SpaceExp/outputs/benchmark_train_traj_real_multi_step.json')
    # parser.add_argument('--observation-root', default='/data4/haibo/data/3D_Env/train_traj_real_multi_step_renders')

    # parser.add_argument('--trajectory-type', choices=ACTION_TOOLS, default='multi_step')
    # parser.add_argument('--observation-source', choices=('render', 'frame'), default='frame')
    # parser.add_argument('--trajectory-json', default='/data4/haibo/code/SpaceExp/outputs/benchmark_train_traj_real_multi_step.json')
    # parser.add_argument('--observation-root', default='/data4/haibo/data/3D_Env')

    parser.add_argument('--trajectory-type', choices=ACTION_TOOLS, default='single_step')
    parser.add_argument('--observation-source', choices=('render', 'frame'), default='render')
    parser.add_argument('--trajectory-json', default='/data4/haibo/code/SpaceExp/outputs/benchmark_train_traj_sim_single_step.json')
    parser.add_argument('--observation-root', default='/data4/haibo/data/3D_Env/train_traj_sim_single_step_renders')


    parser.add_argument('--model-path', type=Path, default='/data4/haibo/weights/Qwen3.5-9B')
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error('--num-samples must be positive')

    from transformers import AutoProcessor

    from sft.utils.qwen import IGNORE_INDEX, encode_navigation_sample

    dataset = NavigationTrajectoryDataset(
        trajectory_json=args.trajectory_json,
        data_root=args.data_root,
        observation_root=args.observation_root,
        trajectory_type=args.trajectory_type,
        observation_source=args.observation_source,
    )
    processor = AutoProcessor.from_pretrained(args.model_path.expanduser(), local_files_only=True)
    indices = random.Random(args.seed).sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    image_pad = '<|image_pad|>'
    image_pad_runs = re.compile(rf'(?:{re.escape(image_pad)})+')
    print(f'{len(dataset):,} {dataset.trajectory_type} trajectories ({dataset.observation_source} observations)')
    for index in indices:
        sample = dataset[index]
        encoded = encode_navigation_sample(sample, processor)
        print(f"\nSample {index}  {sample['scene']}/{sample['pair_id']}  ({sample['distance_group']})")
        decoded = processor.batch_decode(
            encoded['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        print(image_pad_runs.sub(lambda run: f'{image_pad}*{run[0].count(image_pad)}', decoded))
        image_paths = [
            part['image']
            for message in sample['messages']
            if isinstance(message['content'], list)
            for part in message['content']
            if part['type'] == 'image'
        ]
        print('Image paths in conversation order')
        for path in image_paths:
            print(f'  {path}')
        for name, value in encoded.items():
            print(f'{name}.shape = {tuple(value.shape)}')
        print("encoded['labels'] = [[")
        supervised_spans = []
        for ignored, run in groupby(encoded['labels'][0].tolist(), key=lambda token: token == IGNORE_INDEX):
            tokens = list(run)
            if ignored:
                text = f'{IGNORE_INDEX}*{len(tokens)}'
            else:
                supervised_spans.append(tokens)
                text = ', '.join(map(str, tokens))
            print(f'  {text},')
        print(']]')
        print('Decoded labels (one block per supervised turn)')
        decoded_labels = processor.batch_decode(
            supervised_spans, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        print('\n\n'.join(decoded_labels))


if __name__ == '__main__':
    main()
