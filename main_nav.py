"""Run one image-goal navigation task from a recorded frame's camera pose."""

from datetime import datetime
import json
from pathlib import Path
import shutil

from agent import Agent
from agent.backend import VLLMBackend
from agent.protocols import create_protocol
from agent.tools import create_tools
from main_agent import (
    AGENT_CONFIG as DEFAULT_AGENT_CONFIG,
    MODEL,
    MODEL_GENERATION_CONFIGS,
    MODEL_SERVER_CONFIG,
    OUTPUT_DIR,
    SCENE_CONFIG as DEFAULT_SCENE_CONFIG,
    SYSTEM_PROMPT,
    TOOL_CONFIG as DEFAULT_TOOL_CONFIG,
)
from scene.client import CameraClient


SCENE_CONFIG = {
    **DEFAULT_SCENE_CONFIG,
    'scene_dir': '/data4/haibo/data/3D_Env/scannetpp/307e3262f1',
}
# Example frames. Paths may be absolute or relative to scene_dir.
INITIAL_VIEW = 'frames/frame_000000.jpg'
GOAL_VIEW = 'frames/frame_012039.jpg'
PROMPT = """Navigate from the initial view to the location and viewing direction shown in the goal view.
The camera starts at the recorded pose of the initial view. The goal view is a reference, not a current observation.
Use the camera tools to inspect the current rendered scene and navigate toward the goal.
Match both the viewpoint and viewing direction, rather than merely finding the same object.
The reference images are video frames; rendered observations may contain point-cloud gaps and a different image crop.
When you believe you have reached the goal viewpoint, stop and explain the visual evidence. If you cannot reach it, say so."""
AGENT_CONFIG = {
    **DEFAULT_AGENT_CONFIG,
    'preserve_prompt_images': True,  # Retain both references while trimming older observations.
}
TOOL_CONFIG = {
    **DEFAULT_TOOL_CONFIG,
    'enable_move_backward': False,
    'enable_move_left': False,
    'enable_move_right': False,
    'enable_top_down': False,  # Optional map tool for navigation experiments.
}


def reference_frame(scene_dir, image, cameras):
    path = Path(image).expanduser()
    path = (path if path.is_absolute() else scene_dir / path).resolve()
    if not path.is_relative_to((scene_dir / 'frames').resolve()):
        raise ValueError(f'Reference image must be inside the scene frames directory: {path}')
    if not path.is_file():
        raise FileNotFoundError(path)
    matches = [frame for frame in cameras
               if (scene_dir / frame['image_path']).resolve() == path]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one camera record for {path}; found {len(matches)}')
    return path, matches[0]


def image_content(path):
    return {'type': 'image_url', 'image_url': {'url': str(path.resolve())}}


def main():
    scene_dir = Path(SCENE_CONFIG['scene_dir']).expanduser().resolve()
    cameras = json.loads((scene_dir / 'cameras.json').read_text())
    initial_path, initial_camera = reference_frame(scene_dir, INITIAL_VIEW, cameras)
    goal_path, goal_camera = reference_frame(scene_dir, GOAL_VIEW, cameras)
    config = {**SCENE_CONFIG, 'scene_dir': str(scene_dir),
              'initial_camera': initial_camera, 'target_camera': goal_camera}
    output = (Path(OUTPUT_DIR).expanduser().resolve()
              / datetime.now().strftime('nav_%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    initial_copy = output / f'initial_view{initial_path.suffix.lower()}'
    goal_copy = output / f'target_view{goal_path.suffix.lower()}'
    shutil.copy2(initial_path, initial_copy)
    shutil.copy2(goal_path, goal_copy)
    content = [
        {'type': 'text', 'text': '<initial_view>'},
        image_content(initial_copy),
        {'type': 'text', 'text': '</initial_view>\n<goal_view>'},
        image_content(goal_copy),
        {'type': 'text', 'text': f'</goal_view>\n{PROMPT}'},
    ]
    # Camera parameters are local metadata; the model receives images and task text only.
    (output / 'task.json').write_text(json.dumps({
        'scene_dir': str(scene_dir), 'initial_view': str(initial_copy), 'goal_view': str(goal_copy),
        'prompt': PROMPT, 'initial_camera': initial_camera, 'goal_camera': goal_camera,
        'tool_config': TOOL_CONFIG,
    }, indent=2) + '\n')

    model_config = MODEL_GENERATION_CONFIGS[MODEL]
    backend = VLLMBackend(
        MODEL, **MODEL_SERVER_CONFIG,
        **{key: model_config[key] for key in ('max_tokens', 'temperature', 'top_p', 'top_k')},
    )
    try:
        with CameraClient(config['blender'], config, output, timeout=config['timeout']) as camera:
            camera.observe()  # Record the actual starting pose and its local trajectory map.
            agent = Agent(
                backend,
                create_protocol(MODEL, **{key: model_config[key] for key in (
                    'enable_thinking', 'reasoning_effort', 'preserve_thinking')}),
                create_tools(camera, move_step=config['move_step'], look_step=config['look_step'], **TOOL_CONFIG),
                system_prompt=SYSTEM_PROMPT, **AGENT_CONFIG,
                conversation_path=output / 'conversation.json',
            )
            print(f'Agent model: {MODEL}\nInitial view: {initial_path}\nGoal view: {goal_path}\nOutput: {output}',
                  flush=True)
            print('Agent:', agent.run(content), flush=True)
    finally:
        backend.client.close()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
