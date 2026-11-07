"""Explore an existing point-cloud scene with an agent and a persistent Blender worker."""

from datetime import datetime
from pathlib import Path

from agent import Agent
from agent.models import create_model
from agent.tools import create_tools
from scene.client import CameraClient


MODEL = "qwen3.8-27b"
MODEL_GENERATION_CONFIGS = {
    "qwen3.8-27b": {
        "max_tokens": 16 * 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "enable_thinking": True,
        "reasoning_effort": "medium",  # "low", "medium", "xhigh"
        "preserve_thinking": True,
    },
    "qwen3.8-flash-next": {
        "max_tokens": 16 * 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "enable_thinking": True,
        "reasoning_effort": "medium",  # "low", "medium", "xhigh"
        "preserve_thinking": True,
    },
    "gpt-6-astra": {
        "max_output_tokens": 32 * 1024,
        "reasoning_effort": "high", # "low", "medium", "high", "xhigh", "max"
        "reasoning_summary": "auto",
    },
    "gpt-6.1-sol": {
        "max_output_tokens": 16 * 1024,
        "reasoning_effort": "medium",  # "low", "medium", "high", "xhigh", "max"
        "reasoning_summary": "auto",
    },
    "gpt-6.1-terra": {
        "max_output_tokens": 16 * 1024,
        "reasoning_effort": "medium", # "low", "medium", "high", "xhigh", "max"
        "reasoning_summary": "auto",
    },
}
MODEL_SERVER_CONFIG = {
    "vllm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "timeout": 180,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "timeout": 300,
        # api_key is read from OPENAI_API_KEY by the SDK.
    },
}
AGENT_CONFIG = {
    "max_steps": 100,
    "max_conversation_images": 7,
    "verbose": True,  # Print tool calls, OpenAI reasoning summaries and usage.
}
TOOL_CONFIG = {
    'enable_move_backward': True,
    'enable_move_left': True,
    'enable_move_right': True,
    'enable_top_down': False,
}
SCENE_CONFIG = {
    "scene_dir": "/data4/haibo/data/3D_Env/scannetpp/307e3262f1",
    "blender": "/data4/haibo/software/blender/blender-4.5.13-linux-x64/blender",
    "timeout": 300,
    "resolution": 960,
    "samples": 16,
    "trajectory_resolution": 960,
    "trajectory_samples": 4,
    "trajectory_max_height": 1.8,
    "trajectory_dpi": 180,
    "trajectory_arrow_length": 0.25,
    "boundary_enforced": True,  # Reject translations whose full path leaves the inset region.
    "boundary_preview": True,  # Show the boundary on the local top-down chart.
    "boundary_cell_size": 0.05,
    "boundary_close_radius": 0.15,
    "camera_radius": 0.15,
    "point_radius": 0.015,
    "move_step": 0.3,
    "look_step": 15,
    "up": None,  # Scene up for ground movement, yaw, and the map; None estimates it from the scene.
}
OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")

SYSTEM_PROMPT = """You are a spatial agent exploring a reconstructed 3D point-cloud scene.
Use the available tools to gather evidence and complete the user's task.

Visual evidence
Inspect each new image before continuing, using its view label and any legend to interpret it.
Images describe the viewpoint when obtained. Point-cloud gaps may be reconstruction artifacts.
State uncertainty when the visual evidence is insufficient, and answer once you have enough evidence."""


def new_conversation_path() -> Path:
    scene_name = Path(SCENE_CONFIG['scene_dir']).expanduser().resolve().name
    return (Path(OUTPUT_DIR).expanduser().resolve()
            / f"chat_{scene_name}_{MODEL}_{datetime.now():%Y%m%d_%H%M%S_%f}" / "conversation.json")


def run_conversation(backend, protocol, camera, conversation_path):
    agent = Agent(
        backend,
        protocol,
        create_tools(camera, move_step=SCENE_CONFIG["move_step"], look_step=SCENE_CONFIG["look_step"], **TOOL_CONFIG),
        system_prompt=SYSTEM_PROMPT, **AGENT_CONFIG,
        conversation_path=conversation_path,
    )
    print(f"Agent model: {MODEL}\nScene: {SCENE_CONFIG['scene_dir']}\nConversation: {agent.conversation_path}")
    print("Ask a question. /new starts a fresh exploration; /exit quits.")
    while True:
        try:
            prompt = input("agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"/exit", "exit", "quit"}:
            break
        if prompt.lower() == "/new":
            next_path = new_conversation_path()
            camera.start_session(next_path.parent)
            agent.conversation_path = next_path
            agent.reset()
            print(f"New exploration: {agent.conversation_path}", flush=True)
            continue
        if prompt:
            try:
                print("Agent:", agent.run(prompt), flush=True)
            except Exception as error:
                print(f"Error: {error}", flush=True)


def main():
    model_config = MODEL_GENERATION_CONFIGS[MODEL]
    backend, protocol = create_model(MODEL, model_config, MODEL_SERVER_CONFIG)
    try:
        config = {
            **SCENE_CONFIG,
            'scene_dir': str(Path(SCENE_CONFIG['scene_dir']).expanduser().resolve()),
        }
        conversation_path = new_conversation_path()
        with CameraClient(config['blender'], config, conversation_path.parent, timeout=config['timeout']) as camera:
            run_conversation(backend, protocol, camera, conversation_path)
    finally:
        backend.client.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
