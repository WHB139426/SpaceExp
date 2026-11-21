"""Command-line arguments for first-person navigation SFT."""

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, List
from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='/data4/haibo/weights/Qwen3.5-9B')
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=True)

@dataclass
class DataArguments:
    data_root: Optional[str] = field(default='/data4/haibo/data/3D_Env')
    trajectory_json: Optional[str] = field(default='/data4/haibo/code/SpaceExp/outputs/benchmark_train_traj_real_multi_step.json')
    trajectory_type: Literal['multi_step', 'single_step'] = 'multi_step'
    observation_source: Literal['render', 'frame'] = 'render'


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=48 * 1024)
