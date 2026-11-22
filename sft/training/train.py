"""Train Qwen3.5 on first-person navigation trajectories."""

from dataclasses import dataclass
import logging
from pathlib import Path
import sys

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, HfArgumentParser, Qwen3_5ForConditionalGeneration, Trainer
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.trainer_utils import get_last_checkpoint

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sft.dataset.navigation import NavigationTrajectoryDataset
from sft.training.argument import DataArguments, ModelArguments, TrainingArguments
from sft.utils.qwen import IGNORE_INDEX, encode_navigation_sample


def set_model(model_args, model):
    model.model.visual.requires_grad_(model_args.tune_mm_vision)
    model.model.visual.merger.requires_grad_(model_args.tune_mm_mlp)
    model.model.language_model.requires_grad_(model_args.tune_mm_llm)
    model.lm_head.requires_grad_(model_args.tune_mm_llm)
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Enable at least one tune_mm_* option to train the model.')


def supervised_positions(labels):
    """Select the sequence positions that predict a supervised next token."""
    return labels[:, 1:].ne(IGNORE_INDEX).any(dim=0).nonzero(as_tuple=True)[0]


def compute_loss(outputs, labels, num_items_in_batch=None):
    targets = labels[:, 1:].index_select(1, supervised_positions(labels))
    return ForCausalLMLoss(
        logits=outputs.logits,
        labels=labels,
        vocab_size=outputs.logits.size(-1),
        num_items_in_batch=num_items_in_batch,
        ignore_index=IGNORE_INDEX,
        shift_labels=targets,
    )


@dataclass
class DataCollatorForSupervisedDataset:
    processor: AutoProcessor
    model_max_length: int

    def __call__(self, instances) -> dict[str, torch.Tensor]:
        if self.model_max_length <= 0:
            raise ValueError('--model_max_length must be positive.')
        tokenizer = self.processor.tokenizer
        padding_values = {
            'input_ids': tokenizer.pad_token_id,
            'labels': IGNORE_INDEX,
            'attention_mask': 0,
            'mm_token_type_ids': 0,
        }
        encoded = []
        for sample in instances:
            item = encode_navigation_sample(sample, self.processor)
            if item['input_ids'].size(-1) > self.model_max_length:
                prefix = item['input_ids'][0, :self.model_max_length]
                vision_end = tokenizer.convert_tokens_to_ids('<|vision_end|>')
                image_count = int(prefix.eq(vision_end).sum())
                vision_start = tokenizer.convert_tokens_to_ids('<|vision_start|>')
                image_starts = prefix.eq(vision_start).nonzero(as_tuple=True)[0]
                # A cutoff inside an image leaves its partial token span padded.
                pad_from = int(image_starts[-1]) if len(image_starts) > image_count else self.model_max_length
                for name, value in padding_values.items():
                    item[name] = item[name][:, :self.model_max_length].clone()
                    item[name][:, pad_from:] = value
                grid = item['image_grid_thw'][:image_count]
                patch_count = int(grid.prod(dim=1).sum())
                item['image_grid_thw'] = grid.clone()
                item['pixel_values'] = item['pixel_values'][:patch_count].clone()
            encoded.append(item)

        if len(encoded) == 1:
            batch = dict(encoded[0])
        else:
            batch = {
                name: pad_sequence([item[name][0] for item in encoded], batch_first=True, padding_value=value)
                for name, value in padding_values.items()
            }
            for name in ('pixel_values', 'image_grid_thw'):
                batch[name] = torch.cat([item[name] for item in encoded], dim=0)

        # Project only states that contribute to the loss.
        batch['logits_to_keep'] = supervised_positions(batch['labels'])
        if not batch['logits_to_keep'].numel():
            raise ValueError('The batch has no supervised tokens. Increase --model_max_length.')
        return batch


def make_supervised_data_module(processor, data_args, model_max_length):
    train_dataset = NavigationTrajectoryDataset(**vars(data_args))
    return {
        'train_dataset': train_dataset,
        'eval_dataset': None,
        'data_collator': DataCollatorForSupervisedDataset(processor, model_max_length),
    }


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            'use_reentrant': False, **(training_args.gradient_checkpointing_kwargs or {})
        }
    logging.basicConfig(level=training_args.get_process_log_level())

    output_dir = Path(training_args.output_dir)
    resume = training_args.resume_from_checkpoint
    if resume is None and output_dir.is_dir():
        resume = get_last_checkpoint(output_dir)

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = training_args.model_max_length
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_module = make_supervised_data_module(processor, data_args, training_args.model_max_length)
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir, attn_implementation="flash_attention_2", dtype=dtype
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.text_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    set_model(model_args, model)

    if training_args.process_index == 0:
        # ZeRO-3 may hold an empty shard in parameter.numel().
        trainable = sum(
            getattr(parameter, 'ds_numel', parameter.numel())
            for parameter in model.parameters() if parameter.requires_grad
        )
        print(f'Training trajectories: {len(data_module["train_dataset"]):,}')
        print(f'Trainable parameters: {trainable:,}')
        if resume:
            print(f'Resuming training from {resume}')

    trainer = Trainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_loss_func=compute_loss,
        **data_module,
    )

    use_cache = model.config.text_config.use_cache
    model.config.text_config.use_cache = False
    try:
        trainer.train(resume_from_checkpoint=resume)
    finally:
        model.config.text_config.use_cache = use_cache
    trainer.save_state()
    trainer.save_model()


if __name__ == '__main__':
    train()
