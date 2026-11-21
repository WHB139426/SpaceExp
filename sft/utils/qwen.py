"""Format full navigation trajectories for action-only Qwen3.5 SFT."""

from __future__ import annotations

import torch


IGNORE_INDEX = -100
ASSISTANT_PREFIX = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
ASSISTANT_TEMPLATE_CONDITION = '{%- if loop.index0 > ns.last_query_index %}'


def encode_navigation_sample(sample, processor):
    """Encode a full trajectory and supervise each tool call through its end token."""
    template = processor.chat_template
    if not isinstance(template, str) or template.count(ASSISTANT_TEMPLATE_CONDITION) != 1:
        raise ValueError('Expected the Qwen3.5 chat template with its assistant turn condition')

    # Render the same empty thinking prefix as non-thinking generation on each
    # assistant turn. The processor's original template still serves inference.
    training_template = template.replace(ASSISTANT_TEMPLATE_CONDITION, '{%- if true %}', 1)
    encoded = processor.apply_chat_template(
        sample['messages'],
        tools=sample['tools'],
        chat_template=training_template,
        enable_thinking=False,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
    )

    tokenizer = processor.tokenizer
    prefix_ids = tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
    tool_start_ids = tokenizer.encode('<tool_call>', add_special_tokens=False)
    turn_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    input_ids = encoded['input_ids']
    tokens = input_ids[0].tolist()
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    supervised_turns = 0
    for start in (input_ids[0] == prefix_ids[0]).nonzero(as_tuple=True)[0].tolist():
        tool_start = start + len(prefix_ids)
        if tokens[start:tool_start] != prefix_ids:
            continue
        if tokens[tool_start:tool_start + len(tool_start_ids)] != tool_start_ids:
            raise ValueError('Expected a tool call after the empty thinking prefix')
        end = tokens.index(turn_end_id, tool_start) + 1
        labels[0, tool_start:end] = input_ids[0, tool_start:end]
        supervised_turns += 1

    expected_turns = sum(message['role'] == 'assistant' for message in sample['messages'])
    if not expected_turns or supervised_turns != expected_turns:
        raise ValueError(f'Expected {expected_turns} assistant turns, found {supervised_turns}')
    encoded['labels'] = labels
    return encoded
