"""Translate shared history to Responses items, preserving opaque reasoning state."""

import copy
import json

from ..types import Message, ModelInput, ModelOutput


def function_schema(schema):
    """Adapt existing tool schemas to Responses strict mode without changing tools."""
    function = copy.deepcopy(schema['function'])
    parameters = function['parameters']
    properties = parameters['properties']
    required = parameters.get('required', [])
    for name, definition in properties.items():
        if name not in required:
            # Existing camera tools use flat arguments; null retains Python defaults.
            parameters['properties'][name] = {'anyOf': [definition, {'type': 'null'}]}
    parameters['required'] = list(properties)
    parameters['additionalProperties'] = False
    return {'type': 'function', **function, 'strict': True}


class OpenAIProtocol:
    def render(self, messages: list[Message], tools: list[dict[str, object]]) -> ModelInput:
        items = []
        instructions = []
        for message in messages:
            role = message['role']
            if role == 'system':
                instructions.append(message['content'])
            elif role == 'assistant':
                # Replay every original item, including encrypted reasoning, call IDs,
                # message phases and refusals. Summary text alone cannot replace it.
                if 'response_items' not in message:
                    raise ValueError('OpenAI history requires original response_items')
                items.extend(copy.deepcopy(message['response_items']))
            elif role == 'tool':
                items.append({'type': 'function_call_output',
                              'call_id': message['tool_call_id'], 'output': message['content']})
            elif role == 'user':
                content = message['content']
                if isinstance(content, list):
                    converted = []
                    for part in content:
                        if part['type'] == 'text':
                            converted.append({'type': 'input_text', 'text': part['text']})
                        elif part['type'] == 'image_url':
                            image = copy.deepcopy(part['image_url'])
                            converted.append({'type': 'input_image', 'image_url': image.pop('url'), **image})
                        else:
                            raise ValueError(f'Unsupported content type: {part["type"]}')
                    content = converted
                items.append({'role': 'user', 'content': content})
            else:
                raise ValueError(f'Unsupported message role: {role}')
        return ModelInput(messages=items, tools=[function_schema(tool) for tool in tools],
                          instructions='\n\n'.join(instructions))

    def parse(self, response: ModelOutput) -> Message:
        if not isinstance(response, dict):
            raise TypeError('OpenAI protocol expects a structured Responses result')
        items = copy.deepcopy(response['output'])
        texts, summaries, calls = [], [], []
        for item in items:
            if item['type'] == 'reasoning':
                summaries.extend(part['text'] for part in item.get('summary', [])
                                 if part['type'] == 'summary_text')
            elif item['type'] == 'message':
                for part in item['content']:
                    if part['type'] == 'output_text':
                        texts.append(part['text'])
                    elif part['type'] == 'refusal':
                        texts.append(part['refusal'])
            elif item['type'] == 'function_call':
                arguments = json.loads(item['arguments'])
                if not isinstance(arguments, dict):
                    raise ValueError('Tool arguments must be a JSON object')
                calls.append({'id': item['call_id'], 'type': 'function',
                              'function': {'name': item['name'], 'arguments': arguments}})
            else:
                raise ValueError(f'Unexpected OpenAI output item: {item["type"]}')
        if not calls and not texts:
            raise RuntimeError('OpenAI returned neither a tool call nor an answer')
        message = {
            'role': 'assistant', 'content': '\n\n'.join(texts),
            'reasoning_summary': '\n\n'.join(summaries),
            'response_items': items, 'response_id': response['id'],
            'usage': response.get('usage'),
        }
        if calls:
            message['tool_calls'] = calls
        elif all(item.get('phase') == 'commentary' for item in items if item['type'] == 'message'):
            # A standalone progress message is not a final answer.
            message['continue_generation'] = True
        return message
