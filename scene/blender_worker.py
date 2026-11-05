"""Run the scene in Blender and answer camera requests over a response pipe."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene.renderer import SceneRenderer


def redirect_log(output):
    sys.stdout.flush()
    sys.stderr.flush()
    with (Path(output) / 'render.log').open('a') as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)


def main():
    arguments = sys.argv[sys.argv.index('--') + 1:]
    options = json.loads(arguments[0])
    response_fd = int(arguments[1])
    renderer = SceneRenderer(options['config'], options['output'])
    with os.fdopen(response_fd, 'w', encoding='utf-8', buffering=1) as responses:
        responses.write(json.dumps({'ok': True, 'result': None}) + '\n')
        for line in sys.stdin:
            closing = False
            try:
                request = json.loads(line)
                operation = request['op']
                params = request.get('params', {})
                if operation == 'observe':
                    result = renderer.observe()
                elif operation == 'action':
                    result = renderer.action(**params)
                elif operation == 'start_session':
                    renderer.start_session(params['output'])
                    redirect_log(params['output'])
                    result = None
                elif operation == 'close':
                    result = None
                    closing = True
                else:
                    raise ValueError(f'Unknown camera operation: {operation}')
                reply = {'ok': True, 'result': result}
            except Exception as error:
                reply = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
            sys.stdout.flush()
            sys.stderr.flush()
            responses.write(json.dumps(reply) + '\n')
            if closing:
                break


if __name__ == '__main__':
    main()
