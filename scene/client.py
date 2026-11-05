"""Own one Blender process and exchange sequential camera requests."""

import json
import os
from pathlib import Path
import select
import subprocess
import sys

from .trajectory import draw_trajectory


class CameraClient:
    def __init__(self, blender, config, output, *, timeout=300):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.log_path = output / 'render.log'
        self.timeout = timeout
        self.latest_observation = None
        options = json.dumps({'config': config, 'output': str(output)}, allow_nan=False)
        read_fd, write_fd = os.pipe()
        command = [str(blender), '--background', '--factory-startup', '--python-exit-code', '1',
                   '--python', str(Path(__file__).with_name('blender_worker.py')), '--', options, str(write_fd)]
        try:
            with self.log_path.open('a') as log:
                self.process = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', bufsize=1, pass_fds=(write_fd,),
                )
        except BaseException:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
        self.responses = os.fdopen(read_fd, encoding='utf-8')
        try:
            self._read_result()
        except BaseException:
            self._stop()
            raise

    def _read_result(self):
        try:
            if not select.select([self.responses], [], [], self.timeout)[0]:
                raise TimeoutError(f'Blender response timed out; log: {self.log_path}')
            line = self.responses.readline()
            if not line:
                raise RuntimeError(f'Blender response pipe closed; log: {self.log_path}')
            response = json.loads(line)
        except BaseException:
            # An interrupted exchange cannot safely be reused for another request.
            self._stop()
            raise
        if not response['ok']:
            raise RuntimeError(f'{response["error"]}; log: {self.log_path}')
        return response['result']

    def _request(self, operation, **params):
        request = json.dumps({'op': operation, 'params': params}, allow_nan=False)
        try:
            self.process.stdin.write(request + '\n')
            self.process.stdin.flush()
        except (OSError, ValueError) as error:
            self._stop()
            raise RuntimeError(f'Cannot send camera request; log: {self.log_path}') from error
        return self._read_result()

    def observe(self):
        return self._observation('observe')

    def action(self, key, magnitude=None):
        return self._observation('action', key=key, magnitude=magnitude)

    def top_down(self):
        if self.latest_observation is None:
            raise RuntimeError('No observation is available yet; call camera_observe first.')
        record = self.latest_observation
        image = Path(record['image'])
        metadata = json.loads(image.with_suffix('.json').read_text())
        top_down = Path(metadata['top_down_image'])
        if not top_down.is_file():
            observations = json.loads((image.parent / 'trajectory.json').read_text())
            draw_trajectory(metadata['top_down_frame'], observations, top_down)
        # Use the saved step image: top_down.png will be overwritten by later actions.
        return {'observation_id': record['observation_id'], 'image': str(top_down)}

    def _observation(self, operation, **params):
        record = self._request(operation, **params)
        self.latest_observation = record
        try:
            metadata = json.loads(Path(record['image']).with_suffix('.json').read_text())
            observations = json.loads((Path(record['image']).parent / 'trajectory.json').read_text())
            draw_trajectory(metadata['top_down_frame'], observations, metadata['top_down_image'])
        except Exception as error:
            # The action has completed. A local chart failure must not hide its observation.
            print(f'Local trajectory plot failed for {record["image"]}: {error}', file=sys.stderr)
        return record

    def start_session(self, output):
        self._request('start_session', output=str(output))
        self.latest_observation = None
        self.log_path = Path(output) / 'render.log'

    def _stop(self):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        self.responses.close()
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def close(self):
        try:
            if self.process.poll() is None:
                self._request('close')
            self.process.wait(timeout=5)
        finally:
            self._stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self._stop()
