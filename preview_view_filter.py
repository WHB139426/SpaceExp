"""Preview a conservative image-structure filter without changing the benchmark."""

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm


DATA_ROOT = Path('/data4/haibo/data/3D_Env')
BENCHMARK_JSON = Path(__file__).resolve().parent / 'outputs/benchmark_test.json'
OUTPUT_ROOT = Path(__file__).resolve().parent / 'outputs'
EXTRA_SCENES = ['scannetpp/1ada7a0617', 'scannetpp/5ee7c22ba0']
EXTRA_FRAMES_PER_SCENE = 24
FEATURE_LONG_EDGE = 384
BORDER_CROP = 0.03
MIN_NORMALIZATION_SPAN = 40
GRID_SIZE = 4
MIN_CONTRAST = 8
MIN_SHARPNESS = 10
MIN_CELL_EDGE_DENSITY = 0.015
MIN_CELL_ENTROPY = 1.5  # Bits, after quantizing gray levels into 32 bins.
MIN_KEEP_CORNERS = 30
MIN_KEEP_COVERAGE = 0.375
MIN_KEEP_SCORE = 45
PREVIEW_PER_CATEGORY = 12
DETAIL_PER_CATEGORY = 4
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def entropy(gray):
    counts = np.bincount((gray.ravel() // 8), minlength=32)
    probabilities = counts[counts > 0] / gray.size
    return float(-(probabilities * np.log2(probabilities)).sum())


def analyze(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        old = image.convert('L')
        old.thumbnail((320, 240))
        legacy = np.asarray(old, dtype=float)
        contrast = float(legacy.std())
        lap = legacy[2:, 1:-1] + legacy[:-2, 1:-1] + legacy[1:-1, 2:] + legacy[1:-1, :-2] - 4 * legacy[1:-1, 1:-1]
        sharpness = float(lap.var())
        image.thumbnail((FEATURE_LONG_EDGE, FEATURE_LONG_EDGE))
        rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    by, bx = round(h * BORDER_CROP), round(w * BORDER_CROP)
    rgb = rgb[by:h - by, bx:w - bx].copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Normalize exposure before measuring structure; limit amplification of noise.
    low, high = np.percentile(gray, [2, 98])
    gray = np.clip((gray.astype(float) - low) * 255 / max(float(high - low), MIN_NORMALIZATION_SPAN), 0, 255).astype(np.uint8)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.5)
    edges = cv2.Canny(smooth, 50, 120, L2gradient=True)
    mask = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    points = cv2.goodFeaturesToTrack(smooth, maxCorners=300, qualityLevel=.02,
                                   minDistance=8, blockSize=7, mask=mask)
    points = np.empty((0, 2)) if points is None else points[:, 0, :]
    h, w = gray.shape
    cells = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x0, x1 = col * w // GRID_SIZE, (col + 1) * w // GRID_SIZE
            y0, y1 = row * h // GRID_SIZE, (row + 1) * h // GRID_SIZE
            density = float((edges[y0:y1, x0:x1] > 0).mean())
            information = entropy(gray[y0:y1, x0:x1])
            corners = int(((points[:, 0] >= x0) & (points[:, 0] < x1) &
                           (points[:, 1] >= y0) & (points[:, 1] < y1)).sum())
            active = density >= MIN_CELL_EDGE_DENSITY and corners >= 1 and information >= MIN_CELL_ENTROPY
            cells.append({'box': [x0, y0, x1, y1], 'edge_density': density,
                          'entropy': information, 'corners': corners, 'active': bool(active)})
    coverage = sum(cell['active'] for cell in cells) / len(cells)
    edge_density = float((edges > 0).mean())
    local_entropy = float(np.mean([cell['entropy'] for cell in cells]))
    score = 100 * (.35 * min(coverage / .6, 1) + .30 * min(len(points) / 60, 1)
                   + .25 * min(edge_density / .07, 1) + .10 * min(local_entropy / 3, 1))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 35, minLineLength=min(h, w) * .15, maxLineGap=8)
    parallel_share = 0.0
    if lines is not None and len(lines) >= 4:
        delta = lines[:, 0, 2:] - lines[:, 0, :2]
        angles = np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) % 180
        weights = np.linalg.norm(delta, axis=1)
        histogram = np.bincount((angles // 10).astype(int), weights=weights, minlength=18)
        parallel_share = float((np.roll(histogram, 1) + histogram + np.roll(histogram, -1)).max() / weights.sum())
    legacy_pass = contrast >= MIN_CONTRAST and sharpness >= MIN_SHARPNESS
    if not legacy_pass:
        decision, reason = 'reject', 'low_contrast' if contrast < MIN_CONTRAST else 'blurred'
    elif (edge_density < .006 and len(points) < 12) or (len(points) < 10 and coverage < .3125):
        decision, reason = 'reject', 'sparse_structure'
    elif (score < MIN_KEEP_SCORE or len(points) < MIN_KEEP_CORNERS or coverage < MIN_KEEP_COVERAGE):
        decision, reason = 'review', 'limited_structure'
    elif parallel_share >= .8 and len(points) < 60:
        decision, reason = 'review', 'mostly_parallel_edges'
    else:
        decision, reason = 'keep', 'sufficient_structure'
    metrics = {'decision': decision, 'reason': reason, 'score': round(score, 2),
               'legacy_pass': legacy_pass, 'contrast': contrast, 'sharpness': sharpness,
               'edge_density': edge_density, 'corner_count': len(points), 'coverage': coverage,
               'mean_local_entropy': local_entropy, 'parallel_share': parallel_share}
    return metrics, (rgb, edges, points, cells)


def collect(tasks):
    records = {}
    for task in tasks:
        for role in ['initial_view', 'target_view']:
            path = (DATA_ROOT / task['scene'] / task[role]).resolve()
            record = records.setdefault(str(path), {'image': str(path), 'scene': task['scene'], 'uses': []})
            record['uses'].append({'task_id': task['id'], 'role': role})
    for scene in EXTRA_SCENES:
        paths = sorted((DATA_ROOT / scene / 'frames').glob('*.jpg'))
        indices = np.linspace(0, len(paths) - 1, min(len(paths), EXTRA_FRAMES_PER_SCENE), dtype=int)
        for index in indices:
            path = str(paths[index].resolve())
            records.setdefault(path, {'image': path, 'scene': scene, 'uses': []})
    return list(records.values())


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def select_examples(records, decision, count):
    # Prefer newly flagged views that already passed the existing quality checks.
    pool = [r for r in records if r['decision'] == decision]
    pool.sort(key=lambda r: (not r['legacy_pass'], r['score'], r['image']))
    selected, scenes = [], set()
    for record in pool:
        if record['scene'] not in scenes:
            selected.append(record)
            scenes.add(record['scene'])
        if len(selected) == count:
            break
    return selected


def card(record, size=(400, 355)):
    color = {'keep': '#16854a', 'review': '#b97700', 'reject': '#c42d33'}[record['decision']]
    canvas = Image.new('RGB', size, 'white')
    with Image.open(record['image']) as source:
        photo = ImageOps.contain(source.convert('RGB'), (size[0] - 12, 260))
    canvas.paste(photo, ((size[0] - photo.width) // 2, 5 + (260 - photo.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=color, width=4)
    lines = [f"{record['decision'].upper()} | score {record['score']:.1f} | old {'PASS' if record['legacy_pass'] else 'FAIL'}",
             f"{record['scene']} / {Path(record['image']).stem}",
             f"edge {record['edge_density']:.1%} | corners {record['corner_count']} | cells {record['coverage']:.0%}",
             record['reason']]
    for i, line in enumerate(lines):
        draw.text((10, 271 + 19 * i), line, fill=color if i == 0 else '#202020', font=font(14))
    return canvas


def sheet(records, output, title, columns=3):
    rows = max(1, (len(records) + columns - 1) // columns)
    canvas = Image.new('RGB', (columns * 400, 55 + rows * 355), '#ececec')
    ImageDraw.Draw(canvas).text((15, 15), title, fill='#202020', font=font(23))
    for i, record in enumerate(records):
        canvas.paste(card(record), ((i % columns) * 400, 55 + (i // columns) * 355))
    canvas.save(output, quality=92)


def detail(record, output):
    _, (rgb, edges, points, cells) = analyze(record['image'])
    overlay = rgb.copy()
    overlay[edges > 0] = [255, 60, 60]
    for x, y in points:
        cv2.circle(overlay, (round(float(x)), round(float(y))), 2, (255, 240, 0), -1)
    grid = Image.fromarray(rgb)
    tint = Image.new('RGBA', grid.size)
    draw = ImageDraw.Draw(tint)
    for cell in cells:
        color = (0, 180, 80, 65) if cell['active'] else (230, 30, 30, 65)
        x0, y0, x1, y1 = cell['box']
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=color, outline='white', width=1)
        draw.text((x0 + 3, y0 + 3), f"{cell['corners']} pts", fill='white', font=font(12))
    grid = Image.alpha_composite(grid.convert('RGBA'), tint).convert('RGB')
    canvas = Image.new('RGB', (1200, 420), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (picture, title) in enumerate([(Image.fromarray(rgb), 'Original (3% border excluded)'),
                                         (Image.fromarray(overlay), 'Edges: red | Corners: yellow'),
                                         (grid, '4x4 grid: green = active structure')]):
        fitted = ImageOps.contain(picture, (390, 300))
        canvas.paste(fitted, (i * 400 + (400 - fitted.width) // 2, 35 + (300 - fitted.height) // 2))
        draw.text((i * 400 + 10, 10), title, fill='black', font=font(16))
    draw.text((10, 345), f"{record['decision'].upper()} | {record['scene']} / {Path(record['image']).name} | {record['reason']}", fill='black', font=font(18))
    draw.text((10, 375), f"Score {record['score']:.1f} | Edges {record['edge_density']:.1%} | Corners {record['corner_count']} | Active cells {record['coverage']:.0%} | Local entropy {record['mean_local_entropy']:.2f}", fill='black', font=font(16))
    canvas.save(output, quality=93)


def save_report(output, records, tasks, original):
    previews = {}
    for decision in ['reject', 'review', 'keep']:
        selected = select_examples(records, decision, PREVIEW_PER_CATEGORY)
        previews[decision] = selected
        sheet(selected, output / f'{decision}.jpg', f'{decision.upper()}: {len(selected)} scene-diverse examples')
        for record in selected[:DETAIL_PER_CATEGORY]:
            detail(record, output / 'details' / f"{decision}_{Path(record['scene']).name}_{Path(record['image']).stem}.jpg")
    previous_examples = []
    for scene in EXTRA_SCENES:
        pool = sorted([r for r in records if r['scene'] == scene], key=lambda r: r['score'])
        if pool:
            previous_examples.extend(pool[i] for i in np.linspace(0, len(pool) - 1, min(6, len(pool)), dtype=int))
    sheet(previous_examples, output / 'previous_scenes.jpg', 'Previously discussed scenes: low to high structure scores')
    overview = [previews[d][i] for i in range(4) for d in ['reject', 'review', 'keep'] if i < len(previews[d])]
    sheet(overview, output / 'overview.jpg', 'Structure filter preview | REJECT / REVIEW / KEEP')
    benchmark_records = [r for r in records if r['uses']]
    by_image = {r['image']: r for r in records}
    task_counts = Counter()
    for task in tasks:
        decisions = [by_image[str((DATA_ROOT / task['scene'] / task[role]).resolve())]['decision'] for role in ['initial_view', 'target_view']]
        task_counts['reject' if 'reject' in decisions else 'review' if 'review' in decisions else 'keep'] += 1
    summary = {'images': len(records), 'benchmark_images': len(benchmark_records),
               'decisions': dict(Counter(r['decision'] for r in records)),
               'benchmark_image_decisions': dict(Counter(r['decision'] for r in benchmark_records)),
               'benchmark_task_decisions': dict(task_counts),
               'new_rejects_passing_legacy': sum(r['legacy_pass'] and r['decision'] == 'reject' for r in records)}
    report = {'benchmark': str(BENCHMARK_JSON), 'benchmark_sha256': hashlib.sha256(original).hexdigest(),
              'parameters': {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (str, int, float, list))},
              'summary': summary, 'previews': {k: [r['image'] for r in v] for k, v in previews.items()},
              'images': records}
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    readme = ['# Image structure filter preview\n\n',
              'Heuristic labels, not human annotations. Review means uncertain, not automatic rejection.\n\n',
              'Original benchmark and image files are unchanged. Scores measure image structure, not semantic uniqueness. Repeated textures may pass.\n\n',
              '[Overview](overview.jpg) | [Reject](reject.jpg) | [Review](review.jpg) | [Keep](keep.jpg) | [Previous scenes](previous_scenes.jpg) | [Detailed overlays](details/) | [Full metrics](report.json)\n\n',
              '| Decision | Images | Benchmark Images | Benchmark Tasks |\n|---|---:|---:|---:|\n']
    for decision in ['reject', 'review', 'keep']:
        readme.append(f"| {decision.title()} | {summary['decisions'].get(decision, 0)} | {summary['benchmark_image_decisions'].get(decision, 0)} | {task_counts[decision]} |\n")
    readme.append('\nA task is marked Reject if either endpoint is rejected; otherwise Review if either endpoint needs review.\n')
    readme.append('\nMethod: resize, exclude 3% borders, normalize exposure using the 2nd/98th intensity percentiles, then compute Canny edges and Shi-Tomasi corners. '
                  'A 4x4 cell is active when it has >=1 corner, >=1.5% edge pixels, and >=1.5 bits of local gray-level entropy. '
                  'The score combines cell coverage (35%), corners (30%), edge density (25%), and local entropy (10%), each normalized and capped. '
                  'Keep requires >=30 corners, >=6/16 active cells, and score >=45. Strongly parallel edges with few corners trigger Review. '
                  'Reject requires failure of the existing contrast/sharpness checks or extremely sparse edges/corners; other uncertain views need Review.\n')
    (output / 'index.md').write_text(''.join(readme))
    assert BENCHMARK_JSON.read_bytes() == original
    print(json.dumps(summary, indent=2))
    print(f'Preview: {output / "overview.jpg"}')
    print(f'Index: {output / "index.md"}')


def main():
    cv2.setNumThreads(1)
    original = BENCHMARK_JSON.read_bytes()
    tasks = json.loads(original)
    output = OUTPUT_ROOT / f'view_filter_preview_{datetime.now():%Y%m%d_%H%M%S}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'details').mkdir()
    records = collect(tasks)
    for record in tqdm(records, desc='Image structure', unit='view', mininterval=1):
        metrics, _ = analyze(record['image'])
        record.update(metrics)
    save_report(output, records, tasks, original)


if __name__ == '__main__':
    main()
