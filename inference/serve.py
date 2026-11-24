import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)

TASKS = {
    "inpainting": {
        "file": "inpainting_cube_restored.npy",
        "missed_axis": 0,
        "shape": [580, 951, 462],
        "missing_boxes": [
            [[None, None], [368, 581], [179, 282]],
        ],
    },
    "outpainting": {
        "file": "outpainting_cube_restored.npy",
        "missed_axis": 1,
        "shape": [460, 481, 800],
        "missing_boxes": [
            [[None, None], [0, 24], [None, None]],
            [[None, None], [457, 481], [None, None]],
        ],
    },
}

INPUT_SUB = "input_seismic_data"
INPUT_DIR = os.path.join(PROJECT, "input_seismic_data")
INPUT_TASKS = {
    "inpainting": "inpainting_cube_missed_data.npy",
    "outpainting": "outpainting_cube_missed_data.npy",
}

_lock = threading.Lock()
_cube = None
_cube_key = None


def subdirs():
    out = os.path.join(PROJECT, "output")
    if not os.path.isdir(out):
        return []
    return sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)))


def manifest():
    cubes = []
    for sub in subdirs():
        for task, cfg in TASKS.items():
            if os.path.isfile(os.path.join(PROJECT, "output", sub, cfg["file"])):
                cubes.append({
                    "sub": sub,
                    "task": task,
                    "shape": cfg["shape"],
                    "missed_axis": cfg["missed_axis"],
                    "missing_boxes": cfg["missing_boxes"],
                })
    for task, fname in INPUT_TASKS.items():
        if os.path.isfile(os.path.join(INPUT_DIR, fname)):
            cubes.append({
                "sub": INPUT_SUB,
                "task": task,
                "label": "исходный",
                "shape": TASKS[task]["shape"],
                "missed_axis": TASKS[task]["missed_axis"],
                "missing_boxes": TASKS[task]["missing_boxes"],
            })
    return {"cubes": cubes}


def clamped(box, shape):
    return [
        (0 if s is None else max(0, min(s, n)), n if e is None else min(e, n))
        for (s, e), n in zip(box, shape)
    ]


def cube_path(sub, task):
    if sub == INPUT_SUB:
        return os.path.join(INPUT_DIR, INPUT_TASKS[task])
    return os.path.join(PROJECT, "output", sub, TASKS[task]["file"])


def load(sub, task):
    global _cube, _cube_key
    key = (sub, task)
    with _lock:
        if _cube_key == key:
            return _cube
        cfg = TASKS[task]
        t0 = time.perf_counter()
        cube = np.load(cube_path(sub, task))
        if list(cube.shape) != cfg["shape"]:
            print(f"[serve] {sub}/{task}: shape {cube.shape} != expected {cfg['shape']}, boxes clamped")
        mask = np.zeros(cube.shape, dtype=bool)
        for box in cfg["missing_boxes"]:
            mask[tuple(slice(s, e) for s, e in clamped(box, cube.shape))] = True
        known = cube[~mask]
        if known.size:
            vmin, vmax = (float(v) for v in np.percentile(known, [1, 99]))
        else:
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        _cube_key, _cube = key, (cube, vmin, vmax)
        print(f"[serve] {sub}/{task}: {cube.shape} loaded in {time.perf_counter() - t0:.1f}s "
              f"(vmin={vmin:.4g}, vmax={vmax:.4g})")
        return _cube


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, ctype, body):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, "text/html; charset=utf-8", f.read())
        elif url.path == "/manifest":
            self._send(200, "application/json; charset=utf-8", json.dumps(manifest()).encode())
        elif url.path == "/slice":
            self.handle_slice(parse_qs(url.query))
        else:
            self.send_error(404)

    def handle_slice(self, q):
        try:
            sub, task = q["sub"][0], q["task"][0]
            axis, i = int(q["axis"][0]), int(q["i"][0])
        except (IndexError, KeyError, ValueError):
            self.send_error(400, "expected ?sub=&task=&axis=&i=")
            return
        if task not in TASKS or sub not in subdirs() + [INPUT_SUB] or not os.path.isfile(cube_path(sub, task)):
            self.send_error(404, f"cube {sub}/{task} not found")
            return
        cube, vmin, vmax = load(sub, task)
        if not (0 <= axis < cube.ndim and 0 <= i < cube.shape[axis]):
            self.send_error(400, "index out of range")
            return
        idx = [slice(None)] * cube.ndim
        idx[axis] = i
        plane = cube[tuple(idx)]
        q8 = np.clip((plane - vmin) * (255.0 / (vmax - vmin)), 0, 255).round().astype(np.uint8)
        self._send(200, "application/octet-stream", q8.ravel().tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if not subdirs():
        print("[serve] output/ absent or empty — no cubes yet")
    print(f"[serve] http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
