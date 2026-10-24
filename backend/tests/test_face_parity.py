"""The browser (faceFrame.js) and the trainer (face_features.py) must compute identical primitives."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tilt.face_features import PRIMITIVES, frame_primitives
from test_face_research import neutral_face

JS = Path(__file__).resolve().parents[2] / "frontend" / "src" / "sensors" / "faceFrame.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_and_python_primitives_match(tmp_path):
    rng = np.random.default_rng(1)
    faces = [neutral_face(dx=rng.normal(0, .02), dy=rng.normal(0, .02), gaze=rng.normal(0, .01),
                          eye=rng.uniform(.1, 1.2), scale=rng.uniform(.6, 1.4)) for _ in range(20)]
    faces = [f + rng.normal(0, .002, f.shape) for f in faces]
    (tmp_path / "faces.json").write_text(json.dumps([[{"x": p[0], "y": p[1]} for p in f] for f in faces]))
    script = (f"import {{ framePrimitives }} from {json.dumps(JS.as_uri())};\n"
              "import fs from 'fs';\n"
              f"const faces = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / 'faces.json'))}));\n"
              "console.log(JSON.stringify(faces.map(framePrimitives)));\n")
    (tmp_path / "run.mjs").write_text(script)
    out = json.loads(subprocess.check_output(["node", str(tmp_path / "run.mjs")], text=True))
    for face, js in zip(faces, out):
        py = frame_primitives(face)
        assert np.allclose([py[k] for k in PRIMITIVES], js, rtol=1e-12, atol=1e-12)
