#!/bin/bash
# ergo 90-second judge demo (zero AWS cost, mock backends).
# Usage: bash demo.sh [video]
set -e
cd "$(dirname "$0")"
export PYTHONPATH=src ERGO_MOCK_BEDROCK=1 ERGO_MOCK_S3=1
VID="${1:-data/env2/video_env_2_anonymized/jv-scrivania-terra-001.mp4}"
echo "== 1/3 passive watch (YOLOv8-pose if models/ present, else mock) =="
python3 -m ergo.cli watch --source "$VID" --seconds 25 --subject-height-cm 196 --threshold 0.3
echo "== 2/3 trace (the Agentic Vision award evidence) =="
cat data/trace.jsonl | python3 -c "import sys,json; [print(json.loads(l).get('event'), '|', {k:v for k,v in json.loads(l).items() if k in ('score','decision','trunk_deg')}) for l in sys.stdin]"
echo "== 3/3 agent unit matrix (all three branches, no video needed) =="
PYTHONPATH=src python3 -c "
import numpy as np
from ergo.agent import make_agent
for s in (0.81, 0.55, 0.2):
    r = make_agent().run({'score': s, 'angles': {'trunk_flexion_deg': 68.0}}, [np.zeros((32,32,3),np.uint8)], '')
    print(f'  OpenCV {s} -> VLM {r[\"vlm\"][\"label\"]}@{r[\"vlm\"][\"confidence\"]} -> {r[\"decision\"]}')"
