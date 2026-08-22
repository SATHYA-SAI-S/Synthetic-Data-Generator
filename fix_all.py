import os
import json
import re

# 1. requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    req = f.read()
req = req.replace('torch\n', 'torch==2.3.1\n')
with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(req)

# 2. kaggle_runner/kernel-metadata.json
with open('kaggle_runner/kernel-metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)
meta['accelerator'] = 'nvidiaTeslaT4'
with open('kaggle_runner/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

# 3. kaggle_runner/run_pipeline.py
with open('kaggle_runner/run_pipeline.py', 'r', encoding='utf-8') as f:
    run_pipe = f.read()
if 'torch.zeros(1).cuda()' not in run_pipe:
    run_pipe = run_pipe.replace(
        "assert torch.cuda.is_available(), 'CUDA not available'; ",
        "assert torch.cuda.is_available(), 'CUDA not available'; _ = torch.zeros(1).cuda(); "
    )
    with open('kaggle_runner/run_pipeline.py', 'w', encoding='utf-8') as f:
        f.write(run_pipe)

print("Applied 1-3.")
