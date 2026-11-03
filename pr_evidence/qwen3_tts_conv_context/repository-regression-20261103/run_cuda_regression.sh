#!/usr/bin/env bash
set -euo pipefail
export VLLM_OMNI_TARGET_DEVICE=cuda VLLM_TARGET_DEVICE=cuda PYTHONDONTWRITEBYTECODE=1
export UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=2 UV_NO_PROGRESS=1
SRC=/tmp/cuda-pr-source
OUT=/out/cuda-evidence
mkdir -p "$SRC" "$OUT" /out/wheel-cache
if [[ ! -d "$SRC/.git" ]]; then
    git -C "$SRC" init
    cp -a /source-cache/objects/. "$SRC/.git/objects/"
    if [[ -f /source-cache/shallow ]]; then cp /source-cache/shallow "$SRC/.git/shallow"; fi
    git -C "$SRC" checkout --detach bff6f65ac32fba6739b8327d7caa41996bf7d1b6
fi
test "$(git -C "$SRC" rev-parse HEAD)" = bff6f65ac32fba6739b8327d7caa41996bf7d1b6
python3 /kit/download_cuda_wheel.py /out/wheel-cache
uv pip install --system --torch-backend=cu130 /out/wheel-cache/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl
uv pip install --system --torch-backend=cu130 -e "$SRC"
python3 - "$SRC" "$OUT" <<'PY'
import pathlib,sys,tomllib
src,out=map(pathlib.Path,sys.argv[1:])
dev=tomllib.loads((src/'pyproject.toml').read_text())['project']['optional-dependencies']['dev']
names={'pytest','pytest-asyncio','pytest-mock','pytest-xdist'}
reqs=[r for r in dev if r.split('==')[0] in names]
assert len(reqs)==len(names)
(out/'test-tools.txt').write_text('\n'.join(reqs)+'\n')
PY
uv pip install --system --torch-backend=cu130 -r "$OUT/test-tools.txt"
env -u UV_OVERRIDE uv pip install --system 'nvidia-nccl-cu13==2.29.7'
python3 - <<'PY' > "$OUT/nccl-distributions.json"
import importlib.metadata as m,json,sys
rows=[{'name':d.metadata['Name'],'version':d.version,'metadata_path':str(d._path)}
      for d in m.distributions() if d.metadata['Name'].lower().replace('_','-')=='nvidia-nccl-cu13']
print(json.dumps({'python':sys.executable,'distributions':rows},indent=2))
PY
python3 -m pip check > "$OUT/pip-check.log"
python3 -m pip freeze --all > "$OUT/pip-freeze.log"
git -C "$SRC" show --format=fuller --stat HEAD > "$OUT/git-provenance.log"
python3 /kit/run_cuda_checks.py --source "$SRC" --result-dir "$OUT"
