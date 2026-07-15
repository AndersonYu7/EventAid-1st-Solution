#!/bin/bash
# ============================================================================
# One-command verification of the EventAid_Final release (team yunyu8).
#
# Quick mode (default, ~3 min, needs 1 GPU):
#   bash EventAid_Final/verify.sh
#     [1] all release .py files compile
#     [2] ensemble composition: 30 members, 11/19/18/20 per skip   (paper Tab. 1)
#     [3] RefinerV2 (shipped config feat48/base96) = 4.86 M params   (paper §2.7)
#     [4] all shipped checkpoints load                             (models/)
#     [5] zero-init graft is bit-identical at step 0, GPU          (paper §2.2)
#     [6] OFFICIAL evaluator on the final submission
#         -> expects PSNR 43.1056 / SSIM 0.980260                  (paper §4)
#
# Full mode (adds ~5 min GPU work):
#   bash EventAid_Final/verify.sh --full
#     [7] re-infer the ERF-specialist member (ball, 7skip, 8-way TTA) from the
#         shipped checkpoint and compare against the stored member directory
#         -> expects 28/28 frames bit-identical                    (paper §2.5)
#
# Run from anywhere; paths are resolved relative to this script's location.
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../EventAid_Final
ROOT="$(dirname "$HERE")"                              # project root
cd "$ROOT"
PASS=0; FAIL=0
ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "=== EventAid_Final release verification ==="
echo "root: $ROOT"
echo ""

# ---------------------------------------------------------------- [1] compile
echo "[1] compiling every release .py ..."
CFAIL=0
for f in $(find "$HERE/code" -name "*.py"); do
  python3 -m py_compile "$f" 2>/dev/null || { echo "    compile error: $f"; CFAIL=1; }
done
[ $CFAIL -eq 0 ] && ok "all $(find "$HERE/code" -name '*.py' | wc -l) scripts compile" || bad "compile errors"

# ------------------------------------------------- [2] ensemble composition
echo "[2] ensemble composition from code/fusion/recipe_v16.json (final recipe) ..."
python3 - "$HERE" <<'PY' && ok "30 members, per-skip 11/19/18/20 (= paper Tab. 1)" || bad "ensemble composition mismatch"
import json, sys
r = json.load(open(sys.argv[1] + "/code/fusion/recipe_v16.json"))
per, allm = {}, set()
for skip in ["1skip", "3skip", "7skip", "15skip"]:
    pool = set()
    for bk in r[skip]:
        for k in ("weights_y", "weights_c", "weights"):
            pool |= set(bk.get(k, {}))
    per[skip] = len(pool); allm |= pool
print(f"    per-skip {per}, union {len(allm)}")
assert len(allm) == 30 and [per[s] for s in ["1skip","3skip","7skip","15skip"]] == [11,19,18,20]
PY

# ---------------------------------------------------- [3] RefinerV2 params
echo "[3] RefinerV2 parameter count (shipped config feat=48, base=96) ..."
python3 - "$HERE" <<'PY' && ok "RefinerV2 = 4.86 M params, matches shipped checkpoint (= paper §2.7)" || bad "param count mismatch"
import sys, torch; sys.path.insert(0, sys.argv[1] + "/code/arch")
from refiner_v2arch import RefinerV2
n = sum(p.numel() for p in RefinerV2(feat=48, base=96).parameters())
sd = torch.load(sys.argv[1] + "/models/fusion_refiner_v2arch_s2500.pth", map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
m = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
print(f"    arch(feat48,base96): {n:,} = {n/1e6:.2f}M ; shipped ckpt: {m:,}")
assert abs(n/1e6 - 4.86) < 0.01 and n == m
PY

# -------------------------------------------------- [4] checkpoints loadable
echo "[4] loading every shipped checkpoint in models/ ..."
python3 - "$HERE" <<'PY' && ok "all checkpoints load" || bad "checkpoint load failure"
import glob, sys, torch
files = sorted(glob.glob(sys.argv[1] + "/models/*.p*"))
for f in files:
    torch.load(f, map_location="cpu", weights_only=False)
print(f"    loaded {len(files)}/{len(files)}")
PY

# ------------------------------------------- [5] zero-init graft (GPU, ~40s)
echo "[5] zero-init graft bit-identical at step 0 (GPU) ..."
python3 - "$ROOT" "$HERE" <<'PY' && ok "max |diff| = 0.000 exactly (= paper §2.2)" || bad "zero-init check failed"
import sys, torch
root, here = sys.argv[1], sys.argv[2]
sys.path.insert(0, root); sys.path.insert(0, here + "/code/arch")
sys.path.insert(0, root + "/models/EMA-VFI")
from emae_arch import build_emae
net = build_emae(root + "/weights/emavfi/ours.pkl").eval()
torch.manual_seed(0)
imgs = torch.rand(1, 6, 256, 448, device="cuda")
vox = torch.randn(1, 16, 256, 448, device="cuda") * 5
with torch.no_grad():
    _, _, _, a = net(torch.cat([imgs, vox], 1))                    # with events
    _, _, _, b = net(torch.cat([imgs, torch.zeros_like(vox)], 1))  # without
d = (a - b).abs().max().item()
print(f"    max |pred(events) - pred(none)| = {d:.3e}")
assert d == 0.0
PY

# --------------------------------------- [6] OFFICIAL evaluator on the final
echo "[6] official evaluator on the final submission (expect 43.1056 / 0.980260) ..."
RES="$ROOT/submission_final/results"
if [ ! -d "$RES" ]; then
  echo "    staging dir missing -> extracting submission_final.zip (one-time) ..."
  mkdir -p "$ROOT/submission_final"
  python3 -c "import zipfile; zipfile.ZipFile('$ROOT/submission_final.zip').extractall('$ROOT/submission_final')"
fi
OUT=$(python3 "$ROOT/challenge_data/evaluate_results.py" \
        --data-dir "$ROOT/challenge_data" --results-dir "$RES" 2>&1)
echo "$OUT" | sed 's/^/    /'
echo "$OUT" | grep -q "psnr=43.1056 ssim=0.980260" \
  && ok "official score 43.1056 / 0.980260 (= paper headline)" \
  || bad "official score does not match"

# --------------------------------------------------------- [7] full mode only
if [ "${1:-}" = "--full" ]; then
  echo "[7] re-inferring ERF-specialist member (ball, 7skip, 8-way TTA) from the shipped checkpoint ..."
  TMP="$ROOT/_verify_member"; rm -rf "$TMP"; mkdir -p "$TMP"
  cp "$HERE/code/inference/run_emae.py" "$TMP/"    # scripts resolve ROOT as parent-of-parent
  EMAE_DIHEDRAL=1 python3 "$TMP/run_emae.py" \
      --ckpt "$HERE/models/erf_specialist_7skip_gaps8-16_step12000.pkl" \
      --splits test --skips 7skip --seqs ball --tta \
      --anchor-dir "$ROOT/submission_v16/results" \
      --output-dir "$TMP/out" > "$TMP/run.log" 2>&1
  python3 - "$ROOT" "$TMP" <<'PY' && ok "28/28 frames bit-identical to the stored member (= paper §2.5)" || bad "member reproduction mismatch"
import glob, os, sys
import numpy as np
from PIL import Image
root, tmp = sys.argv[1], sys.argv[2]
new = sorted(glob.glob(tmp + "/out/7skip/ball/*.png"))
ident = 0
for p in new:
    q = f"{root}/results_test_v1_8way/7skip/ball/{os.path.basename(p)}"
    if (np.asarray(Image.open(p)) == np.asarray(Image.open(q))).all():
        ident += 1
print(f"    bit-identical frames: {ident}/{len(new)}")
assert new and ident == len(new)
PY
  rm -rf "$TMP"
fi

echo ""
echo "=== RESULT: $PASS passed, $FAIL failed ==="
[ $FAIL -eq 0 ] && echo "ALL CHECKS PASSED — release reproduces the paper numbers." || echo "SOME CHECKS FAILED — see above."
exit $FAIL
