"""Phase 1 v3 wide — wider outage mask (539 rows, 8 사이트 + 2h neighbor + 8-16).

비교용 — 옵션 B (278) vs wide (539) 차이 측정.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/models"))

# 기존 outage-masked 스크립트 import + path/mask만 변경
import train_resmlp_adaln_v3_outage_masked as base
base.OUT_DIR  = ROOT / "pv/experiments/resmlp_adaln_v3_outage_masked_wide"
base.OUT_DIR.mkdir(parents=True, exist_ok=True)
base.MASK_PATH = ROOT / "data/processed/phase1_outage_mask_wide.parquet"

if __name__ == "__main__":
    base.main()
