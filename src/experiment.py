import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

def create_run_dir(base_dir: str, config: Dict[str, Any]) -> Path:
    """
    Crea results/runs/<run_id>/ con:
      - config.json
      - artifacts/
      - metrics/
      - figures/
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha1(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    run_id = f"{ts}_{h}"

    run_dir = base / run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return run_dir

def save_metrics(run_dir: Path, metrics: Dict[str, Any], rows: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    Guarda metrics.json y opcional metrics.csv.
    """
    with open(run_dir / "metrics" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if rows is not None:
        pd.DataFrame(rows).to_csv(run_dir / "metrics" / "metrics.csv", index=False)

def save_artifact_csv(run_dir: Path, name: str, df: pd.DataFrame) -> Path:
    """
    Guarda un df en artifacts/<name>.csv y devuelve el path.
    """
    out = run_dir / "artifacts" / name
    df.to_csv(out, index=False)
    return out
