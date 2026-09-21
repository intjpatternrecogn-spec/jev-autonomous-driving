"""Command-line demo for synthetic driving scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controller import DrivingDecisionEngine
from .types import DrivingState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.scenario.read_text(encoding="utf-8"))
    result = DrivingDecisionEngine().decide(
        DrivingState(**payload["state"]), payload["question"], payload["candidates"]
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
