"""beattrace - command-line helper for the interactive-beat diagnosis.

The graphical entry point is the panel (``scripts/run_panel.py`` /
``python3 -m interbeat.panel``). This CLI deliberately only assists it:

    # build a controlled slow-drift input and immediately diagnose it
    beattrace construct --kind drift --out out/drift.json
    beattrace diagnose out/drift.json

    # build a phase-jump input and compare the attribution
    beattrace construct --kind jump --beats 0.5 --out out/jump.json
    beattrace diagnose out/jump.json

    # build a gradual-tempo input (multi-step one-way tempo change)
    beattrace construct --kind ramp --tempo 120 --tempo-end 168 --out out/ramp.json
    beattrace diagnose out/ramp.json

    # replay an exported panel run offline (no original process)
    beattrace diagnose out/panel-evidence.json --json

``construct`` and ``diagnose`` never read live session state: every input is
a standalone evidence document, so a reproduction can be replayed anywhere.
Missing measurements are reported, never substituted with defaults.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, ".pylocal"), ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from interbeat.diagnostics import (concise, diagnose, diagnose_file,  # noqa: E402
                                   load_maps, run_scenario)
from interbeat.trace import write_evidence  # noqa: E402


def cmd_construct(args) -> int:
    jump_at = None
    if args.kind == "jump":
        jump_at = args.jump_at if args.jump_at is not None else args.seconds / 2.0
    doc = run_scenario(
        seconds=args.seconds, sr=args.sr, tempo=args.tempo,
        drift_ppm=args.drift_ppm if args.kind == "drift" else 0.0,
        jump_beats=args.beats if args.kind == "jump" else 0.0,
        jump_at=jump_at,
        tempo_end=args.tempo_end if args.kind == "ramp" else None,
        tempo_steps=args.tempo_steps if args.kind == "ramp" else 0,
        seed=args.seed, jitter=args.jitter, loss=args.loss,
        name=args.kind)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        write_evidence(args.out, doc["meta"], doc["nodes"])
        print(f"证据已写入 {args.out}（节点 {', '.join(doc['nodes'])}）")
    else:
        json.dump(doc, sys.stdout, ensure_ascii=False, indent=1)
        sys.stdout.write("\n")
    return 0


def cmd_diagnose(args) -> int:
    result = diagnose_file(args.path, args.maps)
    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=1)
        sys.stdout.write("\n")
    else:
        print(concise(result))
    # Non-zero exit when anomalies were found: the tool's verdict must be
    # usable in a script, and incomplete evidence must not look like a pass.
    return 0 if result["ok"] else 2


def cmd_maps(args) -> int:
    maps = load_maps(args.maps)
    print(f"映射来源: {maps.source}")
    print("阈值:")
    for group, values in maps.thresholds.items():
        for key, value in values.items():
            print(f"  {group}.{key} = {value}")
    print("原因 → 阶段:")
    for key, info in maps.causes.items():
        print(f"  {key}: {info['label']} → {info['stage']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="beattrace",
        description="互动节拍诊断辅助：构造漂移/突跳证据并离线归因。"
                    "图形入口见 scripts/run_panel.py。")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("construct", help="构造受控输入并导出可重放证据")
    c.add_argument("--kind", choices=("drift", "jump", "ramp"), default="drift")
    c.add_argument("--drift-ppm", type=float, default=-200.0,
                   help="follower 时钟漂移（ppm，负为慢）")
    c.add_argument("--beats", type=float, default=0.5,
                   help="相位突跳拍数")
    c.add_argument("--jump-at", type=float, default=None,
                   help="突跳发生的相对秒数，默认时长一半")
    c.add_argument("--seconds", type=float, default=12.0)
    c.add_argument("--sr", type=int, default=4000)
    c.add_argument("--tempo", type=float, default=120.0)
    c.add_argument("--tempo-end", type=float, default=160.0,
                   help="ramp 终速 BPM")
    c.add_argument("--tempo-steps", type=int, default=8,
                   help="ramp 分段数（>1 才是渐变）")
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--jitter", type=float, default=0.0)
    c.add_argument("--loss", type=float, default=0.0)
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_construct)

    d = sub.add_parser("diagnose", help="离线诊断证据文档")
    d.add_argument("path")
    d.add_argument("--maps", default=None,
                   help="诊断映射（文件或目录），默认 interbeat/diag_maps")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_diagnose)

    m = sub.add_parser("maps", help="打印当前诊断映射（无默认值回退）")
    m.add_argument("--maps", default=None)
    m.set_defaults(func=cmd_maps)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
