#!/usr/bin/env python3
"""
ABVT3R — Master Pipeline Runner
=================================
Orchestrates all subsystems for batch processing.

Usage
-----
    # New 30-degree dataset — run full Procedure Alpha on all specimens
    python run_pipeline.py alpha --all

    # New dataset — specific specimen
    python run_pipeline.py alpha --specimen DG041_20260609_B02

    # Legacy 4-view data
    python run_pipeline.py alpha --legacy --plants 1 2 3 5

    # Biomass dashboard (reads procedure_alpha/outputs/)
    python run_pipeline.py dashboard

    # Cross-method evaluation
    python run_pipeline.py evaluate --export results/comparison.pdf

    # Simulate a 12-view capture session (no hardware)
    python run_pipeline.py capture DG042_20260610_B02 --simulate

    # Launch dataset viewer
    python run_pipeline.py viewer

Legacy integers still work for backward compatibility:
    python run_pipeline.py 1            # process legacy plant 1
    python run_pipeline.py 1 3 5        # process legacy plants 1, 3, 5
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand handlers
# ─────────────────────────────────────────────────────────────────────────────

def cmd_alpha(args):
    from procedure_alpha.pipeline import ProcedureAlpha
    pa = ProcedureAlpha()
    if args.legacy:
        plants = args.plants if args.plants else list(range(1, 41))
        for pid in plants:
            try:
                pa.run_legacy(int(pid))
            except Exception as exc:
                print(f"[Run] plant_{pid} failed: {exc}")
    elif args.specimen:
        pa.run_specimen(args.specimen, cam_label=args.cam)
    elif args.all:
        from shared.config import SPECIMENS_DIR
        for spec_dir in sorted(SPECIMENS_DIR.iterdir()):
            if spec_dir.is_dir():
                try:
                    pa.run_specimen(spec_dir.name, cam_label=args.cam)
                except Exception as exc:
                    print(f"[Run] {spec_dir.name} failed: {exc}")
    else:
        print("Specify --specimen ID, --all, or --legacy [--plants N ...]")


def cmd_dashboard(_args):
    from biomass_engine.visualisation.results_dashboard import main
    main()


def cmd_evaluate(args):
    from evaluation_suite.comparison import run_comparison
    out = Path(args.export) if args.export else None
    run_comparison(out)


def cmd_capture(args):
    from acquisition.capture.session_manager import CaptureSession
    from shared.config import LEGACY_ANGLES_DEG, CAPTURE_ANGLES_DEG
    angles  = LEGACY_ANGLES_DEG if args.legacy else CAPTURE_ANGLES_DEG
    session = CaptureSession(
        specimen_id  = args.specimen_id,
        serial_port  = args.port,
        settle_delay = args.settle,
        simulate     = args.simulate,
        trigger      = args.trigger,
        record_gt    = args.gt,
        angles_deg   = angles,
    )
    ok = session.run()
    sys.exit(0 if ok else 1)


def cmd_viewer(_args):
    from acquisition.dataset.dataset_viewer import main
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Legacy integer-arg fallback  (python run_pipeline.py 1 3 5)
# ─────────────────────────────────────────────────────────────────────────────

def _legacy_int_mode(plant_ids: list[int]):
    """Run Procedure Alpha on a list of legacy plant IDs."""
    from procedure_alpha.pipeline import ProcedureAlpha
    pa = ProcedureAlpha()
    for pid in plant_ids:
        try:
            pa.run_legacy(pid)
        except Exception as exc:
            print(f"[Legacy] plant_{pid} failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="ABVT3R Master Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = root.add_subparsers(dest="command")

    # alpha ───────────────────────────────────────────────────────────────────
    p_alpha = sub.add_parser("alpha", help="Procedure Alpha 3D reconstruction")
    grp = p_alpha.add_mutually_exclusive_group()
    grp.add_argument("--specimen", metavar="ID",
                     help="New-protocol specimen ID (e.g. DG041_20260609_B02)")
    grp.add_argument("--all",    action="store_true",
                     help="Process all specimens in acquisition/dataset/specimens/")
    grp.add_argument("--legacy", action="store_true",
                     help="Use legacy data_collection/ directory")
    p_alpha.add_argument("--plants", nargs="+", type=int,
                          help="Plant IDs when using --legacy (default: 1-40)")
    p_alpha.add_argument("--cam", default="A", choices=["A", "B"],
                          help="Camera label for depth (new protocol, default A)")

    # dashboard ───────────────────────────────────────────────────────────────
    sub.add_parser("dashboard", help="Biomass prediction results dashboard")

    # evaluate ────────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Cross-method evaluation report")
    p_eval.add_argument("--export", metavar="FILE",
                         help="Save comparison figure to PNG or PDF")

    # capture ─────────────────────────────────────────────────────────────────
    p_cap = sub.add_parser("capture", help="Run a 12-view (or 4-view) capture session")
    p_cap.add_argument("specimen_id",
                        help="Specimen ID, e.g. DG042_20260610_B02")
    p_cap.add_argument("--port",     default=None,
                        help="Arduino serial port (e.g. /dev/ttyUSB0)")
    p_cap.add_argument("--settle",   default=2.0, type=float,
                        help="Settle delay after turntable move (seconds, default 2.0)")
    p_cap.add_argument("--simulate", action="store_true",
                        help="Simulate cameras — no hardware required")
    p_cap.add_argument("--trigger", action="store_true",
                        help="Pause at each angle and wait for Enter before capturing")
    p_cap.add_argument("--gt", action="store_true",
                        help="Prompt for ground-truth measurements after capture")
    p_cap.add_argument("--legacy",   action="store_true",
                        help="Use 4-view 90° protocol instead of 12-view 30°")

    # viewer ──────────────────────────────────────────────────────────────────
    sub.add_parser("viewer", help="Interactive dataset viewer")

    return root


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Detect legacy integer-only invocation (e.g. python run_pipeline.py 1 3 5)
    if len(sys.argv) > 1 and sys.argv[1].lstrip("-").isdigit():
        try:
            plant_ids = [int(a) for a in sys.argv[1:]]
            _legacy_int_mode(plant_ids)
            return
        except ValueError:
            pass

    parser  = build_parser()
    args    = parser.parse_args()

    dispatch = {
        "alpha":     cmd_alpha,
        "dashboard": cmd_dashboard,
        "evaluate":  cmd_evaluate,
        "capture":   cmd_capture,
        "viewer":    cmd_viewer,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
