from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
import math
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from Eva.utils import load_from_checkpoint
from utils.wsi_inference import LevelInferenceResult, run_level_inference

DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "Eva_ft.ckpt"
DEFAULT_BIOMARKERS_PATH = REPO_ROOT / "examples" / "biomarkers.npy"
DEFAULT_WHITE_THRESHOLD = 200.0 / 255.0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the WSI inference CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run Eva virtual-staining inference for a whole-slide image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--slide-path", required=True, help="Path to the input slide, typically an .svs file.")
    parser.add_argument("--output-dir", required=True, help="Directory where level outputs are written.")
    parser.add_argument(
        "--checkpoint-path",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to the Eva checkpoint to load.",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the Eva YAML config file.",
    )
    parser.add_argument(
        "--biomarkers-path",
        default=str(DEFAULT_BIOMARKERS_PATH),
        help="Path to the NumPy biomarker name list used for output channels.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[0],
        help="Slide levels to process.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=224,
        help="Tile size in level pixels.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=224,
        help="Stride between tile origins in level pixels.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of non-background tiles per inference batch.",
    )
    parser.add_argument(
        "--white-threshold",
        type=float,
        default=DEFAULT_WHITE_THRESHOLD,
        help="Tiles are skipped when at least 99.9%% of pixels have all RGB channels above this threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, for example 'cpu' or 'cuda:0'. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--ome-dtype",
        choices=["uint16", "float32"],
        default=None,
        help="Optional OME-TIFF dtype override. Defaults to uint16 for quantized modes and float32 for --ome-quant-mode none.",
    )
    parser.add_argument(
        "--ome-quant-mode",
        choices=["global", "tile", "none"],
        default="global",
        help="OME-TIFF export mode: whole-image quantization, per-tile quantization, or raw float32 export with no quantization.",
    )
    parser.add_argument(
        "--quant-min",
        type=float,
        default=None,
        help="Optional fixed quantization minimum for the OME-TIFF export.",
    )
    parser.add_argument(
        "--quant-max",
        type=float,
        default=None,
        help="Optional fixed quantization maximum for the OME-TIFF export.",
    )
    return parser


def load_biomarkers(path: str | Path) -> list[str]:
    """Load biomarker names from a NumPy array file."""

    biomarkers = np.load(Path(path), allow_pickle=False)
    biomarker_array = np.asarray(biomarkers)

    if biomarker_array.ndim != 1:
        raise ValueError("biomarker list must be a 1D NumPy array of strings")

    biomarker_names = biomarker_array.tolist()

    if not biomarker_names:
        raise ValueError("biomarker list must not be empty")

    if any(not isinstance(name, str) for name in biomarker_names):
        raise ValueError("biomarker list must contain only strings")

    return biomarker_names


def main(argv: Sequence[str] | None = None) -> int:
    """Run Eva WSI inference from the command line."""

    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    _validate_args(args)

    config = OmegaConf.load(args.config_path)
    _validate_model_geometry(args, config)
    biomarkers = load_biomarkers(args.biomarkers_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_from_checkpoint(args.checkpoint_path, config, device=device)
    quant_min, quant_max = _resolve_quantization_args(args)
    ome_dtype = _resolve_ome_dtype(args)

    slide = _open_slide(args.slide_path)
    try:
        patch_size = _load_patch_size(config)
        for level in args.levels:
            result = run_level_inference(
                slide=slide,
                level=level,
                biomarkers=biomarkers,
                model=model,
                output_dir=args.output_dir,
                tile_size=args.tile_size,
                stride=args.stride,
                white_threshold=args.white_threshold,
                batch_size=args.batch_size,
                patch_size=patch_size,
                device=device,
                quant_min=quant_min,
                quant_max=quant_max,
                ome_quant_mode=args.ome_quant_mode,
                ome_dtype=ome_dtype,
            )
            _print_level_summary(result)
    finally:
        close = getattr(slide, "close", None)
        if callable(close):
            close()

    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if any(level < 0 for level in args.levels):
        raise ValueError("--levels must contain only non-negative integers")
    if args.stride < args.tile_size:
        raise ValueError("--stride must be greater than or equal to --tile-size")
    if args.ome_quant_mode in {"tile", "none"}:
        if args.quant_min is not None or args.quant_max is not None:
            warnings.warn(
                f"In {args.ome_quant_mode} mode, --quant-min and --quant-max are ignored.",
                UserWarning,
                stacklevel=2,
            )
    if args.ome_dtype is not None:
        if args.ome_quant_mode == "none" and args.ome_dtype != "float32":
            raise ValueError("--ome-dtype must be 'float32' when --ome-quant-mode is 'none'")
        if args.ome_quant_mode in {"global", "tile"} and args.ome_dtype != "uint16":
            raise ValueError("--ome-dtype must be 'uint16' for quantized OME-TIFF export")
    if args.ome_quant_mode in {"tile", "none"}:
        return
    if (args.quant_min is None) != (args.quant_max is None):
        raise ValueError("--quant-min and --quant-max must be provided together")
    if args.quant_min is not None:
        if not math.isfinite(args.quant_min) or not math.isfinite(args.quant_max):
            raise ValueError("--quant-min and --quant-max must be finite")
        if args.quant_max <= args.quant_min:
            raise ValueError("--quant-max must be greater than --quant-min")


def _open_slide(path: str | Path) -> object:
    try:
        import openslide
    except ImportError as exc:
        raise ImportError("openslide-python is required to run WSI inference") from exc

    return openslide.OpenSlide(str(path))


def _load_patch_size(config: object) -> int:
    try:
        patch_size = config.ds.token_size
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("config must define ds.token_size as a positive integer") from exc

    if not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size <= 0:
        raise ValueError("config must define ds.token_size as a positive integer")

    return patch_size


def _load_model_tile_size(config: object) -> int:
    try:
        tile_size = config.ds.patch_size
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("config must define ds.patch_size as a positive integer") from exc

    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
        raise ValueError("config must define ds.patch_size as a positive integer")

    return tile_size


def _resolve_quantization_args(args: argparse.Namespace) -> tuple[float | None, float | None]:
    if args.ome_quant_mode in {"tile", "none"}:
        return None, None

    return args.quant_min, args.quant_max


def _resolve_ome_dtype(args: argparse.Namespace) -> str | None:
    if args.ome_dtype is not None:
        return args.ome_dtype
    if args.ome_quant_mode == "none":
        return "float32"
    return "uint16"


def _validate_model_geometry(args: argparse.Namespace, config: object) -> None:
    expected_tile_size = _load_model_tile_size(config)
    if args.tile_size != expected_tile_size:
        raise ValueError(
            f"--tile-size must match config ds.patch_size ({expected_tile_size}) for the loaded Eva checkpoint"
        )


def _print_level_summary(result: LevelInferenceResult) -> None:
    print(
        "Finished level "
        f"{result.level}: predicted {result.predicted_tiles}/{result.total_tiles} tiles, "
        f"skipped {result.skipped_tiles}, raw={result.raw_npy_path}, ome={result.ome_tiff_path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
