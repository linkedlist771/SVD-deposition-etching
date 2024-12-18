import argparse
from pathlib import Path
from typing import List, Dict
from loguru import logger
from collections import defaultdict

from eval.metrics.metrics_calculator import MetricsCalculator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate metrics between real and generated datasets"
    )

    parser.add_argument(
        "--real-dir",
        type=str,
        required=True,
        help="Path to directory containing real images",
    )

    parser.add_argument(
        "--gen-dir",
        type=str,
        required=True,
        help="Path to directory containing generated images",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["CLIP_SCORE", "LPIPS", "PSNR", "FID"],
        help="List of metrics to calculate. Available options: CLIP_SCORE, LPIPS, PSNR, FID",
    )

    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for FID calculation"
    )

    parser.add_argument(
        "--dims", type=int, default=2048, help="Feature dimensions for FID calculation"
    )
    parser.add_argument(
        "--include_subdir",
        action="store_true",
        help="Whether to process subdirectories"
    )
    return parser.parse_args()


def get_subdirs(path: Path) -> List[Path]:
    """Get all immediate subdirectories of the given path"""
    return [x for x in path.iterdir() if x.is_dir()]


def calculate_metrics_for_dirs(real_dir: Path, gen_dir: Path, args) -> Dict:
    calculator = MetricsCalculator(
        real_dataset_dir_path=real_dir,
        generated_dataset_dir_path=gen_dir,
        metrics=args.metrics,
        batch_size=args.batch_size,
        dims=args.dims,
    )
    return calculator.calculate_all()


def main():
    args = parse_args()

    logger.info(f"Calculating metrics between:")
    logger.info(f"Real images dir: {args.real_dir}")
    logger.info(f"Generated images dir: {args.gen_dir}")
    logger.info(f"Selected metrics: {args.metrics}")
    logger.info(f"Include subdirectories: {args.include_subdir}")

    if not args.include_subdir:
        # Original behavior
        results = calculate_metrics_for_dirs(Path(args.real_dir), Path(args.gen_dir), args)
        logger.info("Results:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value}")
    else:
        # Process subdirectories
        real_subdirs = get_subdirs(Path(args.real_dir))
        gen_subdirs = get_subdirs(Path(args.gen_dir))

        # Get common subdir names
        real_subdir_names = {d.name for d in real_subdirs}
        gen_subdir_names = {d.name for d in gen_subdirs}
        common_subdir_names = real_subdir_names.intersection(gen_subdir_names)

        if not common_subdir_names:
            logger.warning("No matching subdirectories found between real and generated directories")
            return

        # Store results for each metric across all subdirs
        all_results = defaultdict(list)

        for subdir_name in common_subdir_names:
            real_subdir = Path(args.real_dir) / subdir_name
            gen_subdir = Path(args.gen_dir) / subdir_name

            logger.info(f"\nProcessing subdirectory: {subdir_name}")
            results = calculate_metrics_for_dirs(real_subdir, gen_subdir, args)

            # Store results
            for metric, value in results.items():
                all_results[metric].append(value)

        # Calculate and display averages
        logger.info("\nAverage Results across all subdirectories:")
        for metric, values in all_results.items():
            avg_value = sum(values) / len(values)
            logger.info(f"{metric}: {avg_value}")


if __name__ == "__main__":
    main()
