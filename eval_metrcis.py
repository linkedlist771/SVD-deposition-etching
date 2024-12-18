import argparse
from pathlib import Path
from typing import List
from loguru import logger

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

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Calculating metrics between:")
    logger.info(f"Real images dir: {args.real_dir}")
    logger.info(f"Generated images dir: {args.gen_dir}")
    logger.info(f"Selected metrics: {args.metrics}")

    calculator = MetricsCalculator(
        real_dataset_dir_path=Path(args.real_dir),
        generated_dataset_dir_path=Path(args.gen_dir),
        metrics=args.metrics,
        batch_size=args.batch_size,
        dims=args.dims,
    )

    results = calculator.calculate_all()

    logger.info("Results:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value}")


if __name__ == "__main__":
    main()
