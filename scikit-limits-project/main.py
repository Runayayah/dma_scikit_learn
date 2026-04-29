from __future__ import annotations

import traceback

from src.config import (
    BENCHMARK_MODE,
    FIGURES_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    ensure_directories,
    get_mode_summary,
)
from src.experiments import run_full_benchmark
from src.logging_utils import BenchmarkLogger
from src.reporting import generate_report_from_dataframes
from src.scalability import run_full_scalability_benchmark


def main() -> None:
    ensure_directories()

    logger = BenchmarkLogger(
        log_dir=LOGS_DIR,
        benchmark_mode=BENCHMARK_MODE,
        verbose=True,
    )

    logger.save_environment_snapshot()
    logger.log_run_start(config=get_mode_summary())

    try:
        print("\n=== Lancement du benchmark principal ===")
        benchmark_results = run_full_benchmark()

        raw_results_df = benchmark_results["raw_results"]
        aggregated_results_df = benchmark_results["global_aggregated_results"]
        leaderboard_df = benchmark_results["leaderboard"]

        logger.info(
            event_type="benchmark_completed",
            message="Benchmark principal terminé.",
            payload={
                "n_raw_rows": len(raw_results_df),
                "n_aggregated_rows": len(aggregated_results_df),
                "n_leaderboard_rows": len(leaderboard_df),
            },
        )

        print("\n=== Génération du reporting ===")
        report = generate_report_from_dataframes(
            raw_results_df=raw_results_df,
            aggregated_df=aggregated_results_df,
            reports_dir=REPORTS_DIR,
            figures_dir=FIGURES_DIR,
            sort_metric="f1_mean",
        )

        logger.info(
            event_type="reporting_completed",
            message="Reporting terminé.",
            payload=report["summary"],
        )

        # Scalabilité optionnelle
        # Tu peux commenter ce bloc si le run est trop long.
        print("\n=== Lancement du benchmark de scalabilité ===")
        scalability_results = run_full_scalability_benchmark()

        combined_summary = scalability_results.get("combined_summary")
        logger.info(
            event_type="scalability_completed",
            message="Benchmark de scalabilité terminé.",
            payload={
                "n_rows": 0 if combined_summary is None else len(combined_summary),
            },
        )

        logger.log_run_end(
            summary={
                "best_model": report["summary"].get("best_model"),
                "best_family": report["summary"].get("best_family"),
                "best_preprocessor": report["summary"].get("best_preprocessor"),
                "best_score": report["summary"].get("best_score"),
            }
        )

        print("\n=== Projet terminé avec succès ===")
        print(f"Reports : {REPORTS_DIR}")
        print(f"Figures : {FIGURES_DIR}")
        print(f"Logs    : {LOGS_DIR}")

    except Exception as exc:
        logger.log_exception(
            exc,
            event_type="main_error",
            message="Erreur dans le lancement principal du projet.",
        )
        logger.log_run_end(summary={"status": "error", "message": str(exc)})

        print("\n=== Erreur pendant l'exécution ===")
        print(str(exc))
        print("\nTraceback complet :")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()