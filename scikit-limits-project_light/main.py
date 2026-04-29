from __future__ import annotations

from src.advanced_analysis import run_advanced_analysis
from src.config import FIGURES_DIR, REPORTS_DIR, ensure_directories
from src.experiments import run_experiments
from src.reporting import generate_report
from src.visualization import generate_figures


def main() -> None:
    ensure_directories()

    print("\n=== Étape 1 : benchmark principal ===")
    results = run_experiments()

    print("\n=== Étape 2 : analyse avancée sklearn ===")
    advanced_results = run_advanced_analysis()

    print("\n=== Étape 3 : visualisations ===")
    generate_figures(
        leaderboard=results["leaderboard"],
        aggregated_results=results["aggregated_results"],
        output_dir=FIGURES_DIR,
    )

    print("\n=== Étape 4 : reporting ===")
    generate_report(
        leaderboard=results["leaderboard"],
        aggregated_results=results["aggregated_results"],
        reports_dir=REPORTS_DIR,
        advanced_results=advanced_results,
    )

    print("\n=== Projet terminé ===")
    print(f"Rapports : {REPORTS_DIR}")
    print(f"Figures  : {FIGURES_DIR}")


if __name__ == "__main__":
    main()