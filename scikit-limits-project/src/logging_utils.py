from __future__ import annotations

import json
import platform
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =========================================================
# HELPERS GÉNÉRAUX
# =========================================================

def utc_now_iso() -> str:
    """
    Retourne un timestamp UTC ISO 8601.
    """
    return datetime.now(timezone.utc).isoformat()


def ensure_directory(path: str | Path) -> Path:
    """
    Crée le dossier s'il n'existe pas.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_id(prefix: str = "run") -> str:
    """
    Génère un identifiant de run.
    """
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _make_json_serializable(obj: Any) -> Any:
    """
    Convertit récursivement un objet en structure JSON-sérialisable.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_make_json_serializable(v) for v in obj]

    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return {
                k: _make_json_serializable(v)
                for k, v in vars(obj).items()
            }
        except Exception:
            pass

    return str(obj)


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """
    Sauvegarde un dictionnaire en JSON.
    """
    path = Path(path)
    ensure_directory(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_make_json_serializable(data), f, indent=2, ensure_ascii=False)

    return path


def append_jsonl(record: dict[str, Any], path: str | Path) -> Path:
    """
    Ajoute un enregistrement JSON sur une ligne.
    """
    path = Path(path)
    ensure_directory(path.parent)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_make_json_serializable(record), ensure_ascii=False) + "\n")

    return path


# =========================================================
# ENVIRONNEMENT
# =========================================================

def get_environment_info() -> dict[str, Any]:
    """
    Retourne les informations d'environnement.
    """
    env = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    try:
        import sklearn
        env["sklearn_version"] = sklearn.__version__
    except Exception:
        env["sklearn_version"] = None

    try:
        import numpy
        env["numpy_version"] = numpy.__version__
    except Exception:
        env["numpy_version"] = None

    try:
        import pandas
        env["pandas_version"] = pandas.__version__
    except Exception:
        env["pandas_version"] = None

    return env


# =========================================================
# STRUCTURES DE LOG
# =========================================================

@dataclass
class RunContext:
    run_id: str
    benchmark_mode: str | None = None
    seed: int | None = None
    started_at: str = field(default_factory=utc_now_iso)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioContext:
    scenario_id: int | None = None
    scenario_name: str | None = None
    scenario_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelContext:
    model_name: str | None = None
    model_family: str | None = None
    preprocessor_name: str | None = None
    fold: int | None = None


@dataclass
class LogRecord:
    timestamp: str
    level: str
    event_type: str
    message: str
    run_context: dict[str, Any] = field(default_factory=dict)
    scenario_context: dict[str, Any] = field(default_factory=dict)
    model_context: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


# =========================================================
# LOGGER SIMPLE
# =========================================================

class BenchmarkLogger:
    """
    Logger simple pour le benchmark.
    """

    def __init__(
        self,
        log_dir: str | Path,
        run_id: str | None = None,
        benchmark_mode: str | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.log_dir = ensure_directory(log_dir)
        self.run_id = run_id or make_run_id()
        self.verbose = verbose

        self.run_context = RunContext(
            run_id=self.run_id,
            benchmark_mode=benchmark_mode,
            seed=seed,
        )

        self.jsonl_path = self.log_dir / f"{self.run_id}.jsonl"
        self.run_summary_path = self.log_dir / f"{self.run_id}_summary.json"
        self.environment_path = self.log_dir / f"{self.run_id}_environment.json"

        self._records_count = 0
        self._warnings_count = 0
        self._errors_count = 0

    # -------------------------
    # bas niveau
    # -------------------------
    def _emit(
        self,
        level: str,
        event_type: str,
        message: str,
        scenario_context: ScenarioContext | None = None,
        model_context: ModelContext | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = LogRecord(
            timestamp=utc_now_iso(),
            level=level,
            event_type=event_type,
            message=message,
            run_context=asdict(self.run_context),
            scenario_context=asdict(scenario_context) if scenario_context else {},
            model_context=asdict(model_context) if model_context else {},
            payload=payload or {},
        )

        record_dict = asdict(record)
        append_jsonl(record_dict, self.jsonl_path)

        self._records_count += 1
        if level.lower() == "warning":
            self._warnings_count += 1
        if level.lower() == "error":
            self._errors_count += 1

        if self.verbose:
            print(f"[{level.upper()}] {event_type} - {message}")

        return record_dict

    # -------------------------
    # niveaux standard
    # -------------------------
    def info(
        self,
        event_type: str,
        message: str,
        scenario_context: ScenarioContext | None = None,
        model_context: ModelContext | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._emit(
            level="info",
            event_type=event_type,
            message=message,
            scenario_context=scenario_context,
            model_context=model_context,
            payload=payload,
        )

    def warning(
        self,
        event_type: str,
        message: str,
        scenario_context: ScenarioContext | None = None,
        model_context: ModelContext | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._emit(
            level="warning",
            event_type=event_type,
            message=message,
            scenario_context=scenario_context,
            model_context=model_context,
            payload=payload,
        )

    def error(
        self,
        event_type: str,
        message: str,
        scenario_context: ScenarioContext | None = None,
        model_context: ModelContext | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._emit(
            level="error",
            event_type=event_type,
            message=message,
            scenario_context=scenario_context,
            model_context=model_context,
            payload=payload,
        )

    # -------------------------
    # méthodes utilitaires
    # -------------------------
    def log_run_start(
        self,
        config: dict[str, Any] | None = None,
        environment_info: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "config": config or {},
            "environment_info": environment_info or get_environment_info(),
        }

        self.info(
            event_type="run_start",
            message="Début du benchmark.",
            payload=payload,
        )

    def log_run_end(
        self,
        summary: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "summary": summary or {},
            "records_count": self._records_count,
            "warnings_count": self._warnings_count,
            "errors_count": self._errors_count,
            "finished_at": utc_now_iso(),
        }

        self.info(
            event_type="run_end",
            message="Fin du benchmark.",
            payload=payload,
        )

        save_json(
            {
                "run_context": asdict(self.run_context),
                "records_count": self._records_count,
                "warnings_count": self._warnings_count,
                "errors_count": self._errors_count,
                "summary": summary or {},
            },
            self.run_summary_path,
        )

    def log_scenario_start(
        self,
        scenario_id: int,
        scenario_name: str,
        scenario_config: dict[str, Any] | None = None,
    ) -> None:
        self.info(
            event_type="scenario_start",
            message=f"Début du scénario {scenario_name}.",
            scenario_context=ScenarioContext(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                scenario_config=scenario_config or {},
            ),
        )

    def log_scenario_end(
        self,
        scenario_id: int,
        scenario_name: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.info(
            event_type="scenario_end",
            message=f"Fin du scénario {scenario_name}.",
            scenario_context=ScenarioContext(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
            ),
            payload=payload,
        )

    def log_model_start(
        self,
        model_name: str,
        model_family: str | None = None,
        preprocessor_name: str | None = None,
        fold: int | None = None,
        scenario_id: int | None = None,
        scenario_name: str | None = None,
    ) -> None:
        self.info(
            event_type="model_start",
            message=f"Début évaluation modèle {model_name}.",
            scenario_context=ScenarioContext(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
            ),
            model_context=ModelContext(
                model_name=model_name,
                model_family=model_family,
                preprocessor_name=preprocessor_name,
                fold=fold,
            ),
        )

    def log_model_end(
        self,
        model_name: str,
        model_family: str | None = None,
        preprocessor_name: str | None = None,
        fold: int | None = None,
        scenario_id: int | None = None,
        scenario_name: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.info(
            event_type="model_end",
            message=f"Fin évaluation modèle {model_name}.",
            scenario_context=ScenarioContext(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
            ),
            model_context=ModelContext(
                model_name=model_name,
                model_family=model_family,
                preprocessor_name=preprocessor_name,
                fold=fold,
            ),
            payload={"metrics": metrics or {}},
        )

    def log_warning_list(
        self,
        warnings_list: list[str],
        model_name: str | None = None,
        preprocessor_name: str | None = None,
        fold: int | None = None,
        scenario_id: int | None = None,
        scenario_name: str | None = None,
    ) -> None:
        for warning_message in warnings_list:
            self.warning(
                event_type="warning_captured",
                message=warning_message,
                scenario_context=ScenarioContext(
                    scenario_id=scenario_id,
                    scenario_name=scenario_name,
                ),
                model_context=ModelContext(
                    model_name=model_name,
                    preprocessor_name=preprocessor_name,
                    fold=fold,
                ),
            )

    def log_exception(
        self,
        exc: Exception,
        event_type: str = "exception",
        message: str | None = None,
        scenario_id: int | None = None,
        scenario_name: str | None = None,
        model_name: str | None = None,
        model_family: str | None = None,
        preprocessor_name: str | None = None,
        fold: int | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if extra_payload:
            payload.update(extra_payload)

        self.error(
            event_type=event_type,
            message=message or str(exc),
            scenario_context=ScenarioContext(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
            ),
            model_context=ModelContext(
                model_name=model_name,
                model_family=model_family,
                preprocessor_name=preprocessor_name,
                fold=fold,
            ),
            payload=payload,
        )

    def save_environment_snapshot(self) -> None:
        save_json(get_environment_info(), self.environment_path)


# =========================================================
# HELPERS RECORDS
# =========================================================

def build_metric_log_payload(
    metrics: dict[str, Any] | None = None,
    times: dict[str, Any] | None = None,
    matrix_info: dict[str, Any] | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """
    Construit un payload standard pour logguer les résultats d'un fold / modèle.
    """
    return {
        "status": status,
        "metrics": metrics or {},
        "times": times or {},
        "matrix_info": matrix_info or {},
    }


def build_scenario_payload(
    dataset_shape: tuple[int, int] | None = None,
    target_positive_rate: float | None = None,
    missing_global_rate: float | None = None,
) -> dict[str, Any]:
    """
    Payload standard pour décrire un scénario / dataset.
    """
    return {
        "dataset_shape": dataset_shape,
        "target_positive_rate": target_positive_rate,
        "missing_global_rate": missing_global_rate,
    }


if __name__ == "__main__":
    logger = BenchmarkLogger(
        log_dir="data/outputs/logs",
        benchmark_mode="debug",
        seed=42,
        verbose=True,
    )

    logger.save_environment_snapshot()
    logger.log_run_start(config={"example": True})

    logger.log_scenario_start(
        scenario_id=1,
        scenario_name="baseline",
        scenario_config={"n_samples": 1000, "noise": 1.0},
    )

    logger.log_model_start(
        model_name="logistic_regression",
        model_family="linear",
        preprocessor_name="basic_tfidf_robust",
        fold=1,
        scenario_id=1,
        scenario_name="baseline",
    )

    logger.log_warning_list(
        warnings_list=["ConvergenceWarning: maximum iterations reached"],
        model_name="logistic_regression",
        preprocessor_name="basic_tfidf_robust",
        fold=1,
        scenario_id=1,
        scenario_name="baseline",
    )

    logger.log_model_end(
        model_name="logistic_regression",
        model_family="linear",
        preprocessor_name="basic_tfidf_robust",
        fold=1,
        scenario_id=1,
        scenario_name="baseline",
        metrics={"f1": 0.72, "roc_auc": 0.81},
    )

    try:
        raise ValueError("Exemple d'erreur")
    except Exception as e:
        logger.log_exception(
            e,
            event_type="model_error",
            message="Erreur pendant l'évaluation du modèle.",
            scenario_id=1,
            scenario_name="baseline",
            model_name="svc_rbf",
            model_family="kernel",
            preprocessor_name="sparse_tfidf_maxabs",
            fold=2,
        )

    logger.log_scenario_end(
        scenario_id=1,
        scenario_name="baseline",
        payload=build_scenario_payload(
            dataset_shape=(1000, 42),
            target_positive_rate=0.30,
            missing_global_rate=0.05,
        ),
    )

    logger.log_run_end(summary={"best_model": "logistic_regression"})