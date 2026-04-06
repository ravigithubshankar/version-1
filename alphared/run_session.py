import os
import threading
import time
import requests
import queue
import uuid
import torch
from .utils.system_telemetry import get_system_metrics
from .utils.variable_observer import VariableObserver
from .diagnostics_engine import DiagnosticsEngine
from .ratchet_loop import RatchetLoop, AgentBudget

import sys


class OutputInterceptor:
    """Intercepts stdout/stderr and mirrors it to the SDK log queue."""
    def __init__(self, original_stream, callback):
        self.original_stream = original_stream
        self.callback = callback

    def write(self, text):
        self.original_stream.write(text)
        self.original_stream.flush()
        if text.strip():
            self.callback(text.strip())

    def flush(self):
        self.original_stream.flush()


class RunSession:
    def __init__(self, project, name=None, api_key=None, host="http://localhost:8000"):
        self.project  = project
        self.name     = name or f"run-{uuid.uuid4().hex[:6]}"
        self.api_key  = api_key
        self.host     = host.rstrip("/")
        self.run_id   = None
        self.running  = False
        self._queue   = queue.Queue()
        self._thread  = None
        self._telemetry_thread = None
        self._observer = None
        self._session  = requests.Session()

        # DiagnosticsEngine -- real-time training problem detection
        self._diagnostics   = DiagnosticsEngine()
        self._epoch_counter = 0

        # Training metadata
        self._model_type = None
        self._model_path = None
        self._dataset_id = None

        # Auto-tracking
        self._tracked_model    = None
        self._tracked_datasets = []

        # Agent (RatchetLoop) -- activated via enable_agent()
        self._agent_enabled        = False
        self._openrouter_api_key   = ""
        self._train_file_path      = ""
        self._agent_budget         = AgentBudget()
        self._agent_train_fn       = None
        self._ratchet_thread       = None

        # Capture stdout/stderr to mirror terminal logs to dashboard
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = OutputInterceptor(self._original_stdout, lambda msg: self.log_text(msg, "TRAINING"))
        sys.stderr = OutputInterceptor(self._original_stderr, lambda msg: self.log_text(msg, "ERROR"))

    # ------------------------------------------------------------------ #
    #  Agent control                                                       #
    # ------------------------------------------------------------------ #

    def enable_agent(
        self,
        openrouter_api_key:  str   = "",
        train_file_path:     str   = "",
        max_experiments:     int   = 50,
        time_per_experiment: float = 300.0,
        train_fn=None,
    ):
        """
        Enables the autonomous RatchetLoop agent (AutoResearch-style).
        Call this before finish() to activate overnight auto-improvement.

        The agent will:
          1. Create a git branch autoresearch/<run_id>
          2. Use OpenRouter LLM to suggest code changes
          3. Apply changes to train_file_path
          4. Run training, check if improved
          5. Commit if improved, git reset if not (ratchet mechanic)
          6. Repeat until max_experiments or manually stopped
          7. Write changes.md + results.tsv for review

        Args:
            openrouter_api_key:  OpenRouter API key (or set OPENROUTER_API_KEY env var)
            train_file_path:     Path to training script the agent edits (like AutoResearch train.py)
            max_experiments:     Max experiments per session (default 50, ~4h at 5min each)
            time_per_experiment: Seconds per experiment (default 300 = 5 min, same as AutoResearch)
            train_fn:            Optional custom train function
                                 signature: fn(config: ExperimentConfig) -> dict
                                 If None, re-runs train_file_path as subprocess
        """
        self._agent_enabled      = True
        self._openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._train_file_path    = train_file_path
        self._agent_train_fn     = train_fn
        self._agent_budget       = AgentBudget(
            max_experiments     = max_experiments,
            time_per_experiment = time_per_experiment,
        )
        print(f"[AlphaRed] Agent enabled")
        print(f"[AlphaRed]   LLM:         {'OpenRouter' if self._openrouter_api_key else 'static fallback (set OPENROUTER_API_KEY)'}")
        print(f"[AlphaRed]   Experiments: {max_experiments} max")
        print(f"[AlphaRed]   Time/exp:    {time_per_experiment/60:.0f} min")
        print(f"[AlphaRed]   Train file:  {train_file_path or '(not set -- provide train_fn)'}")

    # ------------------------------------------------------------------ #
    #  Core SDK                                                            #
    # ------------------------------------------------------------------ #

    def start(self):
        """Initializes the run on the backend and starts background threads."""
        try:
            response = self._session.post(
                f"{self.host}/api/sdk/init",
                json={"project": self.project, "name": self.name, "api_key": self.api_key},
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                self.run_id = data.get("run_id")
                self.running = True
                self._create_initial_artifact()
                self._thread = threading.Thread(target=self._worker, daemon=True)
                self._thread.start()
                self._telemetry_thread = threading.Thread(target=self._telemetry_worker, daemon=True)
                self._telemetry_thread.start()
                print(f"[AlphaRed] Run initialized: {self.run_id}")
                print(f"[AlphaRed] View dashboard at: {self.host}/runs/{self.run_id}")
            else:
                print(f"[AlphaRed] Failed to initialize: {response.status_code} {response.text}")
        except Exception as e:
            print(f"[AlphaRed] Error connecting to backend: {e}")

    def _create_initial_artifact(self):
        try:
            self._session.post(
                f"{self.host}/api/artifacts",
                json={
                    "name": f"run-{self.run_id}", "type": "run", "run_id": self.run_id,
                    "path": f"runs/{self.run_id}", "size": 0,
                    "metadata": {
                        "project": self.project, "run_name": self.name,
                        "status": "running", "created_by": "sdk", "created_at": time.time(),
                    },
                },
                timeout=5,
            )
            print(f"[AlphaRed] Initial artifact created")
        except Exception as e:
            print(f"[AlphaRed] Warning: Could not create initial artifact: {e}")

    def log(self, metrics, step=None):
        """
        Logs metrics to dashboard.
        Standardizes names, runs DiagnosticsEngine every 5 epochs.
        """
        if not self.running:
            return

        clean_metrics = {}
        for k, v in metrics.items():
            key = k.lower()
            if key in ["acc", "accuracy", "train_acc", "train_accuracy"]:
                clean_metrics["train_accuracy"] = v
            elif key in ["loss", "train_loss"]:
                clean_metrics["train_loss"] = v
            elif key in ["val_acc", "val_accuracy"]:
                clean_metrics["val_accuracy"] = v
            elif key in ["val_loss"]:
                clean_metrics["val_loss"] = v
            elif key in ["lr", "learning_rate"]:
                clean_metrics["lr"] = v
            else:
                clean_metrics[k] = v

        found_step  = metrics.get("step") or metrics.get("batch") or metrics.get("iteration")
        found_epoch = metrics.get("epoch")

        payload = {
            "type": "metrics", "data": clean_metrics,
            "timestamp": time.time(),
            "step":  step if step is not None else found_step,
            "epoch": found_epoch,
        }

        # DiagnosticsEngine: update every call, analyze every 5 epochs
        self._diagnostics.update(clean_metrics)
        self._epoch_counter += 1

        if self._epoch_counter % 5 == 0:
            diagnosis = self._diagnostics.analyze(epoch=self._epoch_counter)
            if diagnosis.is_problem():
                self._queue.put({
                    "type": "diagnosis",
                    "data": diagnosis.to_dict(),
                    "timestamp": time.time(),
                })
                icon = "🔴" if diagnosis.severity == "critical" else "🟡"
                print(f"\n{icon} [AlphaRed Diagnosis] {diagnosis.message}")
                print(f"   Suggestion: {diagnosis.suggestion}\n")

        self._queue.put(payload)

    def push(self, scope=None, model=None, interval=1.0):
        """
        Activates real-time variable monitoring.
        Auto-detects model and datasets from scope.
        """
        if not self.running:
            self.start()

        if model is not None:
            self._tracked_model = model
            self._detect_model_type(model)
            print(f"[AlphaRed] Model tracked: {type(model).__name__}")

        if scope is not None:
            self._observer = VariableObserver(scope, callback=self.log, interval=interval)
            self._observer.start()
            print(f"[AlphaRed] Monitoring variables: {list(self._observer.watchlist)}")

            if model is None:
                try:
                    import torch.nn as nn
                    for var_name, var_value in scope.items():
                        if isinstance(var_value, nn.Module):
                            self._tracked_model = var_value
                            self._detect_model_type(var_value)
                            print(f"[AlphaRed] Auto-detected PyTorch model: {var_name}")
                            break
                except ImportError:
                    pass

                if self._tracked_model is None:
                    for var_name, var_value in scope.items():
                        if self._is_model(var_value):
                            self._tracked_model = var_value
                            self._detect_model_type(var_value)
                            print(f"[AlphaRed] Auto-detected model: {var_name}")
                            break

            self._auto_detect_datasets(scope)

    def finish(self, run_tests=True):
        """
        Stops training session, saves model, triggers testing,
        then starts RatchetLoop agent if enable_agent() was called.
        """
        self.running = False
        if self._observer:
            self._observer.stop()
        if self._thread:
            self._thread.join(timeout=2)

        # Auto-save model
        if self._tracked_model is not None:
            print("\n[AlphaRed] Auto-saving tracked model...")
            try:
                self.save_model(self._tracked_model)
                time.sleep(0.5)
            except Exception as e:
                print(f"[AlphaRed] Warning: Could not auto-save model: {e}")

        # Dataset registry + artifacts
        if self._tracked_datasets:
            print(f"\n[AlphaRed] Registering {len(self._tracked_datasets)} dataset(s)...")
            self._register_datasets_to_registry()
            self._create_dataset_artifacts()

        # Automated testing
        if self.run_id and run_tests:
            self._trigger_automated_testing()

        # DiagnosticsEngine summary -> backend
        summary = self._diagnostics.summary()
        try:
            self._session.post(
                f"{self.host}/api/sdk/runs/{self.run_id}/diagnosis_summary",
                json=summary, timeout=5,
            )
            print(f"\n[AlphaRed] Training Summary:")
            print(f"   Epochs:         {summary['total_epochs']}")
            print(f"   Final Loss:     {summary['final_train_loss']}")
            print(f"   Final Val Loss: {summary['final_val_loss']}")
            print(f"   Final Val Acc:  {summary['final_val_acc']}")
            print(f"   Last Diagnosis: {summary['last_diagnosis']}")
        except Exception:
            pass

        # Start RatchetLoop agent if enabled
        if self._agent_enabled and self.run_id:
            self._start_ratchet_loop()

        # Restore stdout/stderr
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

    # ------------------------------------------------------------------ #
    #  RatchetLoop integration                                             #
    # ------------------------------------------------------------------ #

    def _start_ratchet_loop(self):
        """
        Launches RatchetLoop in background thread after training completes.
        Baseline metrics come from DiagnosticsEngine training summary.
        """
        summary = self._diagnostics.summary()

        # Build baseline from training summary
        baseline_metrics = {}
        if summary.get("final_val_acc") is not None:
            baseline_metrics["val_accuracy"] = summary["final_val_acc"]
        if summary.get("final_val_loss") is not None:
            baseline_metrics["val_loss"] = summary["final_val_loss"]

        if not baseline_metrics:
            print("[AlphaRed] Agent: No val metrics found in training summary.")
            print("[AlphaRed] Tip: Log val_accuracy or val_loss in your training loop.")
            return

        output_dir = f"./alphared_agent/{self.run_id}"
        os.makedirs(output_dir, exist_ok=True)

        def default_train_fn(config):
            """
            Default train_fn: re-runs train_file_path as subprocess.
            Parses val_accuracy / val_loss from stdout.
            Used when user does not provide a custom train_fn.
            """
            import subprocess

            if not config.train_file_path or not os.path.exists(config.train_file_path):
                raise RuntimeError(f"train_file_path not found: {config.train_file_path}")

            result = subprocess.run(
                ["python", config.train_file_path],
                capture_output=True, text=True,
                timeout=config.time_budget_sec * 2,
            )

            # Parse metrics from combined stdout+stderr
            metrics = {}
            output  = result.stdout + result.stderr
            for line in output.split("\n"):
                for key in ["val_accuracy", "val_loss", "accuracy", "loss", "vram_mb"]:
                    if key + ":" in line.lower():
                        try:
                            val = float(line.lower().split(key + ":")[1].strip().split()[0])
                            metrics[key] = val
                        except Exception:
                            pass

            if not metrics:
                raise RuntimeError(
                    f"Could not parse metrics.\n"
                    f"Make sure your script prints e.g. 'val_accuracy: 0.85'\n"
                    f"Last 300 chars of output:\n{output[-300:]}"
                )
            return metrics

        train_fn = self._agent_train_fn or default_train_fn

        loop = RatchetLoop(
            run_id             = self.run_id,
            train_fn           = train_fn,
            train_file_path    = self._train_file_path,
            model_type         = self._model_type or "deep_learning",
            budget             = self._agent_budget,
            output_dir         = output_dir,
            repo_dir           = "./",
            openrouter_api_key = self._openrouter_api_key,
            git_enabled        = os.path.exists(".git"),
            status_callback    = lambda msg: self.log_text(msg, "AGENT"),
        )

        print(f"\n{'='*60}")
        print(f"[AlphaRed] Autonomous Agent Loop Starting")
        print(f"   Run ID:      {self.run_id}")
        print(f"   Baseline:    {baseline_metrics}")
        print(f"   Budget:      {self._agent_budget.max_experiments} experiments")
        print(f"   Time/exp:    {self._agent_budget.time_per_experiment/60:.0f} min")
        print(f"   LLM:         {'OpenRouter' if self._openrouter_api_key else 'static fallback'}")
        print(f"   Output:      {output_dir}/changes.md")
        print(f"   Results:     {output_dir}/results.tsv")
        print(f"   Dashboard:   {self.host}/runs/{self.run_id}")
        print(f"{'='*60}\n")

        self._ratchet_thread = loop.run_async(baseline_metrics)

    # ------------------------------------------------------------------ #
    #  Testing                                                             #
    # ------------------------------------------------------------------ #

    def _trigger_automated_testing(self):
        try:
            dataset_info = {"dataset_id": self._dataset_id, "datasets": []}
            for ds in self._tracked_datasets:
                dataset_info["datasets"].append({
                    "name": ds.get("name"), "type": ds.get("type"),
                    "purpose": ds.get("purpose"), "size": ds.get("size"),
                    "dataset_type": ds.get("dataset_type"),
                })
            response = self._session.post(
                f"{self.host}/api/runs/{self.run_id}/complete",
                json={
                    "model_type":   self._model_type or "traditional_ml",
                    "model_path":   self._model_path,
                    "dataset_id":   self._dataset_id,
                    "dataset_info": dataset_info,
                },
                timeout=30,
            )
            if response.status_code == 200:
                print(f"\n[AlphaRed] Automated Testing Started")
                print(f"   View at: {self.host}/runs/{self.run_id}")
            else:
                print(f"[AlphaRed] Warning: Testing trigger failed: {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"[AlphaRed] Testing running in background -- check dashboard")
        except Exception as e:
            print(f"[AlphaRed] Warning: Could not trigger testing: {e}")

    # ------------------------------------------------------------------ #
    #  Model helpers                                                       #
    # ------------------------------------------------------------------ #

    def _is_model(self, obj):
        if obj is None:
            return False
        module     = type(obj).__module__
        class_name = type(obj).__name__
        exclude    = ["Compose", "Transform", "DataLoader", "Dataset", "Optimizer", "Scheduler"]
        if any(p in class_name for p in exclude):
            return False
        try:
            import torch.nn as nn
            if isinstance(obj, nn.Module):
                return True
        except ImportError:
            pass
        ml_modules  = ["sklearn", "xgboost", "lightgbm", "tensorflow", "keras"]
        ml_patterns = ["Model", "Classifier", "Regressor", "Estimator", "Network", "Net"]
        if any(m in module for m in ml_modules):
            if "transform" not in module.lower() and "utils" not in module.lower():
                return True
        if any(p in class_name for p in ml_patterns):
            return True
        if hasattr(obj, "fit") and hasattr(obj, "predict"):
            return True
        return False

    def _detect_model_type(self, model):
        module = type(model).__module__
        try:
            import torch.nn as nn
            if isinstance(model, nn.Module):
                self._model_type = "deep_learning"
                return
        except ImportError:
            pass
        if any(m in module for m in ["sklearn", "xgboost", "lightgbm"]):
            self._model_type = "traditional_ml"
        elif any(m in module for m in ["torch", "tensorflow", "keras"]):
            self._model_type = "deep_learning"
        else:
            self._model_type = "traditional_ml"

    def save_model(self, model, path=None):
        import pickle, joblib, os
        if not self.run_id:
            print("[AlphaRed] Warning: Run not started. Call start() first.")
            return None
        self._detect_model_type(model)
        models_dir  = "backend/app/apis/models"
        os.makedirs(models_dir, exist_ok=True)
        module      = type(model).__module__
        model_class = type(model).__name__
        is_pytorch  = False
        try:
            import torch.nn as nn
            is_pytorch = isinstance(model, nn.Module)
        except ImportError:
            pass
        if path is None:
            if is_pytorch or "torch" in module:
                path = f"{models_dir}/{self.run_id}.pth"
            elif "tensorflow" in module or "keras" in module:
                path = f"{models_dir}/{self.run_id}.h5"
            else:
                path = f"{models_dir}/{self.run_id}.joblib"
        try:
            if is_pytorch or "torch" in module:
                import torch
                try:
                    import dill
                    with open(path, "wb") as f:
                        dill.dump(model, f)
                    torch.save(
                        {"state_dict": model.state_dict(), "model_class": model_class},
                        path.replace(".pth", "_state_dict.pth"),
                    )
                    print(f"[AlphaRed] PyTorch model saved (dill): {path}")
                except ImportError:
                    torch.save(model, path)
                    print(f"[AlphaRed] PyTorch model saved: {path}")
            elif "tensorflow" in module or "keras" in module:
                model.save(path)
                print(f"[AlphaRed] TF/Keras model saved: {path}")
            else:
                joblib.dump(model, path)
                print(f"[AlphaRed] Model saved: {path}")
        except Exception as e:
            print(f"[AlphaRed] Error saving model: {e}")
            if "torch" not in module:
                try:
                    with open(path, "wb") as f:
                        pickle.dump(model, f)
                    print(f"[AlphaRed] Model saved (pickle fallback): {path}")
                except Exception:
                    return None
            else:
                return None
        self._model_path = path
        file_size = os.path.getsize(path) if os.path.exists(path) else 0
        try:
            self._session.post(
                f"{self.host}/api/artifacts",
                json={
                    "name": f"model-{self.run_id}", "type": "model",
                    "run_id": self.run_id, "path": path, "size": file_size,
                    "metadata": {
                        "model_type": self._model_type, "model_class": model_class,
                        "model_module": module,
                        "file_format": os.path.splitext(path)[1], "created_by": "sdk",
                    },
                },
                timeout=5,
            )
        except Exception:
            pass
        try:
            self._session.post(
                f"{self.host}/api/sdk/runs/{self.run_id}/metadata",
                json={"model_path": path, "model_type": self._model_type},
                timeout=5,
            )
        except Exception:
            pass
        return path

    def set_dataset(self, dataset_id):
        self._dataset_id = dataset_id
        print(f"[AlphaRed] Dataset registered: {dataset_id}")
        import os
        upload_dir   = "backend/app/apis/uploads"
        val_patterns = [
            f"{upload_dir}/ds-{dataset_id}_val.csv",
            f"{upload_dir}/{dataset_id}_val.csv",
            f"{upload_dir}/ds-{dataset_id}_val.parquet",
            f"{upload_dir}/{dataset_id}_val.parquet",
        ]
        dataset_path = next((p for p in val_patterns if os.path.exists(p)), None)
        if dataset_path:
            try:
                self._session.post(
                    f"{self.host}/api/artifacts",
                    json={
                        "name": f"dataset-{dataset_id}", "type": "dataset",
                        "run_id": self.run_id, "path": dataset_path,
                        "size": os.path.getsize(dataset_path),
                        "metadata": {"dataset_id": dataset_id, "purpose": "validation", "created_by": "sdk"},
                    },
                    timeout=5,
                )
            except Exception:
                pass
        if self.run_id:
            try:
                self._session.post(
                    f"{self.host}/api/sdk/runs/{self.run_id}/metadata",
                    json={"dataset_id": dataset_id}, timeout=5,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Dataset auto-detection (unchanged from original)                   #
    # ------------------------------------------------------------------ #

    def _auto_detect_datasets(self, scope):
        import pandas as pd
        import numpy as np
        for var_name, var_value in scope.items():
            if var_name.startswith("_"):
                continue
            if isinstance(var_value, pd.DataFrame):
                dataset_info = {
                    "name": var_name, "type": "DataFrame",
                    "shape": var_value.shape, "columns": list(var_value.columns),
                    "size": len(var_value), "format": "csv",
                    "modality": "Tabular", "libraries": ["pandas"],
                    "tags": ["tabular", "dataframe"],
                }
                if "target" in var_value.columns or "label" in var_value.columns:
                    dataset_info["task_type"] = "Classification"
                    dataset_info["tags"].append("classification")
                elif any("price" in c.lower() or "value" in c.lower() for c in var_value.columns):
                    dataset_info["task_type"] = "Regression"
                    dataset_info["tags"].append("regression")
                else:
                    dataset_info["task_type"] = "Tabular Task"
                if any(k in var_name.lower() for k in ["train", "x_train", "y_train"]):
                    dataset_info["purpose"] = "training"
                elif any(k in var_name.lower() for k in ["test", "x_test", "y_test", "val", "valid"]):
                    dataset_info["purpose"] = "validation"
                if not any(d["name"] == var_name for d in self._tracked_datasets):
                    self._tracked_datasets.append(dataset_info)
                    print(f"[AlphaRed] Auto-detected dataset: {var_name} {var_value.shape}")
            elif isinstance(var_value, np.ndarray) and var_value.ndim >= 1:
                dataset_info = {
                    "name": var_name, "type": "ndarray",
                    "shape": var_value.shape,
                    "size": var_value.shape[0] if len(var_value.shape) > 0 else 0,
                    "format": "numpy", "libraries": ["numpy"], "tags": ["numpy", "array"],
                }
                if len(var_value.shape) >= 3:
                    dataset_info.update({"modality": "Computer Vision", "task_type": "Image Classification"})
                    dataset_info["tags"].extend(["images", "vision"])
                else:
                    dataset_info.update({"modality": "Tabular", "task_type": "Tabular Task"})
                    dataset_info["tags"].append("tabular")
                if any(k in var_name.lower() for k in ["train", "x_train", "y_train"]):
                    dataset_info["purpose"] = "training"
                elif any(k in var_name.lower() for k in ["test", "val", "valid"]):
                    dataset_info["purpose"] = "validation"
                if not any(d["name"] == var_name for d in self._tracked_datasets):
                    self._tracked_datasets.append(dataset_info)
                    print(f"[AlphaRed] Auto-detected dataset: {var_name} {var_value.shape}")
            else:
                var_type   = type(var_value).__name__
                var_module = type(var_value).__module__
                if "DataLoader" in var_type or ("torch" in var_module and hasattr(var_value, "dataset")):
                    try:
                        dataset     = var_value.dataset
                        dataset_info = {
                            "name": var_name, "type": "PyTorch DataLoader",
                            "dataset_type": type(dataset).__name__,
                            "batch_size": getattr(var_value, "batch_size", "unknown"),
                            "format": "pytorch", "libraries": ["PyTorch"], "tags": ["pytorch", "dataloader"],
                        }
                        if hasattr(dataset, "__len__"):
                            dataset_info["size"] = len(dataset)
                        dl = type(dataset).__name__.lower()
                        if any(k in dl for k in ["mnist", "cifar", "imagenet", "vision"]):
                            dataset_info.update({"modality": "Computer Vision", "task_type": "Image Classification"})
                            dataset_info["tags"].extend(["images", "vision"])
                        elif any(k in dl for k in ["text", "nlp", "language"]):
                            dataset_info.update({"modality": "Natural Language", "task_type": "Text Classification"})
                            dataset_info["tags"].extend(["text", "nlp"])
                        else:
                            dataset_info.update({"modality": "Tabular", "task_type": "Deep Learning Task"})
                        dataset_info["purpose"] = "training" if "train" in var_name.lower() else "validation"
                        if not any(d["name"] == var_name for d in self._tracked_datasets):
                            self._tracked_datasets.append(dataset_info)
                            print(f"[AlphaRed] Auto-detected DataLoader: {var_name} size={dataset_info.get('size','?')}")
                    except Exception as e:
                        print(f"[AlphaRed] Warning: Could not detect DataLoader {var_name}: {e}")

    def _register_datasets_to_registry(self):
        if not self.run_id or not self._tracked_datasets:
            return
        try:
            datasets_payload = []
            for ds in self._tracked_datasets:
                payload = {k: ds[k] for k in
                           ["name","type","purpose","task_type","modality","format","tags","libraries"]
                           if k in ds}
                if "shape" in ds:
                    payload["shape"] = list(ds["shape"])
                if "size" in ds:
                    payload["size"] = ds["size"]
                if "batch_size" in ds:
                    payload["batch_size"] = ds["batch_size"]
                if "columns" in ds:
                    payload["columns"] = ds["columns"]
                datasets_payload.append(payload)
            response = self._session.post(
                f"{self.host}/api/sdk/datasets/register",
                json={"run_id": self.run_id, "project": self.project,
                      "datasets": datasets_payload, "model_type": self._model_type},
                timeout=10,
            )
            if response.status_code == 200:
                print(f"[AlphaRed] Datasets registered to Data Registry")
        except Exception as e:
            print(f"[AlphaRed] Warning: Could not register datasets: {e}")

    def _create_dataset_artifacts(self):
        import os
        for dataset_info in self._tracked_datasets:
            try:
                dataset_name = dataset_info["name"]
                dataset_id   = dataset_name.lower().replace("_", "-")
                dataset_type = dataset_info.get("type", "unknown")
                upload_dir   = "backend/app/apis/uploads"
                os.makedirs(upload_dir, exist_ok=True)
                if "PyTorch" in dataset_type or "torch" in dataset_type.lower():
                    self._session.post(
                        f"{self.host}/api/artifacts",
                        json={
                            "name": f"dataset-{dataset_id}", "type": "dataset",
                            "run_id": self.run_id, "path": f"pytorch://{dataset_name}",
                            "size": dataset_info.get("size", 0),
                            "metadata": {
                                "dataset_id": dataset_id, "original_name": dataset_name,
                                "dataset_type": dataset_info.get("dataset_type", "unknown"),
                                "purpose": dataset_info.get("purpose", "training"),
                                "created_by": "sdk_auto", "framework": "pytorch",
                            },
                        },
                        timeout=5,
                    )
                    if dataset_info.get("purpose") == "training" and not self._dataset_id:
                        self._dataset_id = dataset_id
                else:
                    patterns  = [
                        f"{upload_dir}/{dataset_id}_val.csv",
                        f"{upload_dir}/{dataset_id}_train.csv",
                        f"{upload_dir}/{dataset_id}.csv",
                    ]
                    file_path = next((p for p in patterns if os.path.exists(p)), None)
                    path_str  = file_path or f"memory://{dataset_name}"
                    size      = os.path.getsize(file_path) if file_path else 0
                    self._session.post(
                        f"{self.host}/api/artifacts",
                        json={
                            "name": f"dataset-{dataset_id}", "type": "dataset",
                            "run_id": self.run_id, "path": path_str, "size": size,
                            "metadata": {
                                "dataset_id": dataset_id, "original_name": dataset_name,
                                "shape": str(dataset_info.get("shape", "unknown")),
                                "purpose": dataset_info.get("purpose", "unknown"),
                                "created_by": "sdk_auto",
                            },
                        },
                        timeout=5,
                    )
                    if not self._dataset_id:
                        self._dataset_id = dataset_id
                print(f"[AlphaRed]   Dataset artifact created: {dataset_id}")
            except Exception as e:
                print(f"[AlphaRed] Warning: Could not create artifact for {dataset_info['name']}: {e}")

    def _extract_pytorch_samples(self, dataset_obj, dataset_id):
        import os, pandas as pd, torch
        try:
            upload_dir = "backend/app/apis/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            dataset    = dataset_obj.dataset if hasattr(dataset_obj, "dataset") else dataset_obj
            samples    = []
            for i in range(min(1000, len(dataset))):
                try:
                    item = dataset[i]
                    if isinstance(item, tuple) and len(item) == 2:
                        data, label = item
                        data_np     = data.numpy().flatten() if torch.is_tensor(data) else (
                            data.flatten() if hasattr(data, "flatten") else [data])
                        s = {f"feature_{j}": v for j, v in enumerate(data_np)}
                        s["label"] = int(label) if torch.is_tensor(label) else label
                        samples.append(s)
                except Exception:
                    continue
            if samples:
                df        = pd.DataFrame(samples)
                split_idx = int(len(df) * 0.8)
                df.iloc[:split_idx].to_csv(f"{upload_dir}/ds-{self.run_id}_train.csv", index=False)
                df.iloc[split_idx:].to_csv(f"{upload_dir}/ds-{self.run_id}_val.csv",   index=False)
                print(f"[AlphaRed] Extracted {len(samples)} samples from PyTorch dataset")
        except Exception as e:
            print(f"[AlphaRed] Warning: Could not extract PyTorch samples: {e}")

    # ------------------------------------------------------------------ #
    #  Background workers                                                  #
    # ------------------------------------------------------------------ #

    def _worker(self):
        """Consumes metrics/logs from queue and sends to backend in batches."""
        batch      = []
        last_flush = time.time()
        while self.running or not self._queue.empty():
            try:
                try:
                    item = self._queue.get(timeout=0.1)
                    batch.append(item)
                    self._queue.task_done()
                except queue.Empty:
                    pass
                current_time = time.time()
                if len(batch) >= 100 or (batch and current_time - last_flush > 0.1):
                    self._flush_batch(batch)
                    batch      = []
                    last_flush = current_time
            except Exception as e:
                print(f"[AlphaRed] Log worker error: {e}")
                batch = []

    def _flush_batch(self, batch):
        if not batch or not self.run_id:
            return
        try:
            self._session.post(
                f"{self.host}/api/sdk/runs/{self.run_id}/batch",
                json={"items": batch},
                headers={"X-API-KEY": self.api_key} if self.api_key else {},
                timeout=5,
            )
        except Exception:
            pass

    def log_text(self, message, level="INFO"):
        if not self.running:
            return
        self._queue.put({
            "type": "text", "message": message,
            "level": level, "timestamp": time.time(),
        })

    def _telemetry_worker(self):
        """Sends CPU/GPU metrics every 5 seconds."""
        while self.running:
            try:
                metrics = get_system_metrics()
                self._send_payload({"type": "telemetry", "data": metrics, "timestamp": time.time()})
                time.sleep(5)
            except Exception as e:
                print(f"[AlphaRed] Telemetry error: {e}")
                time.sleep(10)

    def _send_payload(self, payload):
        if not self.run_id:
            return
        try:
            endpoint = "log_text" if payload.get("type") == "text" else "log"
            self._session.post(
                f"{self.host}/api/sdk/runs/{self.run_id}/{endpoint}",
                json=payload,
                headers={"X-API-KEY": self.api_key} if self.api_key else {},
                timeout=5,
            )
        except Exception:
            pass
