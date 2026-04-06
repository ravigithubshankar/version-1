"""
RatchetLoop -- Autonomous agent loop following AutoResearch core logic exactly.

AutoResearch loop (program.md):
  1. Create branch autoresearch/<tag>
  2. Read train.py for context
  3. Run baseline (no changes)
  4. LOOP FOREVER:
     a. Groq LLM reads code + history -> proposes code_patch
     b. Apply patch to train file
     c. git commit
     d. Run training with fixed time budget
     e. Read metric from output
     f. If improved -> keep commit (advance branch)
     g. If not -> git reset --hard to pre-experiment state
     h. Log to results.tsv + changes.md
  5. Never stop until budget / manual stop
"""

import os
import re
import time
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable
from .diagnostics_engine import DiagnosticsEngine, ProblemType, DiagnosisResult
from .hypothesis_engine import HypothesisEngine, Hypothesis
from .changes_logger import ChangesLogger, ExperimentRecord


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass
class AgentBudget:
    max_experiments:     int   = 50
    max_cost_usd:        float = 10.0
    max_duration_sec:    float = 8 * 3600       # overnight default
    target_metric:       Optional[float] = None  # stop early if primary metric hits this
    time_per_experiment: float = 5 * 60          # 5 min -- AutoResearch default

    def is_exhausted(self, experiments_done, cost_so_far, elapsed_sec, best_metric):
        if experiments_done >= self.max_experiments:
            return True, f"Reached max experiments ({self.max_experiments})"
        if cost_so_far >= self.max_cost_usd:
            return True, f"Reached cost budget (${self.max_cost_usd:.2f})"
        if elapsed_sec >= self.max_duration_sec:
            return True, f"Reached time budget ({self.max_duration_sec/3600:.1f}h)"
        if self.target_metric and best_metric and best_metric >= self.target_metric:
            return True, f"Hit target metric ({self.target_metric})"
        return False, ""


# ---------------------------------------------------------------------------
# ExperimentConfig -- passed to train_fn
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    hypothesis:      Hypothesis
    code_change:     dict
    code_patch:      str         # actual code patch from LLM (AutoResearch alignment)
    experiment_num:  int
    time_budget_sec: float = 300.0
    git_branch:      str   = ""
    train_file_path: str   = ""  # path to the modified train file


ExperimentFn = Callable[[ExperimentConfig], dict]


# ---------------------------------------------------------------------------
# GitOps -- exact AutoResearch mechanics
# ---------------------------------------------------------------------------

class GitOps:
    """
    Mirrors AutoResearch git usage:
      - create_branch: git checkout -b autoresearch/<tag>
      - commit:        git add -A && git commit -m "..."  -> returns short hash
      - reset_to:      git reset --hard <hash>            -> discard experiment
      - current_hash:  git rev-parse --short HEAD
    """

    def __init__(self, repo_dir: str, enabled: bool = True):
        self.repo_dir = repo_dir
        self.enabled  = enabled

    def _run(self, cmd: list[str], check: bool = True) -> tuple[int, str, str]:
        result = subprocess.run(cmd, cwd=self.repo_dir, capture_output=True, text=True)
        if check and result.returncode != 0:
            raise RuntimeError(f"Git error: {' '.join(cmd)}\n{result.stderr.strip()}")
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    def branch_exists(self, name: str) -> bool:
        if not self.enabled:
            return False
        rc, out, _ = self._run(["git", "branch", "--list", name], check=False)
        return bool(out.strip())

    def create_branch(self, name: str):
        """git checkout -b autoresearch/<tag>"""
        if not self.enabled:
            return
        if self.branch_exists(name):
            raise RuntimeError(f"Branch {name} already exists -- use a new tag")
        self._run(["git", "checkout", "-b", name])

    def current_branch(self) -> str:
        if not self.enabled:
            return "mock-branch"
        _, branch, _ = self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return branch

    def current_hash(self) -> str:
        if not self.enabled:
            return "mock0000"
        _, h, _ = self._run(["git", "rev-parse", "--short", "HEAD"])
        return h

    def commit(self, message: str) -> str:
        """Add all changes and commit. Returns short hash."""
        if not self.enabled:
            return "mock0000"
        self._run(["git", "add", "-A"])
        rc, _, _ = self._run(["git", "commit", "-m", message], check=False)
        if rc != 0:
            return self.current_hash()  # nothing changed
        return self.current_hash()

    def reset_to(self, commit_hash: str):
        """git reset --hard <hash> -- discard failed experiment."""
        if not self.enabled:
            return
        self._run(["git", "reset", "--hard", commit_hash])

    def is_git_repo(self) -> bool:
        rc, _, _ = self._run(["git", "rev-parse", "--git-dir"], check=False)
        return rc == 0


# ---------------------------------------------------------------------------
# CodeApplier -- applies LLM code_patch to the train file
# ---------------------------------------------------------------------------

class CodeApplier:
    """
    Applies code patches to the training file.

    AutoResearch: LLM directly edits train.py.
    We: Apply LLM-generated code_patch to the train file.

    Two patch modes:
    1. Full replacement: if patch looks like complete Python code
    2. Search-replace:  if patch has --- / +++ diff markers
    3. Append mode:     if patch is just new code to add
    """

    def apply(self, train_file_path: str, code_patch: str) -> bool:
        """
        Applies code_patch to train_file_path.
        Returns True if file was modified.
        """
        if not train_file_path or not os.path.exists(train_file_path):
            return False
        if not code_patch or not code_patch.strip():
            return False

        original = open(train_file_path).read()

        # Mode 1: unified diff format (--- +++ lines)
        if "--- " in code_patch and "+++ " in code_patch:
            patched = self._apply_unified_diff(original, code_patch)
            if patched and patched != original:
                open(train_file_path, "w").write(patched)
                return True

        # Mode 2: search-replace blocks (BEFORE/AFTER or OLD/NEW markers)
        if "BEFORE:" in code_patch or "OLD:" in code_patch:
            patched = self._apply_search_replace(original, code_patch)
            if patched and patched != original:
                open(train_file_path, "w").write(patched)
                return True

        # Mode 3: The LLM showed specific lines to change -- try line-level apply
        patched = self._apply_line_level(original, code_patch)
        if patched and patched != original:
            open(train_file_path, "w").write(patched)
            return True

        # Mode 4: fallback -- append patch as comment block for human review
        # We do NOT do this automatically -- instead signal no-change
        return False

    def _apply_unified_diff(self, original: str, patch: str) -> str:
        """Apply a unified diff patch."""
        try:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".orig", delete=False) as f:
                f.write(original)
                orig_path = f.name
            with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
                f.write(patch)
                patch_path = f.name
            result = subprocess.run(
                ["patch", "-o", "-", orig_path, patch_path],
                capture_output=True, text=True
            )
            os.unlink(orig_path)
            os.unlink(patch_path)
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        return original

    def _apply_search_replace(self, original: str, patch: str) -> str:
        """Extract BEFORE/AFTER or OLD/NEW blocks and replace."""
        for before_key, after_key in [("BEFORE:", "AFTER:"), ("OLD:", "NEW:")]:
            if before_key in patch and after_key in patch:
                try:
                    before_part = patch.split(before_key)[1].split(after_key)[0].strip()
                    after_part  = patch.split(after_key)[1].strip()
                    # Strip code fences
                    before_part = re.sub(r"^```\w*|```$", "", before_part, flags=re.MULTILINE).strip()
                    after_part  = re.sub(r"^```\w*|```$", "", after_part,  flags=re.MULTILINE).strip()
                    if before_part in original:
                        return original.replace(before_part, after_part, 1)
                except Exception:
                    pass
        return original

    def _apply_line_level(self, original: str, patch: str) -> str:
        """
        Try to find specific lines from patch in original and replace them.
        Works when LLM says "change line X from ... to ..."
        """
        # Extract lines starting with - (remove) and + (add) from patch
        minus_lines = []
        plus_lines  = []
        for line in patch.split("\n"):
            if line.startswith("- ") or line.startswith("-\t"):
                minus_lines.append(line[2:])
            elif line.startswith("+ ") or line.startswith("+\t"):
                plus_lines.append(line[2:])

        if not minus_lines or len(minus_lines) != len(plus_lines):
            return original

        result = original
        for old_line, new_line in zip(minus_lines, plus_lines):
            if old_line in result:
                result = result.replace(old_line, new_line, 1)

        return result


# ---------------------------------------------------------------------------
# ResultsTSV -- AutoResearch output format
# ---------------------------------------------------------------------------

class ResultsTSV:
    """
    Writes results.tsv (tab-separated, NOT git-tracked -- same as AutoResearch).
    Format: commit | val_metric | memory_gb | status | description
    """

    HEADER = "commit\tval_metric\tmemory_gb\tstatus\tdescription\n"

    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(self.HEADER)

    def append(self, commit_hash: str, val_metric: float,
               memory_gb: float, status: str, description: str):
        with open(self.path, "a") as f:
            f.write(f"{commit_hash}\t{val_metric:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")


# ---------------------------------------------------------------------------
# RatchetLoop
# ---------------------------------------------------------------------------

class RatchetLoop:
    """
    Autonomous improvement loop -- AutoResearch core logic.

    Usage:
        def my_train_fn(config: ExperimentConfig) -> dict:
            # config.code_patch has been applied to config.train_file_path
            # Train for config.time_budget_sec
            # Return metrics: val_accuracy or val_loss, optionally vram_mb
            return {"val_accuracy": 0.85, "val_loss": 0.38, "vram_mb": 4096}

        loop = RatchetLoop(
            run_id          = "apr05",
            train_fn        = my_train_fn,
            train_file_path = "./train.py",      # file LLM edits (AutoResearch: train.py)
            repo_dir        = "./",
            groq_api_key    = "gsk_...",
            budget          = AgentBudget(max_experiments=50),
            output_dir      = "./runs/apr05/",
        )
        summary = loop.run(baseline_metrics={"val_accuracy": 0.81})
    """

    PRIMARY_METRIC      = "val_accuracy"
    FALLBACK_METRIC     = "val_loss"
    COST_PER_EXPERIMENT = 0.05

    def __init__(
        self,
        run_id:           str,
        train_fn:         ExperimentFn,
        train_file_path:  str  = "",          # path to file agent edits (train.py equivalent)
        model_type:       str  = "deep_learning",
        budget:           Optional[AgentBudget] = None,
        output_dir:       str  = "./",
        repo_dir:         str  = "./",
        openrouter_api_key: str = "",
        git_enabled:      bool = True,
        status_callback:  Optional[Callable[[str], None]] = None,
    ):
        self.run_id          = run_id
        self.train_fn        = train_fn
        self.train_file_path = train_file_path
        self.model_type      = model_type
        self.budget          = budget or AgentBudget()
        self.output_dir      = output_dir
        self._status_cb      = status_callback or print

        os.makedirs(output_dir, exist_ok=True)

        self._diagnostics    = DiagnosticsEngine()
        self._hypothesis_eng = HypothesisEngine(openrouter_api_key=openrouter_api_key)
        self._logger         = ChangesLogger(run_id=run_id, output_dir=output_dir)
        self._results_tsv    = ResultsTSV(os.path.join(output_dir, "results.tsv"))
        self._git            = GitOps(repo_dir=repo_dir, enabled=git_enabled)
        self._code_applier   = CodeApplier()

        # State
        self._baseline_metrics:   dict  = {}
        self._best_metrics:       dict  = {}
        self._experiments_done:   int   = 0
        self._cost_so_far:        float = 0.0
        self._start_time:         float = 0.0
        self._stopped:            bool  = False
        self._stop_reason:        str   = ""
        self._experiment_history: list  = []
        self._baseline_hash:      str   = ""

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self, baseline_metrics: dict) -> dict:
        """
        Starts the autonomous loop.
        Follows AutoResearch program.md exactly.
        """
        self._baseline_metrics = dict(baseline_metrics)
        self._best_metrics     = dict(baseline_metrics)
        self._start_time       = time.time()

        # Step 1: Create autoresearch/<tag> branch (AutoResearch setup step 2)
        branch_name = f"autoresearch/{self.run_id}"
        try:
            self._git.create_branch(branch_name)
            self._status(f"[RatchetLoop] Created branch: {branch_name}")
        except Exception as e:
            self._status(f"[RatchetLoop] Branch note: {e} (continuing on current branch)")

        self._baseline_hash = self._git.current_hash()

        # Step 2: Read model code for LLM context (AutoResearch setup step 3)
        model_code = self._read_train_file()

        self._status("=" * 60)
        self._status(f"[RatchetLoop] run_id:    {self.run_id}")
        self._status(f"[RatchetLoop] branch:    {self._git.current_branch()}")
        self._status(f"[RatchetLoop] train_file:{self.train_file_path or '(not set)'}")
        self._status(f"[RatchetLoop] budget:    {self.budget.max_experiments} exp / "
                     f"${self.budget.max_cost_usd} / {self.budget.max_duration_sec/3600:.1f}h")
        self._status(f"[RatchetLoop] llm:       {'OpenRouter' if self._hypothesis_eng._api_key else 'static fallback'}")
        self._status(f"[RatchetLoop] baseline:  {self._fmt(baseline_metrics)}")
        self._status("=" * 60)

        # Step 3: Log baseline to results.tsv
        self._results_tsv.append(
            commit_hash = self._baseline_hash,
            val_metric  = self._primary_value(baseline_metrics),
            memory_gb   = baseline_metrics.get("vram_mb", 0) / 1024,
            status      = "keep",
            description = "baseline",
        )
        self._diagnostics.update(baseline_metrics)

        # Step 4: LOOP FOREVER (AutoResearch: never stop)
        while not self._stopped:

            # Budget check
            elapsed = time.time() - self._start_time
            done, reason = self.budget.is_exhausted(
                self._experiments_done, self._cost_so_far, elapsed,
                self._best_metrics.get(self.PRIMARY_METRIC),
            )
            if done:
                self._stop(reason)
                break

            # Diagnose
            diagnosis = self._diagnostics.analyze(epoch=self._experiments_done + 1)
            if not diagnosis.is_problem():
                diagnosis = self._general_improvement_diagnosis()

            # Get hypothesis from Groq LLM (or static fallback)
            hypotheses = self._hypothesis_eng.generate(
                diagnosis          = diagnosis,
                model_type         = self.model_type,
                top_n              = 1,
                experiment_history = self._experiment_history,
                model_code         = model_code,
            )

            if not hypotheses:
                fallback = self._get_fallback_diagnosis(diagnosis.problem)
                if fallback is None:
                    self._stop("All hypotheses exhausted")
                    break
                hypotheses = self._hypothesis_eng.generate(
                    diagnosis          = fallback,
                    model_type         = self.model_type,
                    top_n              = 1,
                    experiment_history = self._experiment_history,
                    model_code         = model_code,
                )
                if not hypotheses:
                    self._stop("All hypotheses exhausted")
                    break
                diagnosis = fallback

            hypothesis = hypotheses[0]
            self._experiments_done += 1
            exp_num = self._experiments_done

            self._status(f"\n[Exp #{exp_num:03d}] {hypothesis.title}")
            self._status(f"             problem:  {diagnosis.problem.value}")
            self._status(f"             time_budget: {self.budget.time_per_experiment:.0f}s")

            # Snapshot pre-experiment git state
            pre_exp_hash = self._git.current_hash()

            # Apply code patch to train file (AutoResearch: LLM edits train.py)
            patch_applied = False
            if hypothesis.code_patch and self.train_file_path:
                patch_applied = self._code_applier.apply(
                    self.train_file_path, hypothesis.code_patch
                )
                if patch_applied:
                    self._status(f"[Exp #{exp_num:03d}] Code patch applied to {self.train_file_path}")
                    # Re-read model code for next iteration
                    model_code = self._read_train_file()
                else:
                    self._status(f"[Exp #{exp_num:03d}] Patch could not be applied -- running without file change")

            # Build experiment config
            config = ExperimentConfig(
                hypothesis       = hypothesis,
                code_change      = hypothesis.code_change,
                code_patch       = hypothesis.code_patch,
                experiment_num   = exp_num,
                time_budget_sec  = self.budget.time_per_experiment,
                git_branch       = self._git.current_branch(),
                train_file_path  = self.train_file_path,
            )

            # Run experiment with timeout (AutoResearch: kill at 2x budget)
            exp_start   = time.time()
            new_metrics = {}
            status_flag = "discard"

            try:
                new_metrics = self._run_with_timeout(config)
                status_flag = "discard"  # updated to "keep" below if improved
            except TimeoutError:
                self._status(f"[Exp #{exp_num:03d}] TIMEOUT -- treating as crash")
                status_flag = "crash"
                self._git.reset_to(pre_exp_hash)
                self._results_tsv.append(pre_exp_hash, 0.0, 0.0, "crash",
                                         f"TIMEOUT: {hypothesis.title}")
                self._hypothesis_eng.mark_tried(hypothesis, improved=False)
                self._cost_so_far += self.COST_PER_EXPERIMENT
                continue
            except Exception as e:
                self._status(f"[Exp #{exp_num:03d}] CRASH: {e}")
                status_flag = "crash"
                self._git.reset_to(pre_exp_hash)
                self._results_tsv.append(pre_exp_hash, 0.0, 0.0, "crash",
                                         f"CRASH: {hypothesis.title}")
                self._hypothesis_eng.mark_tried(hypothesis, improved=False)
                self._cost_so_far += self.COST_PER_EXPERIMENT
                continue
            finally:
                duration = time.time() - exp_start
                self._cost_so_far += self.COST_PER_EXPERIMENT

            # Ratchet decision
            improved, delta = self._is_improvement(self._best_metrics, new_metrics)

            if improved:
                # KEEP -- git commit, advance branch (AutoResearch: keep the commit)
                commit_hash = self._git.commit(
                    f"autoresearch exp#{exp_num:03d}: {hypothesis.title} delta={delta:+.4f}"
                )
                status_flag = "keep"
                self._best_metrics = dict(new_metrics)
                self._diagnostics.update(new_metrics)
                self._status(f"[Exp #{exp_num:03d}] COMMITTED  delta={delta:+.4f}  hash={commit_hash}")
            else:
                # DISCARD -- git reset --hard (AutoResearch: revert to pre-experiment)
                self._git.reset_to(pre_exp_hash)
                model_code  = self._read_train_file()  # reload original code
                commit_hash = pre_exp_hash
                status_flag = "discard"
                self._status(f"[Exp #{exp_num:03d}] REVERTED   delta={delta:+.4f}")

            # Log results
            vram_gb = new_metrics.get("vram_mb", 0) / 1024
            self._results_tsv.append(
                commit_hash = commit_hash,
                val_metric  = self._primary_value(new_metrics),
                memory_gb   = vram_gb,
                status      = status_flag,
                description = hypothesis.title,
            )

            record = ExperimentRecord(
                experiment_num    = exp_num,
                hypothesis        = hypothesis,
                baseline_metrics  = dict(self._best_metrics),
                new_metrics       = new_metrics,
                improved          = improved,
                improvement_delta = delta,
                duration_seconds  = duration,
                commit_hash       = commit_hash,
                vram_mb           = new_metrics.get("vram_mb", 0.0),
                status            = status_flag,
            )
            self._logger.log(record)
            self._hypothesis_eng.mark_tried(hypothesis, improved=improved)

            self._experiment_history.append({
                "title":             hypothesis.title,
                "improved":          improved,
                "improvement_delta": delta,
                "status":            status_flag,
            })

        # Session complete
        self._logger.write_summary()
        summary = self._build_summary()

        self._status("\n" + "=" * 60)
        self._status(f"[RatchetLoop] DONE: {self._stop_reason}")
        self._status(f"[RatchetLoop] experiments={self._experiments_done}  "
                     f"committed={self._logger.committed_count()}  "
                     f"reverted={self._logger.reverted_count()}")
        self._status(f"[RatchetLoop] best: {self._fmt(self._best_metrics)}")
        self._status("=" * 60)

        return summary

    def run_async(self, baseline_metrics: dict) -> threading.Thread:
        thread = threading.Thread(
            target=self.run, args=(baseline_metrics,),
            daemon=True, name=f"ratchet-{self.run_id}"
        )
        thread.start()
        return thread

    def stop(self):
        self._stop("Stopped by user")

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _read_train_file(self) -> str:
        """Reads train file content for LLM context."""
        if self.train_file_path and os.path.exists(self.train_file_path):
            try:
                return open(self.train_file_path).read()
            except Exception:
                pass
        return ""

    def _run_with_timeout(self, config: ExperimentConfig) -> dict:
        """
        Runs train_fn with hard timeout at 2x time_budget.
        AutoResearch: kill runs exceeding 10min when budget is 5min.
        """
        timeout = config.time_budget_sec * 2
        result  = {}
        error   = []

        def target():
            try:
                result.update(self.train_fn(config))
            except Exception as e:
                error.append(e)

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            raise TimeoutError(f"Exceeded {timeout:.0f}s timeout")
        if error:
            raise error[0]
        if not result:
            raise RuntimeError("train_fn returned empty dict")
        return result

    def _is_improvement(self, baseline: dict, new: dict) -> tuple[bool, float]:
        if self.PRIMARY_METRIC in baseline and self.PRIMARY_METRIC in new:
            delta = new[self.PRIMARY_METRIC] - baseline[self.PRIMARY_METRIC]
            return delta > 0, delta
        if self.FALLBACK_METRIC in baseline and self.FALLBACK_METRIC in new:
            delta = new[self.FALLBACK_METRIC] - baseline[self.FALLBACK_METRIC]
            return delta < 0, -delta
        return False, 0.0

    def _primary_value(self, metrics: dict) -> float:
        return metrics.get(self.PRIMARY_METRIC, metrics.get(self.FALLBACK_METRIC, 0.0))

    def _general_improvement_diagnosis(self) -> DiagnosisResult:
        return DiagnosisResult(
            problem    = ProblemType.PLATEAU,
            confidence = 0.5,
            message    = "No specific problem -- attempting general improvement",
            suggestion = "add_lr_scheduler|change_optimizer",
            severity   = "info",
            epoch      = self._experiments_done,
        )

    def _get_fallback_diagnosis(self, exhausted: ProblemType) -> Optional[DiagnosisResult]:
        for problem in [ProblemType.OVERFITTING, ProblemType.UNDERFITTING,
                        ProblemType.PLATEAU, ProblemType.EXPLODING_GRADIENT,
                        ProblemType.NAN_LOSS]:
            if problem != exhausted and self._hypothesis_eng.remaining_count(problem) > 0:
                return DiagnosisResult(
                    problem    = problem,
                    confidence = 0.4,
                    message    = f"Rotating to {problem.value} for continued improvement",
                    suggestion = "",
                    severity   = "info",
                    epoch      = self._experiments_done,
                )
        return None

    def _stop(self, reason: str):
        self._stopped     = True
        self._stop_reason = reason

    def _status(self, msg: str):
        self._status_cb(msg)

    def _fmt(self, metrics: dict) -> str:
        return "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )

    def _build_summary(self) -> dict:
        best_record = self._logger.get_best_record()
        return {
            "run_id":             self.run_id,
            "stop_reason":        self._stop_reason,
            "experiments_done":   self._experiments_done,
            "committed":          self._logger.committed_count(),
            "reverted":           self._logger.reverted_count(),
            "cost_usd":           round(self._cost_so_far, 4),
            "duration_sec":       round(time.time() - self._start_time, 1),
            "baseline_metrics":   self._baseline_metrics,
            "best_metrics":       self._best_metrics,
            "best_experiment":    best_record.to_dict() if best_record else None,
            "experiment_history": self._experiment_history,
            "changes_md_path":    self._logger._md_path,
            "changes_json_path":  self._logger._json_path,
            "results_tsv_path":   self._results_tsv.path,
        }