"""
ChangesLogger — Records every experiment attempt to changes.md
Replaces AutoResearch's human-written program.md with agent-written changes.md
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from .hypothesis_engine import Hypothesis


@dataclass
class ExperimentRecord:
    """Stores result of one RatchetLoop experiment."""
    experiment_num:    int
    hypothesis:        Hypothesis
    baseline_metrics:  dict
    new_metrics:       dict
    improved:          bool
    improvement_delta: float
    duration_seconds:  float
    commit_hash:       str   = ""         # git commit hash (AutoResearch alignment)
    vram_mb:           float = 0.0        # peak VRAM usage
    patch_applied:     bool  = False      # whether code_patch was successfully applied
    status:            str   = ""         # keep | discard | crash (AutoResearch values)
    timestamp:         str   = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self):
        # Auto-derive status from improved flag if not explicitly set
        if not self.status:
            self.status = "keep" if self.improved else "discard"

    @property
    def status_icon(self) -> str:
        icons = {"keep": "✅ COMMITTED", "discard": "❌ REVERTED", "crash": "💥 CRASH"}
        return icons.get(self.status, "❌ REVERTED")

    def to_dict(self) -> dict:
        return {
            "experiment_num":    self.experiment_num,
            "hypothesis_id":     self.hypothesis.id,
            "title":             self.hypothesis.title,
            "improved":          self.improved,
            "improvement_delta": round(self.improvement_delta, 4),
            "baseline_metrics":  self.baseline_metrics,
            "new_metrics":       self.new_metrics,
            "duration_seconds":  round(self.duration_seconds, 1),
            "commit_hash":       self.commit_hash,
            "vram_mb":           self.vram_mb,
            "patch_applied":     self.patch_applied,
            "status":            self.status,
            "timestamp":         self.timestamp,
        }


class ChangesLogger:
    """
    Writes changes.md after every experiment.
    Developer reads this in the morning to understand what the agent did overnight.

    Usage:
        logger = ChangesLogger(run_id="abc123", output_dir="./")
        logger.log(record)          # called by RatchetLoop after each experiment
        logger.write_summary()      # called at end of agent session
    """

    def __init__(self, run_id: str, output_dir: str = "./"):
        self.run_id     = run_id
        self.output_dir = output_dir
        self.records:   list[ExperimentRecord] = []
        self._md_path   = os.path.join(output_dir, "changes.md")
        self._json_path = os.path.join(output_dir, "changes.json")
        self._init_file()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def log(self, record: ExperimentRecord):
        """Called by RatchetLoop after each experiment completes."""
        self.records.append(record)
        self._append_to_md(record)
        self._save_json()

    def write_summary(self):
        """Called at end of agent session. Appends overall summary to changes.md."""
        summary = self._build_summary()
        with open(self._md_path, "a") as f:
            f.write(summary)

    def get_best_record(self) -> Optional[ExperimentRecord]:
        """Returns the experiment with the highest improvement delta."""
        committed = [r for r in self.records if r.improved]
        if not committed:
            return None
        return max(committed, key=lambda r: r.improvement_delta)

    def committed_count(self) -> int:
        return sum(1 for r in self.records if r.improved)

    def reverted_count(self) -> int:
        return sum(1 for r in self.records if not r.improved)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _init_file(self):
        """Creates changes.md and changes.json on first run."""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self._md_path, "w") as f:
            f.write(self._header())
        # Create empty JSON immediately so it always exists after init
        self._save_json()

    def _header(self) -> str:
        return f"""# Agent Experiment Log
**Run ID:** `{self.run_id}`  
**Started:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}  
**Mode:** Autonomous — agent runs experiments overnight, developer reviews in morning

---

"""

    def _append_to_md(self, record: ExperimentRecord):
        """Appends one experiment block to changes.md."""
        with open(self._md_path, "a") as f:
            f.write(self._format_record(record))

    def _format_record(self, record: ExperimentRecord) -> str:
        """Formats one ExperimentRecord as a readable markdown block."""

        # Build metrics comparison table
        baseline = record.baseline_metrics
        new      = record.new_metrics
        all_keys = sorted(set(list(baseline.keys()) + list(new.keys())))

        metrics_rows = ""
        for key in all_keys:
            old_val = baseline.get(key, "—")
            new_val = new.get(key, "—")

            # Highlight if changed
            if isinstance(old_val, float) and isinstance(new_val, float):
                diff = new_val - old_val
                arrow = f" ↑ +{diff:.4f}" if diff > 0 else (f" ↓ {diff:.4f}" if diff < 0 else " →")
                new_display = f"`{new_val:.4f}`{arrow}"
            else:
                new_display = f"`{new_val}`"

            old_display = f"`{old_val:.4f}`" if isinstance(old_val, float) else f"`{old_val}`"
            metrics_rows += f"| {key} | {old_display} | {new_display} |\n"

        # Build code change block
        code_change_str = json.dumps(record.hypothesis.code_change, indent=2)
        patch_preview = (record.hypothesis.code_patch or "").strip()
        if len(patch_preview) > 2000:
            patch_preview = patch_preview[:2000] + "\n...<truncated>..."

        delta_str = (
            f"+{record.improvement_delta*100:.2f}%"
            if record.improvement_delta >= 0
            else f"{record.improvement_delta*100:.2f}%"
        )

        return f"""## Experiment #{record.experiment_num:03d} — {record.status_icon}

| Field | Value |
|---|---|
| **Hypothesis** | {record.hypothesis.title} |
| **Problem** | {record.hypothesis.problem} |
| **Status** | {record.status} |
| **Commit** | `{record.commit_hash or "n/a"}` |
| **Result** | {delta_str} change in key metric |
| **VRAM** | {record.vram_mb:.0f} MB |
| **Patch applied** | {"yes" if record.patch_applied else "no"} |
| **Duration** | {record.duration_seconds:.1f}s |
| **Time** | {record.timestamp} |

### What changed
{record.hypothesis.description}

```json
{code_change_str}
```

### Proposed code patch
```diff
{patch_preview if patch_preview else "# no code_patch returned"}
```

### Metrics before vs after

| Metric | Before | After |
|---|---|---|
{metrics_rows}
### Agent's reasoning
> *Estimated gain: {record.hypothesis.estimated_gain}*

---

"""

    def _build_summary(self) -> str:
        """Builds the final summary section appended at end of session."""
        total     = len(self.records)
        committed = self.committed_count()
        reverted  = self.reverted_count()
        best      = self.get_best_record()

        total_duration = sum(r.duration_seconds for r in self.records)
        hours = int(total_duration // 3600)
        mins  = int((total_duration % 3600) // 60)

        best_section = ""
        if best:
            best_section = f"""
### Best experiment
- **#{best.experiment_num:03d}** — {best.hypothesis.title}
- Improvement: **+{best.improvement_delta*100:.2f}%**
"""

        committed_list = ""
        for r in self.records:
            if r.improved:
                committed_list += f"- #{r.experiment_num:03d} {r.hypothesis.title} (+{r.improvement_delta*100:.2f}%)\n"

        return f"""---

## Session Summary

| Stat | Value |
|---|---|
| Total experiments | {total} |
| Committed (improved) | {committed} |
| Reverted (no gain) | {reverted} |
| Success rate | {(committed/total*100) if total > 0 else 0:.1f}% | 
| Total runtime | {hours}h {mins}m |
{best_section}
### All committed improvements
{committed_list if committed_list else "— none —"}

### Next steps
The agent suggests continuing with remaining hypotheses for detected problems.
Review committed changes above and click **Promote** in the dashboard to push to production.

*Generated by AlphaRed Agent — {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}*
"""

    def _save_json(self):
        """Saves all records as JSON — used by backend/dashboard."""
        data = {
            "run_id":   self.run_id,
            "records":  [r.to_dict() for r in self.records],
            "stats": {
                "total":     len(self.records),
                "committed": self.committed_count(),
                "reverted":  self.reverted_count(),
            }
        }
        with open(self._json_path, "w") as f:
            json.dump(data, f, indent=2)
