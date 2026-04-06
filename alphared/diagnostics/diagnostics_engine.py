"""
DiagnosticsEngine — Real-time training problem detector
Monitors metrics history and detects: NaN, Overfitting, Underfitting, Plateau, Exploding Gradients
"""

import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class ProblemType(Enum):
    NAN_LOSS            = "nan_loss"
    OVERFITTING         = "overfitting"
    UNDERFITTING        = "underfitting"
    PLATEAU             = "plateau"
    EXPLODING_GRADIENT  = "exploding_gradient"
    HEALTHY             = "healthy"


@dataclass
class DiagnosisResult:
    problem:     ProblemType
    confidence:  float          # 0.0 - 1.0
    message:     str            # Human readable — dashboard లో చూపిస్తుంది
    suggestion:  str            # forwarded to HypothesisEngine
    severity:    str            # "critical" | "warning" | "info"
    epoch:       int = 0

    def is_problem(self) -> bool:
        return self.problem != ProblemType.HEALTHY

    def to_dict(self) -> dict:
        return {
            "problem":    self.problem.value,
            "confidence": round(self.confidence, 3),
            "message":    self.message,
            "suggestion": self.suggestion,
            "severity":   self.severity,
            "epoch":      self.epoch,
        }


@dataclass
class MetricsHistory:
    """Stores metrics from SDK run.log() calls."""
    train_loss:    list = field(default_factory=list)
    val_loss:      list = field(default_factory=list)
    train_accuracy:list = field(default_factory=list)
    val_accuracy:  list = field(default_factory=list)
    gradients:     list = field(default_factory=list)  # optional

    def add(self, metrics: dict):
        """Append one metrics dict from run.log() call."""
        if "train_loss"     in metrics: self.train_loss.append(float(metrics["train_loss"]))
        elif "loss"         in metrics: self.train_loss.append(float(metrics["loss"]))
        if "val_loss"       in metrics: self.val_loss.append(float(metrics["val_loss"]))
        if "train_accuracy" in metrics: self.train_accuracy.append(float(metrics["train_accuracy"]))
        if "val_accuracy"   in metrics: self.val_accuracy.append(float(metrics["val_accuracy"]))
        if "grad_norm"      in metrics: self.gradients.append(float(metrics["grad_norm"]))

    @property
    def epoch_count(self) -> int:
        return len(self.train_loss)


class DiagnosticsEngine:
    """
    Real-time training diagnostics.

    Usage (RunSession లో):
        self._diagnostics = DiagnosticsEngine()
        # log() లో ప్రతిసారి:
        result = self._diagnostics.analyze(metrics_dict)
        if result.is_problem():
            self._send_alert(result)
    """

    # --- Tunable thresholds ---
    OVERFIT_GAP          = 0.15   # val_loss - train_loss > 15% → overfitting
    OVERFIT_MIN_EPOCHS   = 5      # కనీసం ఇన్ని epochs తర్వాత మాత్రమే check
    PLATEAU_WINDOW       = 5      # చివరి N epochs లో improvement చూస్తుంది
    PLATEAU_MIN_DELTA    = 0.001  # ఇంతకంటే తక్కువ improvement → plateau
    PLATEAU_MIN_EPOCHS   = 10     # plateau కి minimum epochs
    SPIKE_MULTIPLIER     = 3.0    # loss N రెట్లు పెరిగితే → exploding
    GRAD_EXPLODE_THRESH  = 100.0  # gradient norm ఇంతకంటే ఎక్కువ → exploding
    UNDERFIT_LOSS_THRESH = 0.7    # loss ఇంత high గా ఉంటే + no improvement
    UNDERFIT_ACC_THRESH  = 0.6    # accuracy ఇంతకంటే తక్కువ

    def __init__(self):
        self.history          = MetricsHistory()
        self._last_diagnosis  = ProblemType.HEALTHY
        self._alert_cooldown  = {}   # Same problem ని repeatedly alert చేయకుండా

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(self, metrics: dict):
        """run.log() call చేసిన ప్రతిసారి ఇది call అవుతుంది"""
        self.history.add(metrics)

    def analyze(self, epoch: Optional[int] = None) -> DiagnosisResult:
        """
        Present metrics base గా diagnosis return చేస్తుంది.
        Caller (RunSession) ఇది call చేసి alert పంపిస్తుంది.
        """
        ep = epoch or self.history.epoch_count

        # Priority order లో check చేస్తుంది — critical first
        checks = [
            self._check_nan,
            self._check_exploding_gradient,
            self._check_overfitting,
            self._check_plateau,
            self._check_underfitting,
        ]

        for check in checks:
            result = check(ep)
            if result.is_problem():
                # Cooldown: same problem ని 5 epochs కి ఒకసారి మాత్రమే alert
                last = self._alert_cooldown.get(result.problem, -999)
                if ep - last >= 5:
                    self._alert_cooldown[result.problem] = ep
                    self._last_diagnosis = result.problem
                    return result

        return DiagnosisResult(
            problem    = ProblemType.HEALTHY,
            confidence = 1.0,
            message    = "Training healthy",
            suggestion = "none",
            severity   = "info",
            epoch      = ep,
        )

    # ------------------------------------------------------------------ #
    #  Individual checks                                                   #
    # ------------------------------------------------------------------ #

    def _check_nan(self, epoch: int) -> DiagnosisResult:
        if not self.history.train_loss:
            return self._healthy(epoch)

        latest = self.history.train_loss[-1]
        if math.isnan(latest) or math.isinf(latest):
            return DiagnosisResult(
                problem    = ProblemType.NAN_LOSS,
                confidence = 1.0,
                message    = f"Loss became NaN/Inf at epoch {epoch}. Training crashed!",
                suggestion = "reduce_lr|add_gradient_clipping|check_data_normalization",
                severity   = "critical",
                epoch      = epoch,
            )
        return self._healthy(epoch)

    def _check_exploding_gradient(self, epoch: int) -> DiagnosisResult:
        tl = self.history.train_loss

        # Loss sudden spike check
        if len(tl) >= 3:
            recent  = tl[-1]
            prev    = tl[-2]
            if prev > 0 and recent > prev * self.SPIKE_MULTIPLIER:
                return DiagnosisResult(
                    problem    = ProblemType.EXPLODING_GRADIENT,
                    confidence = 0.85,
                    message    = f"Loss spiked {recent/prev:.1f}x at epoch {epoch}. Gradients exploding!",
                    suggestion = "add_gradient_clipping|reduce_lr",
                    severity   = "critical",
                    epoch      = epoch,
                )

        # Explicit grad norm check (if tracked)
        if self.history.gradients:
            latest_grad = self.history.gradients[-1]
            if latest_grad > self.GRAD_EXPLODE_THRESH:
                return DiagnosisResult(
                    problem    = ProblemType.EXPLODING_GRADIENT,
                    confidence = 0.95,
                    message    = f"Gradient norm {latest_grad:.1f} >> {self.GRAD_EXPLODE_THRESH} at epoch {epoch}",
                    suggestion = "add_gradient_clipping|reduce_lr",
                    severity   = "critical",
                    epoch      = epoch,
                )

        return self._healthy(epoch)

    def _check_overfitting(self, epoch: int) -> DiagnosisResult:
        tl = self.history.train_loss
        vl = self.history.val_loss

        if len(tl) < self.OVERFIT_MIN_EPOCHS or len(vl) < self.OVERFIT_MIN_EPOCHS:
            return self._healthy(epoch)

        train_loss = tl[-1]
        val_loss   = vl[-1]

        if train_loss <= 0:
            return self._healthy(epoch)

        gap = (val_loss - train_loss) / train_loss

        if gap > self.OVERFIT_GAP:
            confidence = min(0.95, 0.6 + gap)
            return DiagnosisResult(
                problem    = ProblemType.OVERFITTING,
                confidence = confidence,
                message    = (
                    f"Overfitting detected at epoch {epoch}. "
                    f"Val loss ({val_loss:.4f}) >> Train loss ({train_loss:.4f}). "
                    f"Gap: {gap*100:.1f}%"
                ),
                suggestion = "add_dropout|add_l2_regularization|data_augmentation|reduce_model_size",
                severity   = "warning",
                epoch      = epoch,
            )

        return self._healthy(epoch)

    def _check_plateau(self, epoch: int) -> DiagnosisResult:
        tl = self.history.train_loss

        if len(tl) < self.PLATEAU_MIN_EPOCHS:
            return self._healthy(epoch)

        window = tl[-self.PLATEAU_WINDOW:]
        improvement = max(window) - min(window)

        if improvement < self.PLATEAU_MIN_DELTA:
            return DiagnosisResult(
                problem    = ProblemType.PLATEAU,
                confidence = 0.80,
                message    = (
                    f"Training plateaued at epoch {epoch}. "
                    f"Loss improvement only {improvement:.5f} in last {self.PLATEAU_WINDOW} epochs."
                ),
                suggestion = "add_lr_scheduler|change_optimizer|increase_model_complexity|try_different_architecture",
                severity   = "warning",
                epoch      = epoch,
            )

        return self._healthy(epoch)

    def _check_underfitting(self, epoch: int) -> DiagnosisResult:
        tl  = self.history.train_loss
        acc = self.history.train_accuracy

        if len(tl) < self.PLATEAU_MIN_EPOCHS:
            return self._healthy(epoch)

        latest_loss = tl[-1]

        # High loss AND no improvement over last N epochs
        window      = tl[-self.PLATEAU_WINDOW:]
        improvement = max(window) - min(window)

        loss_high    = latest_loss   > self.UNDERFIT_LOSS_THRESH
        acc_low      = acc and acc[-1] < self.UNDERFIT_ACC_THRESH
        not_improving= improvement   < self.PLATEAU_MIN_DELTA * 3

        # If accuracy is available, require that it's still low.
        if loss_high and not_improving and (not acc or acc_low):
            return DiagnosisResult(
                problem    = ProblemType.UNDERFITTING,
                confidence = 0.75,
                message    = (
                    f"Underfitting at epoch {epoch}. "
                    f"Loss still high ({latest_loss:.4f}), model not learning."
                ),
                suggestion = "increase_model_size|add_layers|train_longer|reduce_regularization|check_learning_rate",
                severity   = "warning",
                epoch      = epoch,
            )

        return self._healthy(epoch)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _healthy(self, epoch: int) -> DiagnosisResult:
        return DiagnosisResult(
            problem    = ProblemType.HEALTHY,
            confidence = 1.0,
            message    = "Training healthy",
            suggestion = "none",
            severity   = "info",
            epoch      = epoch,
        )

    def summary(self) -> dict:
        """Run complete అయిన తర్వాత full summary — changes.md కి వాడతారు"""
        tl  = self.history.train_loss
        vl  = self.history.val_loss
        acc = self.history.train_accuracy
        vacc= self.history.val_accuracy

        return {
            "total_epochs":       self.history.epoch_count,
            "final_train_loss":   round(tl[-1],  4) if tl   else None,
            "final_val_loss":     round(vl[-1],  4) if vl   else None,
            "best_train_loss":    round(min(tl),  4) if tl   else None,
            "best_val_loss":      round(min(vl),  4) if vl   else None,
            "final_train_acc":    round(acc[-1],  4) if acc  else None,
            "final_val_acc":      round(vacc[-1], 4) if vacc else None,
            "last_diagnosis":     self._last_diagnosis.value,
        }
