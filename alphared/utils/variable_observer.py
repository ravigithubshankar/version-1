import threading, time

class VariableObserver:
    WATCHLIST = {"loss", "val_loss", "accuracy", "val_accuracy",
                 "train_loss", "train_accuracy", "lr", "epoch"}

    def __init__(self, scope, callback, interval=1.0):
        self.scope     = scope
        self.callback  = callback
        self.interval  = interval
        self.watchlist = {k for k in scope if k in self.WATCHLIST
                          or any(w in k.lower() for w in ["loss","acc","metric"])}
        self._running  = False
        self._thread   = None
        self._prev     = {}

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            metrics = {}
            for k in self.watchlist:
                try:
                    v = self.scope.get(k)
                    if v is not None and v != self._prev.get(k):
                        metrics[k]      = float(v)
                        self._prev[k]   = float(v)
                except Exception:
                    pass
            if metrics:
                self.callback(metrics)
            time.sleep(self.interval)
