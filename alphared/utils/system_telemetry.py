import psutil

def get_system_metrics():
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
    }
    try:
        import torch
        if torch.cuda.is_available():
            metrics["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024**2
            metrics["gpu_percent"]   = torch.cuda.utilization()
    except Exception:
        pass
    return metrics
