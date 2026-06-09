import torch
import pytorch_lightning as pl


class GPUMemoryMonitor(pl.Callback):
    def __init__(self, device_idx: int = None):
        super().__init__()
        self.device_idx = device_idx

    def _get_device_idx(self, pl_module):
        if isinstance(pl_module.device, torch.device) and pl_module.device.type == "cuda":
            if pl_module.device.index is not None:
                return pl_module.device.index

        return torch.cuda.current_device()

    def on_train_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            idx = self.device_idx
            if idx is None:
                idx = self._get_device_idx(pl_module)
            self.device_idx = idx

            torch.cuda.set_device(idx)
            torch.cuda.reset_peak_memory_stats(idx)
            print(f"[GPUMemoryMonitor] Reset peak memory stats on cuda:{idx}")

    def on_train_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            idx = self.device_idx
            if idx is None:
                idx = self._get_device_idx(pl_module)

            torch.cuda.set_device(idx)
            max_alloc = torch.cuda.max_memory_allocated(idx) / 1024 ** 2
            max_reserved = torch.cuda.max_memory_reserved(idx) / 1024 ** 2

            print(f"[GPUMemoryMonitor] Max allocated: {max_alloc:.2f} MB (cuda:{idx})")
            print(f"[GPUMemoryMonitor] Max reserved:  {max_reserved:.2f} MB (cuda:{idx})")

            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {
                        "MaxGPU/allocated_MB": max_alloc,
                        "MaxGPU/reserved_MB": max_reserved,
                    },
                    step=trainer.global_step,
                )
