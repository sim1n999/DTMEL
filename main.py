import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import time

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from codes.utils.functions import setup_parser
from codes.model.lightning_dual_tower import LightningDualTower
from codes.utils.dataset import DataModuleForDualTower
from codes.utils.gpu_monitor import GPUMemoryMonitor


def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h} h {m} m {s:.2f} s"


if __name__ == '__main__':
    start = time.perf_counter()

    args = setup_parser()
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(1)

    data_module = DataModuleForDualTower(args)
    lightning_model = LightningDualTower(args)

    logger = pl.loggers.CSVLogger("./runs", name=args.run_name, flush_logs_every_n_steps=30)

    ckpt_callbacks = ModelCheckpoint(
        monitor='Val/mrr',
        save_weights_only=True,
        mode='max',
        filename='best-{epoch:02d}-{Val/mrr:.4f}'
    )
    early_stop_callback = EarlyStopping(
        monitor="Val/mrr",
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode="max"
    )

    gpu_monitor = GPUMemoryMonitor()
    trainer = pl.Trainer(
        **args.trainer,
        deterministic=True,
        logger=logger,
        default_root_dir="./runs",
        callbacks=[ckpt_callbacks, early_stop_callback, gpu_monitor]
    )

    trainer.fit(lightning_model, datamodule=data_module)
    trainer.test(lightning_model, datamodule=data_module, ckpt_path='best')

    end = time.perf_counter()
    duration = end - start
    print(f"\n{'='*50}")
    print(f"训练完成! 总耗时: {format_duration(duration)}")
    print(f"{'='*50}\n")
