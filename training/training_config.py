"""
training/training_config.py

Comprehensive configuration dataclass and hardware-aware defaults
for the CODEscribe CodeT5 supervised fine-tuning pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import platform
from typing import Any, Dict, Optional

logger = logging.getLogger("CODEscribe.TrainingConfig")


@dataclass
class TrainingExperimentConfig:
    """Experiment configuration for fine-tuning Salesforce/codet5-small."""

    # Model and Tokenizer specifications
    model_name_or_path: str = "Salesforce/codet5-small"
    tokenizer_name_or_path: str = "Salesforce/codet5-small"
    task_prefix: str = "document python code: "

    # Sequence length boundaries
    max_source_length: int = 512
    max_target_length: int = 128

    # Optimization and Schedule
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 32
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Checkpoint and Evaluation Strategy
    evaluation_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 2
    logging_steps: int = 50
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    max_steps: int = -1
    max_eval_samples: Optional[int] = None

    # Hardware & Precision settings
    seed: int = 42
    fp16: bool = False       # Automatically set via detect_hardware()
    bf16: bool = False
    dataloader_num_workers: int = 0  # 0 on Windows avoids multiprocessing IPC issues

    # Paths (relative to workspace root)
    workspace_root: Path = field(default_factory=lambda: Path("."))
    train_data_path: Path = field(default_factory=lambda: Path("data/processed/train.csv"))
    val_data_path: Path = field(default_factory=lambda: Path("data/processed/validation.csv"))
    test_data_path: Path = field(default_factory=lambda: Path("data/processed/test.csv"))
    output_dir: Path = field(default_factory=lambda: Path("model/checkpoints"))
    final_model_dir: Path = field(default_factory=lambda: Path("model/codescribe"))
    logs_dir: Path = field(default_factory=lambda: Path("outputs/training"))
    audit_report_path: Path = field(default_factory=lambda: Path("outputs/training_data_audit.json"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v).replace("\\", "/")
        return d

    def save(self, destination_path: Path) -> None:
        """Persist configuration as JSON."""
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with open(destination_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved training configuration to: %s", destination_path.resolve())

    @classmethod
    def load(cls, source_path: Path) -> TrainingExperimentConfig:
        """Load configuration from JSON."""
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Convert path strings back to Path
        path_fields = [
            "workspace_root", "train_data_path", "val_data_path", "test_data_path",
            "output_dir", "final_model_dir", "logs_dir", "audit_report_path"
        ]
        for pf in path_fields:
            if pf in data:
                data[pf] = Path(data[pf])
        return cls(**data)

    @classmethod
    def create_pilot_config(cls, workspace_root: Path = Path(".")) -> TrainingExperimentConfig:
        """
        Create a bounded pilot configuration for RTX 4050 GPU validation.
        50 total optimization steps with per-device batch size 4 and grad accum 8
        (effective batch size = 32), saving checkpoints at steps 25 and 50.
        """
        return cls(
            workspace_root=workspace_root,
            max_steps=50,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            per_device_eval_batch_size=4,
            eval_steps=25,
            save_steps=25,
            logging_steps=10,
            warmup_steps=10,
            output_dir=Path("outputs/pilot_run_accelerate16/checkpoints"),
            final_model_dir=Path("outputs/pilot_run_accelerate16/model"),
            logs_dir=Path("outputs/pilot_run_accelerate16/logs"),
            max_eval_samples=64,
            fp16=True,
        )


def detect_hardware() -> Dict[str, Any]:
    """
    Detect physical compute hardware (NVIDIA GPU, VRAM, CPU)
    and return recommended training configuration settings.
    """
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cuda_available": False,
        "device_name": "CPU",
        "device_count": 0,
        "total_vram_gb": 0.0,
        "recommended_fp16": False,
        "recommended_batch_size": 4,
        "recommended_grad_accum": 8,
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            info["total_vram_gb"] = round(vram_bytes / (1024 ** 3), 2)
            info["recommended_fp16"] = True
            if info["total_vram_gb"] >= 6.0:
                info["recommended_batch_size"] = 8
                info["recommended_grad_accum"] = 4
            else:
                info["recommended_batch_size"] = 4
                info["recommended_grad_accum"] = 8
        else:
            info["torch_installed_cpu_only"] = True
    except ImportError:
        info["torch_installed"] = False
        info["note"] = "PyTorch not yet installed in active environment."

    return info
