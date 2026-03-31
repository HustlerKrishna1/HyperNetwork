"""
hypernetwork  —  On-the-fly weight generation via hypernetworks.
"""

from .config import (
    ExperimentConfig,
    HypernetworkConfig,
    TargetModelConfig,
    TrainingConfig,
    WEIGHT_TYPE_INDEX,
)
from .hypernetwork import (
    Hypernetwork,
    SharedHypernetwork,
    build_hypernetwork,
    count_parameters,
)
from .target_model import TransformerLM, build_target_model
from .losses import (
    HypernetworkLoss,
    task_loss,
    weight_reconstruction_loss,
    distillation_loss,
    compute_perplexity,
)
from .trainer import (
    TeacherTrainer,
    HypernetworkTrainer,
    run_full_pipeline,
)
from .optimizer_utils import (
    parameter_budget_breakdown,
    size_budget_check,
    quantize_to_fp16,
    quantize_to_int8,
    WeightCache,
    generate_all_weights_batched,
    LoRAHypernetwork,
    TokenConditionedHypernetwork,
    MultiTaskHypernetwork,
)
from .experiments import (
    quick_eval_untrained,
    ablation_rank_sweep,
    run_experiment_suite,
    monitor_health,
)

__all__ = [
    # Config
    "ExperimentConfig", "HypernetworkConfig", "TargetModelConfig",
    "TrainingConfig", "WEIGHT_TYPE_INDEX",
    # Models
    "Hypernetwork", "SharedHypernetwork", "build_hypernetwork",
    "TransformerLM", "build_target_model",
    # Losses
    "HypernetworkLoss", "task_loss", "weight_reconstruction_loss",
    "distillation_loss", "compute_perplexity",
    # Training
    "TeacherTrainer", "HypernetworkTrainer", "run_full_pipeline",
    # Utilities
    "parameter_budget_breakdown", "size_budget_check",
    "quantize_to_fp16", "quantize_to_int8",
    "WeightCache", "generate_all_weights_batched",
    "LoRAHypernetwork", "TokenConditionedHypernetwork", "MultiTaskHypernetwork",
    # Experiments
    "quick_eval_untrained", "ablation_rank_sweep",
    "run_experiment_suite", "monitor_health",
    "count_parameters",
]
