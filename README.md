# Human Pose Estimation (HPE) Pretraining System

A production-level ML system for unsupervised pretraining of Human Pose Estimation models, designed to predict 3D human pose from 2D keypoints.

## Overview

This project implements a transformer-based model architecture for learning the mapping between 2D pose keypoints and 3D pose representations in an unsupervised manner. The system is designed with a focus on modularity, configurability, and production-ready practices.

## System Requirements

- Python 3.8+
- PyTorch 1.12+ (required for Apple Silicon MPS support)
- CUDA-compatible GPU (optional for faster training)
- Apple Silicon M-series chip (optional, supported via MPS backend)

## Code Architecture

```
pretrain/                          # Project root directory
├── config/                        # Configuration files
│   ├── config.yaml                # Main configuration
│   ├── models/                    # Model-specific configs
│   │   ├── pretrain_model1.yaml   # Transformer model config
│   │   └── pretrain_vit.yaml      # Vision Transformer model config
│   └── data/                      # Dataset configs
│       └── h36m.yaml              # Human3.6M dataset config
│
├── src/                           # Source code
│   ├── data/                      # Data pipeline
│   │   ├── datasets/              # Dataset implementations
│   │   │   └── h36m_dataset.py    # Human3.6M dataset
│   │   ├── transforms.py          # Data transformations
│   │   └── dataloader.py          # DataLoader factories
│   │
│   ├── models/                    # ML model architecture
│   │   ├── components/            # Reusable model components
│   │   │   ├── attention.py       # Attention mechanisms
│   │   │   ├── encoders.py        # Encoder/decoder modules
│   │   │   └── vision_transformer.py # Vision Transformer components
│   │   ├── pretrain_model.py      # Main pretraining model
│   │   ├── pretrain_vit_model.py  # Vision Transformer-based model
│   │   └── loss.py                # Loss functions
│   │
│   ├── trainers/                  # Training logic
│   │   └── trainer.py             # Trainer implementation
│   │
│   ├── utils/                     # Utilities
│   │   ├── device.py              # Device detection and configuration
│   │   ├── logger.py              # Logging utilities
│   │   └── visualization.py       # Visualization tools
│   │
│   └── scripts/                   # Execution scripts
│       ├── train.py               # Training script
│       ├── inference.py           # Inference script
│       └── evaluate.py            # Evaluation script
│
├── outputs/                       # Output directory
│   ├── models/                    # Saved models
│   └── logs/                      # Training logs
```

## System Architecture

### Data Layer
- **Human36MDataset**: Flexible dataset implementation supporting both 2D and 3D keypoints
- **Data Transforms**: Comprehensive augmentation pipeline (normalization, flipping, rotation, scaling)
- **Dataloader Factory**: Configuration-driven dataloader creation with path resolution

### Model Layer
- **Encoder**: Transformer-based architecture converting 2D keypoints to latent representations
- **Decoder**: Transforms latent representations to 3D poses
- **Attention Mechanisms**: Multi-head attention with joint relation modeling
- **Vision Transformer**: Specialized ViT architecture for keypoint-to-keypoint mapping (2D-2D, 3D-3D, 2D-3D)
- **Loss Functions**: Modular losses including reconstruction, limb consistency, and temporal smoothness

### Training Layer
- **Trainer**: Handles training loop, validation, checkpointing, and early stopping
- **Optimizer & Scheduler**: Multiple optimization strategies and learning rate schedules
- **Mixed Precision**: Support for faster training with mixed precision

### Hardware Acceleration
- **Multi-architecture Support**: Automatic device detection for CUDA, MPS (Apple Silicon), or CPU
- **Apple Silicon Support**: Optimized performance on M-series chips using Metal Performance Shaders (MPS)
- **Graceful Fallbacks**: Automatically disables unsupported features (like mixed precision on MPS)

### Inference Layer
- **Model Loading**: Utilities for loading trained models
- **Batch Processing**: Efficient batch inference
- **Visualization**: 2D/3D pose visualization tools

### Configuration System
- **Hierarchical Config**: YAML-based configuration with inheritance
- **Command-line Interface**: Override configurations via command line arguments

## Key Features

1. **Modular Design**: Components are decoupled for easy extension and modification
2. **Production-Ready**: Error handling, logging, and performance monitoring
3. **Configurable**: Extensive configuration options without code changes
4. **Efficient Data Pipeline**: Caching, parallel loading, and precomputed statistics
5. **Visualization Tools**: Comprehensive visualization for model outputs
6. **Multiple Model Architectures**: Support for both standard Transformer and Vision Transformer models
7. **Cross-Platform Hardware Support**: Runs on NVIDIA GPUs and Apple Silicon M-series chips
8. **Portable Import Structure**: Relative imports ensure code works regardless of project location

## Import Structure

The project uses a modular import structure that makes it portable and easy to understand:

- **Relative Imports**: All internal imports use relative paths (e.g., `from .datasets import Human36MDataset`)
- **Module Organization**: Components are organized into logical modules that can be imported independently
- **Path Independence**: The codebase automatically adapts to its location with no hardcoded paths

## Progress Tracking

The system provides rich visual feedback during training:

- **Colorful Console Output**: Uses ANSI colors for better readability
- **Progress Bars**: Visual indicators for epoch and batch progress
- **Training Statistics**: Real-time loss values and metrics
- **Device Information**: Automatic detection and display of available hardware

## Workflow Guide

This section provides a comprehensive guide on how to run training, perform inference, evaluate models, and make changes to datasets and model architectures.

### Training a Model

1. **Basic Training:**
   ```bash
   python src/scripts/train.py --config config/config.yaml
   ```

2. **Train with custom experiment name:**
   ```bash
   python src/scripts/train.py --config config/config.yaml --experiment my_experiment
   ```

3. **Resume training from checkpoint:**
   ```bash
   python src/scripts/train.py --config config/config.yaml --resume outputs/models/my_experiment/checkpoint_epoch_50.pt
   ```

4. **Training with debug mode:**
   ```bash
   python src/scripts/train.py --config config/config.yaml --debug
   ```

5. **Using a specific GPU:**
   ```bash
   python src/scripts/train.py --config config/config.yaml --gpu 1
   ```

6. **Training with Vision Transformer model:**
   ```bash
   python src/scripts/train.py --config config/pretrain_vit.yaml --model_type vit
   ```

### Running Inference

1. **Inference on a single input file:**
   ```bash
   python src/scripts/inference.py --checkpoint outputs/models/my_experiment/best_model.pt --input path/to/keypoints.json --output outputs/results
   ```

2. **Inference on a directory of inputs:**
   ```bash
   python src/scripts/inference.py --checkpoint outputs/models/my_experiment/best_model.pt --input path/to/keypoints_dir --output outputs/results
   ```

3. **Visualization during inference:**
   ```bash
   python src/scripts/inference.py --checkpoint outputs/models/my_experiment/best_model.pt --input path/to/keypoints.json --visualize
   ```

4. **Batch processing test dataset:**
   ```bash
   python src/scripts/inference.py --checkpoint outputs/models/my_experiment/best_model.pt --batch-size 32 --test-data
   ```

### Model Evaluation

1. **Basic evaluation:**
   ```bash
   python src/scripts/evaluate.py --checkpoint outputs/models/my_experiment/best_model.pt --config config/config.yaml
   ```

2. **Evaluation with Procrustes alignment:**
   ```bash
   python src/scripts/evaluate.py --checkpoint outputs/models/my_experiment/best_model.pt --config config/config.yaml --procrustes
   ```

3. **Evaluation with per-joint metrics:**
   ```bash
   python src/scripts/evaluate.py --checkpoint outputs/models/my_experiment/best_model.pt --config config/config.yaml --per-joint
   ```

4. **Save visualizations during evaluation:**
   ```bash
   python src/scripts/evaluate.py --checkpoint outputs/models/my_experiment/best_model.pt --config config/config.yaml --visualize --output-dir outputs/visualizations
   ```

### Changing the Dataset

1. **Create a new dataset configuration file:**
   - Create a new configuration file in `config/data/`, e.g., `mpi_inf_3dhp.yaml`
   - Define dataset-specific parameters such as paths, preprocessing, and augmentation settings

2. **Implement dataset class (if needed):**
   - Create a new dataset class in `src/data/datasets/`, e.g., `mpi_dataset.py`
   - Add it to the module imports in `src/data/datasets/__init__.py`

3. **Update the main configuration:**
   ```yaml
   # In config/config.yaml
   data:
     dataset: "mpi_inf_3dhp"  # Change dataset name
     config_file: "config/data/mpi_inf_3dhp.yaml"  # Update config path
     # Update other dataset-specific parameters
   ```

4. **Run training with new dataset:**
   ```bash
   python src/scripts/train.py --config config/config.yaml
   ```

### Changing the Model Architecture

1. **Create a new model configuration file:**
   - Create a new configuration file in `config/models/`, e.g., `pretrain_model2.yaml`
   - Define model-specific parameters such as architecture type, layers, and dimensions

2. **Implement model components (if needed):**
   - Add new components in `src/models/components/`
   - Extend the base model in `src/models/pretrain_model.py` or create a new model class

3. **Update the main configuration:**
   ```yaml
   # In config/config.yaml
   model:
     name: "pretrain_vit"  # Change model name to use Vision Transformer
     config_file: "config/models/pretrain_vit.yaml"  # Update config path
     # Update other model-specific parameters
   ```

4. **Run training with new model:**
   ```bash
   python src/scripts/train.py --config config/config.yaml
   ```

### Hyperparameter Tuning

1. **Create a modified configuration:**
   ```yaml
   # In config/tuning/exp1.yaml
   model:
     hidden_dims: [2048, 1024, 512]
     dropout: 0.2
   
   training:
     learning_rate: 0.0005
     batch_size: 128
   ```

2. **Run training with modified config:**
   ```bash
   python src/scripts/train.py --config config/config.yaml --experiment tuning_exp1 --override config/tuning/exp1.yaml
   ```

### Monitoring and Visualization

1. **View TensorBoard logs:**
   ```bash
   tensorboard --logdir outputs/logs
   ```

2. **Generate evaluation visualizations:**
   ```bash
   python src/scripts/evaluate.py --checkpoint outputs/models/my_experiment/best_model.pt --visualize --output-dir outputs/visualizations
   ```

3. **Compare multiple models:**
   ```bash
   python src/scripts/compare_models.py --checkpoints outputs/models/exp1/best_model.pt outputs/models/exp2/best_model.pt --output-dir outputs/comparisons
   ```

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/hpe-pretrain.git
   cd hpe-pretrain
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run a test training:**
   ```bash
   python src/scripts/train.py --config config/config.yaml --debug
   ```

4. **View device information:**
   ```bash
   python -c "from src.utils.device import print_device_info; print_device_info()"
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 