# Configuration Parameters

This page describes the configuration parameters used in the YAML file for controlling the training and evaluation of the model.

This is what a standard configuration looks like:
```yaml
# lightning.pytorch==2.1.2
seed_everything: true
trainer:
  accelerator: gpu
  strategy: auto
  devices: auto
  num_nodes: 1
  precision: null
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      name: null
      save_dir: null
      version: null
      offline: false
      dir: null
      id: null
      anonymous: null
      project: project-unsat
      log_model: false
      experiment: null
      prefix: ''
      checkpoint_name: null
      job_type: null
      config: null
      entity: null
      reinit: null
      tags: null
      group: null
      notes: null
      magic: null
      config_exclude_keys: null
      config_include_keys: null
      mode: null
      allow_val_change: null
      resume: null
      force: null
      tensorboard: null
      sync_tensorboard: null
      monitor_gym: null
      save_code: null
      settings: null
  callbacks:
  - unsat.callbacks.ClassWeightsCallback
  - class_path: unsat.callbacks.CheckFaultsCallback
    init_args:
      patch_size: 64
  fast_dev_run: false
  max_epochs: 1000
  min_epochs: null
  max_steps: -1
  min_steps: null
  max_time: null
  limit_train_batches: null
  limit_val_batches: null
  limit_test_batches: null
  limit_predict_batches: null
  overfit_batches: 0.0
  val_check_interval: null
  check_val_every_n_epoch: 1
  num_sanity_val_steps: null
  log_every_n_steps: 1
  enable_checkpointing: null
  enable_progress_bar: null
  enable_model_summary: null
  accumulate_grad_batches: 1
  gradient_clip_val: null
  gradient_clip_algorithm: null
  deterministic: null
  benchmark: null
  inference_mode: true
  use_distributed_sampler: true
  profiler: null
  detect_anomaly: false
  barebones: false
  plugins: null
  sync_batchnorm: false
  reload_dataloaders_every_n_epochs: 0
  default_root_dir: null
model:
  network:
    class_path: unsat.models.UNet
    init_args:
      start_channels: 2
      num_blocks: 3
      kernel_size: 3
      block_depth: 2
      batch_norm: true
  optimizer:
    class_path: torch.optim.Adam
    init_args:
      lr: 3e-3
data:
  hdf5_path: /projects/0/einf3381/UNSAT/data/experimental.h5
  faults_path: faults/faults.yaml
  class_names:
  - "water"
  - "background"
  - "air"
  - "root"
  - "soil"
  input_channels: 1
  train_samples:
  - maize/coarse/loose
  - maize/fine/dense
  height_range:
  - 1000
  - 1100
  train_day_range:
  - 2
  - 3
  validation_split: 0.1
  seed: 42
  batch_size: 4
  num_workers: 2
  dimension: 2
  patch_size: 512
  patch_border: 16
ckpt_path: null
```

# Configuration Parameters

## Explanation of Configuration Parameters

- **seed_everything**: Ensures all random number generators are seeded to enable reproducibility. (true or false)

## Trainer Configuration

- **trainer.accelerator**: Specifies the hardware to use for training (gpu, cpu, etc.).
- **trainer.strategy**: Automatically determines the best distributed training strategy.
- **trainer.devices**: Sets the number of devices to use. `auto` uses all available devices.
- **trainer.num_nodes**: Specifies the number of nodes for distributed training.
- **trainer.precision**: Defines the floating-point precision. `null` uses the default (32-bit).

## Logger Settings

- **trainer.logger.class_path**: Specifies the logger to use. This setup uses the `WandbLogger`.
- **trainer.logger.init_args**: Arguments for initializing the logger.
  - **name**: Name for the run. `null` auto-generates a name.
  - **save_dir**: Directory to save logs.
  - **version**: Version number for the logger.
  - **offline**: Enables offline mode for logging.
  - **dir**: Directory for logs.
  - **id**: ID for resuming a run.
  - **anonymous**: Logs anonymously if set to true.
  - **project**: Project name. Defaults to `project-unsat`.
  - **log_model**: Indicates whether to log model checkpoints.
  - **experiment**: Name or identifier for the experiment.
  - **prefix**: Prefix for run names.
  - **checkpoint_name**: Name of the checkpoint.
  - **job_type**: Specifies the job type (e.g., training, validation).
  - **config**: Logs a configuration dictionary.
  - **entity**: W&B entity or team name.
  - **reinit**: Allows re-initialization if set to true.
  - **tags**: Tags for the run.
  - **group**: Group name for organizing runs.
  - **notes**: Additional notes about the run.
  - **magic**: Magic commands.
  - **config_exclude_keys**: Excludes specific keys from the config logging.
  - **config_include_keys**: Includes specific keys in the config logging.
  - **mode**: Sets the mode for the logger.
  - **allow_val_change**: Allows changing validation configuration if true.
  - **resume**: Resumes a previous run if true.
  - **force**: Forces overwriting an existing run.
  - **tensorboard**: Configures TensorBoard integration.
  - **sync_tensorboard**: Synchronizes TensorBoard with the logger.
  - **monitor_gym**: Monitors the gym environment if true.
  - **save_code**: Saves the code related to the run if true.
  - **settings**: Additional logger settings.

## Callbacks

- **trainer.callbacks**: A list of callbacks used during training.
  - **unsat.callbacks.ClassWeightsCallback**: Adjusts class weights dynamically.
  - **unsat.callbacks.CheckFaultsCallback**: Monitors for faults during training.
    - **init_args.patch_size**: Patch size for fault checking, set to 64.

## Additional Trainer Settings

- **trainer.fast_dev_run**: Runs a single batch for debugging if true.
- **trainer.max_epochs**: Maximum number of epochs, set to 1000.
- **trainer.min_epochs**: Minimum number of epochs. `null` means no minimum.
- **trainer.max_steps**: Maximum training steps, set to -1 to disable.
- **trainer.min_steps**: Minimum number of steps. `null` means no minimum.
- **trainer.max_time**: Limits the maximum training time.
- **trainer.limit_train_batches**: Limits the number of training batches per epoch.
- **trainer.limit_val_batches**: Limits the number of validation batches per epoch.
- **trainer.limit_test_batches**: Limits the number of test batches.
- **trainer.limit_predict_batches**: Limits the number of prediction batches.
- **trainer.overfit_batches**: Fraction of data to overfit for debugging, set to 0.0.
- **trainer.val_check_interval**: How often to validate, in terms of training epochs.
- **trainer.check_val_every_n_epoch**: How often to perform validation checks, in epochs.
- **trainer.num_sanity_val_steps**: Number of steps for sanity check validation.
- **trainer.log_every_n_steps**: Frequency of logging, set to 1 for every step.
- **trainer.enable_checkpointing**: Enables checkpointing.
- **trainer.enable_progress_bar**: Shows a progress bar if true.
- **trainer.enable_model_summary**: Displays a summary of the model if true.
- **trainer.accumulate_grad_batches**: Number of batches over which gradients are accumulated, set to 1.
- **trainer.gradient_clip_val**: Clipping value for gradients.
- **trainer.gradient_clip_algorithm**: Algorithm used for gradient clipping.
- **trainer.deterministic**: Ensures deterministic training if true.
- **trainer.benchmark**: Enables benchmarking for better performance.
- **trainer.inference_mode**: Enables inference mode, optimizing evaluation.
- **trainer.use_distributed_sampler**: Uses a distributed sampler for data loading.
- **trainer.profiler**: Profiler for performance analysis.
- **trainer.detect_anomaly**: Enables anomaly detection if true.
- **trainer.barebones**: If true, runs with minimal features.
- **trainer.plugins**: Specifies additional plugins to use.
- **trainer.sync_batchnorm**: Synchronizes batch normalization across devices if true.
- **trainer.reload_dataloaders_every_n_epochs**: Reloads data loaders after a specified number of epochs.
- **trainer.default_root_dir**: Root directory for saving logs and checkpoints.

## Model Configuration

- **model.network.class_path**: Path to the model class, set to `unsat.models.UNet`.
- **model.network.init_args**: Initialization arguments for the model.
  - **start_channels**: Number of starting channels, set to 2.
  - **num_blocks**: Number of blocks in the model, set to 3.
  - **kernel_size**: Size of the convolution kernels, set to 3.
  - **block_depth**: Depth of each block, set to 2.
  - **batch_norm**: Enables batch normalization if true.

## Optimizer Configuration

- **model.optimizer.class_path**: Path to the optimizer class, set to `torch.optim.Adam`.
- **model.optimizer.init_args**: Initialization arguments for the optimizer.
  - **lr**: Learning rate, set to 3e-3.

## Data Configuration

- **data.hdf5_path**: Path to the HDF5 file containing the dataset.
- **data.faults_path**: Path to the YAML file specifying faults.
- **data.class_names**: List of class names for classification:
  - water, background, air, root, soil
- **data.input_channels**: Number of input channels, set to 1.
- **data.train_samples**: List of paths to training samples:
  - maize/coarse/loose, maize/fine/dense
- **data.height_range**: Range of heights to consider, from 1000 to 1100.
- **data.train_day_range**: Days to include in the training set, from 2 to 3.
- **data.validation_split**: Fraction of the data to use for validation, set to 0.1.
- **data.seed**: Random seed for data shuffling, set to 42.
- **data.batch_size**: Batch size, set to 4.
- **data.num_workers**: Number of workers for data loading, set to 2.
- **data.dimension**: Dimensionality of the data, set to 2.
- **data.patch_size**: Patch size for data extraction, set to 512.
- **data.patch_border**: Border size around each patch, set to 16.

## Checkpoint Path

- **ckpt_path**: Path to a checkpoint for resuming training, set to `null` to start fresh.