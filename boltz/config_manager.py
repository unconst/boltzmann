# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# TODO: @formalised consider using hyper
import argparse
import sys
import os
from typing import Any, Dict, List, Union
import json

from omegaconf import OmegaConf, DictConfig
import bittensor as bt
import torch

from boltz.common import logger

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

def string_list(raw_arg):
    return raw_arg.split(",")

class JobConfig:
    """
    A helper class to manage the training configuration using OmegaConf.

    This class loads configurations from a TOML/YAML config file and merges them
    with command-line arguments. It also integrates Bittensor configurations.
    ## Notes

    - Ensure that the configuration file is properly formatted and compatible with OmegaConf.
    - Handles potential errors when parsing the configuration file.
    - Use the `--job.config_file` argument to specify the config file:

      ```bash
      python miner.py --job.config_file=train_configs/llama2_13b.toml
      ```
    """

    def __init__(self):
        # Initialize argument parser
        self.parser = argparse.ArgumentParser(description="Boltzmann argument parser.")

        # Add command-line arguments
        self.add_arguments()

        # Parse command-line arguments
        self.args = self.parse_arguments()

        # Load configurations using OmegaConf
        self.config = self.load_config()

        # Validate the final configuration
        self.validate_config()

    def add_arguments(self):
        """
        Adds command-line arguments to the parser.
        """

        self.parser.add_argument(
            "--job_config_file",
            type=str,
            default=None,
            help="Job config file",
        )

        job_group = self.parser.add_argument_group("job")
        job_group.add_argument(
            "--job_dump_folder",
            type=str,
            default="./boltzman/outputs",
            help="Folder to dump job outputs",
        )
        job_group.add_argument(
            "--job_description",
            type=str,
            default="default job",
            help="Description of the job",
        )
        job_group.add_argument(
            "--job_use_for_integration_test",
            default=False,
            action="store_true",
            help="Add this config to the integration test suite",
        )

        # profiling configs
        profiling_group = self.parser.add_argument_group("profiling")
        profiling_group.add_argument(
            "--enable_profiling",
            action="store_true",
            help="Whether to enable pytorch profiler",
        )
        profiling_group.add_argument(
            "--save_traces_folder",
            type=str,
            default="profile_traces",
            help="Trace files location",
        )
        profiling_group.add_argument(
            "--profile_freq",
            type=int,
            default=10,
            help="How often to collect profiler traces, in iterations",
        )
        profiling_group.add_argument(
            "--enable_memory_snapshot",
            action="store_true",
            default=False,
            help="Whether to dump memory snapshot",
        )
        profiling_group.add_argument(
            "--save_memory_snapshot_folder",
            type=str,
            default="memory_snapshot",
            help="Memory snapshot files location",
        )

        # metrics configs
        metrics_group = self.parser.add_argument_group("metrics")
        metrics_group.add_argument(
            "--log_freq",
            type=int,
            default=10,
            help="How often to log metrics to TensorBoard, in iterations",
        )
        metrics_group.add_argument(
            "--enable_color_printing",
            default=False,
            action="store_true",
            help="Whether to enable color printing",
        )
        metrics_group.add_argument(
            "--enable_tensorboard",
            action="store_true",
            help="Whether to log metrics to TensorBoard",
        )
        metrics_group.add_argument(
            "--save_tb_folder",
            type=str,
            default="tb",
            help="Folder to dump TensorBoard states",
        )
        metrics_group.add_argument(
            "--rank_0_only",
            default=True,
            action="store_true",
            help="""
                Whether to save TensorBoard metrics only for rank 0 or for all ranks.
                When pipeline_parallel_degree is > 1, this option uses the 0th rank of the last stage pipeline group,
                which is the only stage that computes loss metrics.
            """,
        )

        # Model configs
        model_group = self.parser.add_argument_group("model")
        model_group.add_argument(
            "--model.name",
            type=str,
            default="llama",
            help="Which model to train",
        )
        model_group.add_argument(
            "--model.flavor",
            type=str,
            default="debugmodel",
            help="Which model config to train",
        )
        model_group.add_argument(
            "--model.norm_type",
            type=str,
            default="rmsnorm",
            help="Type of layer normalization to use [layernorm, np_layernorm, rmsnorm, fused_rmsnorm]",
        )
        model_group.add_argument(
            "--model.tokenizer_path",
            type=str,
            default="./torchtitan/datasets/tokenizer/tokenizer.model",
            help="Tokenizer path",
        )

        # Optimizer configs
        optimizer_group = self.parser.add_argument_group("optimizer")
        optimizer_group.add_argument(
            "--optimizer.name", 
            type=str, 
            default="AdamW", 
            help="Optimizer to use"
        )
        optimizer_group.add_argument(
            "--optimizer.lr", 
            type=float, 
            default=8e-4, 
            help="Learning rate to use"
        )
        optimizer_group.add_argument(
            "--optimizer.fused",
            default=False,
            action="store_true",
            help="Whether the fused implementation(CUDA only) is used.",
        )

        # Training configs
        training_group = self.parser.add_argument_group("training")
        training_group.add_argument(
            "--training.dataset", 
            type=str, 
            default="c4_mini", 
            help="Dataset to use"
        )
        training_group.add_argument(
            "--training.dataset_path",
            type=str,
            help="""
                Path to the dataset in the file system. If provided, data will be
                loaded from this path instead of downloaded.""",
        )
        training_group.add_argument(
            "--training.batch_size", 
            type=int, 
            default=8, 
            help="Batch size"
        )
        training_group.add_argument(
            "--training.seq_len", 
            type=int, 
            default=2048, 
            help="Sequence length"
        )
        training_group.add_argument(
            "--training.warmup_steps",
            type=int,
            default=200,
            help="Steps for lr scheduler warmup, normally 1/5 of --training.steps",
        )
        training_group.add_argument(
            "--training.max_norm",
            type=Union[float, int],
            default=1.0,
            help="Max norm for gradient clipping",
        )
        training_group.add_argument(
            "--training.steps",
            type=int,
            default=10000,
            help="How many train steps to run",
        )
        training_group.add_argument(
            "--training.data_parallel_replicate_degree",
            type=int,
            default=1,
            help="""
            The `data_parallel_replicate_degree` argument specifies the degree of
            data parallelism for weight replication. When this value is greater
            than 1, weights will be replicated across `data_parallel_replicate_degree`
            ranks. If `data_parallel_shard_degree` is also greater than 1, the parallelism
            method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
            parallelism method used is DDP (Distributed Data Parallelism).
            1 means disabled.""",
        )
        training_group.add_argument(
            "--training.data_parallel_shard_degree",
            type=int,
            default=-1,
            help="""
            The `data_parallel_shard_degree` argument specifies the degree of data
            parallelism for weight sharding. When this value is greater than 1, weights
            will be sharded across `data_parallel_shard_degree` ranks. If
            `data_parallel_replicate_degree` is also greater than 1, the parallelism
            method used is HSDP (Hybrid Sharded Data Parallelism).  Otherwise, the
            parallelism method used is FSDP (Fully Sharded Data Parallelism).

            -1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that
            only one of `data_parallel_replicate_degree` and `data_parallel_shard_degree`
            can be negative.
            1 means disabled.""",
        )
        training_group.add_argument(
            "--training.tensor_parallel_degree",
            type=int,
            default=1,
            help="Tensor Parallelism degree. 1 means disabled.",
        )
        training_group.add_argument(
            "--training.enable_loss_parallel",
            default=True,
            action="store_true",
            help="Whether to apply loss parallel when sequence parallel is enabled",
        )
        training_group.add_argument(
            "--training.mixed_precision_param",
            type=str,
            default="bfloat16",
            choices=["bfloat16", "float32"],
            help="""
                torch dtype to use for parameters when applying mixed precision via FSDP.
                This feature only takes effect when data_parallel_degree > 1
            """,
        )
        training_group.add_argument(
            "--training.mixed_precision_reduce",
            type=str,
            default="float32",
            choices=["float32"],
            help="""
                torch dtype to use for reductions when applying mixed precision via FSDP.
                This feature only takes effect when data_parallel_degree > 1
            """,
        )
        training_group.add_argument(
            "--training.compile",
            action="store_true",
            help="Whether to compile the model",
        )
        training_group.add_argument(
            "--training.gc_freq",
            type=int,
            default=50,
            help="Python garbage control scheduling interval, in steps",
        )
        training_group.add_argument(
            "--training.seed",
            type=int,
            default=None,
            help="Implement reproducibility by setting a Python, PyTorch and CUDA seed",
        )

        # Experimental configs
        experimental_group = self.parser.add_argument_group("experimental")
        experimental_group.add_argument(
            "--experimental.enable_async_tensor_parallel",
            default=False,
            action="store_true",
            help="Whether to apply async tensor parallel (currently only effective when compile is enabled)",
        )
        experimental_group.add_argument(
            "--experimental.pipeline_parallel_degree",
            type=int,
            default=1,
            help="""
                Pipeline Parallelism degree, or number of ranks. 1 means disabled.
                If using looped schedules, this still specifies the number of physical ranks, not the number
                of stages.  Stages per rank are inferred from split points degree, and schedule.""",
        )
        experimental_group.add_argument(
            "--experimental.pipeline_parallel_split_points",
            type=string_list,
            nargs="+",
            default=[],
            help="""
                Specify comma-separated names of modules to use as the beginning of a split point.

                e.g. "layers.0,layers.2" will cause the model to be split into 3 stages,
                the first containing all the layers up to layers.0,
                the second containing layers.0 and up to layers.2,
                the third containing layers.2 and all the remaining layers.

                Note: fully-automated splitting may be enabled in the future,
                but currently the split points must be specified manually.""",
        )
        experimental_group.add_argument(
            "--experimental.pipeline_parallel_schedule",
            type=str,
            default="1F1B",
            help="""
                Specify the Pipeline Parallel schedule to use. The supported schedules are:
                https://github.com/pytorch/pytorch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/torch/distributed/pipelining/schedules.py#L2161.
                The schedule must be compatible with the split points and stages_per_rank.

                Looped schedules (e.g. Interleaved1F1B) require specifying pipeline_parallel_degree = number of ranks,
                and split_points = number of stages - 1""",
        )
        experimental_group.add_argument(
            "--experimental.pipeline_parallel_microbatches",
            type=int,
            default=None,
            help="""
                How many microbatches to split the global training batch into when using pipeline parallelism.

                The global training batch size must be evenly divisible by the number of microbatches.

                The default value will be the number of pipeline stages, if unspecified.
            """,
        )
        experimental_group.add_argument(
            "--experimental.enable_compiled_autograd",
            action="store_true",
            help="Enable CompiledAutograd to compile the backward.",
        )
        # Checkpoint configs
        checkpoint_group = self.parser.add_argument_group("checkpoint")
        checkpoint_group.add_argument(
            "--checkpoint.enable_checkpoint",
            action="store_true",
            help="Whether to enable checkpoint",
        )
        checkpoint_group.add_argument(
            "--checkpoint.folder",
            type=str,
            default="checkpoint",
            help="""
                The folder to store the checkpoints.
                When enable_checkpoint is set to true, checkpoints will be in {--job.dump_folder}/{--checkpoint.folder}.
            """,
        )
        checkpoint_group.add_argument(
            "--checkpoint.interval_type",
            type=str,
            default="steps",
            help="Checkpointing interval unit of measurement ['step', 'seconds']",
        )
        checkpoint_group.add_argument(
            "--checkpoint.interval",
            type=int,
            default=500,
            help="Checkpointing interval, in steps or seconds depending on --checkpoint.interval_type",
        )
        checkpoint_group.add_argument(
            "--checkpoint.model_weights_only",
            action="store_true",
            help="""
                When model_weights_only=True, only model weights will be saved at the end of training.
                With this, checkpoints can be loaded using `torch.load(..., weights_only=True)` after conversion.
                When model_weights_only=False, the full checkpoint will be saved.
                A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
                The default value is false.
            """,
        )
        checkpoint_group.add_argument(
            "--checkpoint.export_dtype",
            type=str,
            default="float32",
            choices=["float16", "bfloat16", "float32"],
            help="""
                Converts to the specified precision when training completes and model_weights_only=true.
                Currently supports float32, float16, and bfloat16.
                The default value is float32.
            """,
        )
        checkpoint_group.add_argument(
            "--checkpoint.create_seed_checkpoint",
            action="store_true",
            help="""
                Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
                Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.
                Could be implemented as a separate script, but this way shares more code.
            """,
        )
        checkpoint_group.add_argument(
            "--checkpoint.async_mode",
            type=str,
            default="disabled",
            help="""
                Which async checkpoint mode to use. Currently there are 3 different modes.
                1. "disabled": synchronized checkpointing will be used.
                2. "async": torch.distributed.checkpoint.async_save will be used.
                3. "async_with_pinned_mem": this option utilizes a dedicated pinned memory
                   space and creates a separate process for faster GPU->CPU transfer
                   performance and eliminating GIL contention. The cost is increased CPU
                   memory usage. If insufficient CPU memory is available, performance may
                   degrade due to memory paging. For most users, "async" should suffice as
                   the performance overhead is typically small (on the order of tens of
                   seconds) compared to checkpointing frequency. This mode can be employed
                   to pursue near-zero checkpointing times (e.g., < 1 second) given
                   appropriate hardware support such as ample CPU memory and fast PCIe.

                "disabled" is the default mode.
            """,
        )
        checkpoint_group.add_argument(
            "--checkpoint.keep_latest_k",
            type=int,
            default=0,
            help="""
                Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints.
                0 is the default value.
            """,
        )

        # Activation checkpointing configs
        activation_checkpoint_group = self.parser.add_argument_group("activation_checkpoint")
        activation_checkpoint_group.add_argument(
            "--activation_checkpoint.mode",
            type=str,
            default="selective",
            help="Type of activation checkpointing to use ['none', 'full', 'selective']",
        )
        activation_checkpoint_group.add_argument(
            "--activation_checkpoint.selective_ac_option",
            type=str,
            default="2",  # 2 = checkpoint every other layer
            help="""
                Selective activation checkpointing options ['int', 'op'].
                'int' (e.g., 2) for every nth layer, or 'op' for op level ac.
            """,
        )

        # Float8 configs
        float8_group = self.parser.add_argument_group("float8")
        float8_group.add_argument(
            "--float8.enable_float8_linear",
            action="store_true",
            help="""
                If true, swaps `torch.nn.Linear` with `Float8Linear`.
                This feature requires you to install 'torchao' which can be found
                here: https://github.com/pytorch/ao
            """,
        )
        float8_group.add_argument(
            "--float8.enable_fsdp_float8_all_gather",
            action="store_true",
            default=False,
            help="Whether enable float8 all-gather in FSDP",
        )
        float8_group.add_argument(
            "--float8.precompute_float8_dynamic_scale_for_fsdp",
            action="store_true",
            default=False,
            help="Whether precompute float8 scales dynamically for FSDP",
        )
        float8_group.add_argument(
            "--float8.scaling_type_input",
            type=str,
            default="dynamic",
            help="float8 scaling for input, dynamic (default) or delayed",
            choices=["dynamic", "delayed"],
        )
        float8_group.add_argument(
            "--float8.scaling_type_weight",
            type=str,
            default="dynamic",
            help="float8 scaling for input, dynamic (default) or delayed",
        )
        float8_group.add_argument(
            "--float8.scaling_type_grad_output",
            type=str,
            default="dynamic",
            help="float8 scaling for input, dynamic (default) or delayed",
        )

        # Communications library settings
        comm_group = self.parser.add_argument_group("comm")
        comm_group.add_argument(
            "--comm.init_timeout_seconds",
            type=int,
            default=300,
            help="Timeout for communication operations, during initialization and first train step.",
        )
        comm_group.add_argument(
            "--comm.train_timeout_seconds",
            type=int,
            default=100,
            help=(
                "Timeout for communication operations after the first train step -- "
                "usually a tighter bound than during initialization."
            ),
        )
        comm_group.add_argument(
            "--comm.trace_buf_size",
            type=int,
            default=20000,
            help="Flight recorder ring buffer size, >0 means recording by default, 0 means disabled",
        )

        # Memory estimation settings
        memory_estimation_group = self.parser.add_argument_group("memory_estimation")
        memory_estimation_group.add_argument(
            "--memory_estimation.enabled",
            help="Whether to estimate memory usage for FSDP",
            action="store_true",
        )
        memory_estimation_group.add_argument(
            "--memory_estimation.disable_fake_mode",
            help="Whether to estimate memory under FakeTensorMode",
            default=False,
            action="store_true",
        )

        # Bittensor configs
        bittensor_group = self.parser.add_argument_group("bittensor")
        bittensor_group.add_argument(
            "--bittensor.project",
            type=str,
            default="aesop2",
            help="Optional Weights and Biases project name",
        )
        bittensor_group.add_argument(
            "--bittensor.netuid",
            type=int,
            default=220,
            help="Bittensor network UID.",
        )
        bittensor_group.add_argument(
            "--bittensor.bucket",
            type=str,
            default="decis",
            help="S3 bucket name",
        )
        bittensor_group.add_argument(
            "--bittensor.actual_batch_size",
            type=int,
            default=8,
            help="Training batch size per accumulation.",
        )
        bittensor_group.add_argument(
            "--bittensor.device",
            type=str,
            default="cuda",
            help="Device to use for training (e.g., cpu or cuda)",
        )
        bittensor_group.add_argument(
            "--bittensor.use_wandb",
            action="store_true",
            help="Use Weights and Biases for logging",
        )
        bittensor_group.add_argument(
            "--bittensor.remote",
            action="store_true",
            help="Connect to other buckets",
        )
        bittensor_group.add_argument(
            "--bittensor.debug",
            action="store_true",
            help="Enable debug logging",
        )
        bittensor_group.add_argument(
            "--bittensor.trace",
            action="store_true",
            help="Enable trace logging",
        )
        bittensor_group.add_argument(
            "--bittensor.random",
            action="store_true",
            help="Train on random",
        )
        bittensor_group.add_argument(
            "--bittensor.sync_state",
            action="store_true",
            help="Syncs the model state by pulling from the history.",
        )
        bittensor_group.add_argument(
            "--bittensor.baseline",
            action="store_true",
            help="Don't perform syncing with other peers, just train.",
        )

        # Wallet configs
        wallet_group = self.parser.add_argument_group("wallet")
        wallet_group.add_argument(
            "--wallet.name",
            type=str,
            default="Alice",
            help="Name of the wallet",
        )
        wallet_group.add_argument(
            "--wallet.hotkey",
            type=str,
            default="default",
            help="Hotkey for the wallet",
        )

        # Subtensor configs
        subtensor_group = self.parser.add_argument_group("subtensor")
        subtensor_group.add_argument(
            "--subtensor.network",
            type=str,
            default="test",
            help="Subtensor Network (test , finney)",
        )
        subtensor_group.add_argument(
            "--subtensor.chain_endpoint",
            type=str,
            default="wss://test.finney.opentensor.ai:443/",
            help="Endpoint for Subtensor node",
        )

    def parse_arguments(self) -> argparse.Namespace:
        """
        Parses the command-line arguments.

        ## Returns

        - `argparse.Namespace`: The parsed command-line arguments.
        """
        args, _ = self.parser.parse_known_args()
        return args

    def _argv_to_dotlist(self) -> List[str]:
        """
        Converts command-line arguments to a dotlist format suitable for OmegaConf.

        This method processes `sys.argv` to create a list of arguments in the
        'key=value' format, which OmegaConf can interpret. It handles flags,
        key-value pairs, and arguments with '--' prefixes.

        ## Returns

        - `List[str]`: A list of command-line arguments in dotlist notation.

        """
        dotlist = []
        args = sys.argv[1:]  # Skip the script name
        idx = 0

        while idx < len(args):
            arg = args[idx]

            if arg.startswith('--'):
                key = arg.lstrip('--')  # Remove '--' prefix
                if '=' in key:
                    # Argument is in the form --key=value
                    dotlist.append(key)
                else:
                    # Argument is in the form --key value or a flag
                    # Check if the next argument exists and isn't another option
                    if (idx + 1) < len(args) and not args[idx + 1].startswith('--'):
                        value = args[idx + 1]
                        dotlist.append(f"{key}={value}")
                        idx += 1  # Skip next argument as it's a value we've just consumed
                    else:
                        # It's a flag option; set it to true
                        dotlist.append(f"{key}=true")
            elif '=' in arg:
                # Argument is in the form key=value
                dotlist.append(arg)
            else:
                # Positional argument or unrecognized format
                logger.warning(f"Unrecognized argument format: {arg}")
            idx += 1

        return dotlist

    def load_config(self) -> DictConfig:
        """
        Loads configurations from default configs, the specified config file, and merges them
        with command-line arguments.

        ## Returns

        - `DictConfig`: The final merged configuration.

        ## Example Usage

        ```python
        config = JobConfig()
        print(config.job.dump_folder)
        print(config.bittensor.project)
        ```
        """
        # 1. Load default configurations
        default_config: DictConfig = self.load_default_config()

        # 2. Convert command-line arguments to dotlist
        cli_dotlist = self._argv_to_dotlist()

        # 3. Create cli_config from dotlist
        cli_config: DictConfig = OmegaConf.from_dotlist(cli_dotlist)

        # 4. Extract config file path from CLI or use default
        config_file: Union[str, None] = cli_config.get("job_config_file", None)

        # Remove 'job_config_file' from CLI config to prevent overriding after loading the file
        if "job_config_file" in cli_config:
            del cli_config.job_config_file

        # 5. Load the configuration file if specified
        if config_file and os.path.isfile(config_file):
            try:
                # Load the configuration file (supports YAML and TOML)
                file_extension: str = os.path.splitext(config_file)[1]
                if file_extension in ['.yaml', '.yml']:
                    file_config: DictConfig = OmegaConf.load(config_file)
                elif file_extension == '.toml':
                    import tomli
                    with open(config_file, 'rb') as f:
                        toml_config: Dict[str, Any] = tomli.load(f)
                    file_config = OmegaConf.create(toml_config)
                else:
                    logger.error(f"Unsupported config file format: {config_file}")
                    sys.exit(1)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load config file {config_file}: {e}")
                sys.exit(1)
        else:
            file_config: DictConfig = OmegaConf.create()
            logger.warning("No config file specified; using default configurations.")

        # 6. Merge configurations: defaults -> file config -> CLI overrides
        config: DictConfig = OmegaConf.merge(default_config, file_config, cli_config)

        # 7. Resolve interpolations and defaults
        OmegaConf.resolve(config)

        # 8. Log the final configuration
        logger.info("Final Configuration:")
        config_dict = OmegaConf.to_container(config, resolve=True)
        logger.info(json.dumps(config_dict, separators=(',', ':')))

        return config

    def load_default_config(self) -> DictConfig:
        """
        Loads the default configuration.

        ## Returns

        - `DictConfig`: The default configuration.
        """
        default_config = OmegaConf.create({
            "job": {
                "dump_folder": "./boltzmann/outputs",
                "description": "default job",
                "use_for_integration_test": False,
            },
            "profiling": {
                "enable_profiling": False,
                "save_traces_folder": "profile_traces",
                "profile_freq": 10,
                "enable_memory_snapshot": False,
                "save_memory_snapshot_folder": False,
            },
            "metrics": {
                "log_freq": 10,
                "enable_color_printing": False,
                "enable_tensorboard": False,
                "save_tb_folder": "tb",
                "rank_0_only": True,
            },
            "model": {
                "name": "llama",
                "flavor": "debugmodel",
                "norm_type": "rmsnorm",
                "tokenizer_path": "./datasets/tokenizer/tokenizer.model",
            },
            "optimizer": {
                "name": "AdamW",
                "lr": 8e-4,
                "fused": False,
            },
            "training": {
                "dataset": "c4_mini",
                "dataset_path": None,
                "batch_size": 8,
                "seq_len": 2048,
                "warmup_steps": 200,
                "max_norm": 1.0,
                "steps": 10000,
                "data_parallel_replicate_degree": 1,
                "data_parallel_shard_degree": -1,
                "tensor_parallel_degree": 1,
                "enable_loss_parallel": True,
                "compile": False,
                "gc_freq": 50,
                "seed": None,
                "mixed_precision_param": "bfloat16",
                "mixed_precision_reduce": "float32"
            },
            "experimental": {
                "enable_async_tensor_parallel": False,
                "pipeline_parallel_degree": 1,
                "pipeline_parallel_split_points": [],
                "pipeline_parallel_schedule": "1F1B",
                "pipeline_parallel_microbatches": None,
                "enable_compiled_autograd" : False,
            },
            "checkpoint": {
                "enable_checkpoint": False,
                "folder": "checkpoint",
                "interval_type": "steps",
                "interval": 500,
                "model_weights_only": False,
                "export_dtype": "float32",
                "async_mode": "disabled",
                "keep_latest_k": 0,
                "create_seed_checkpoint": False,
            },
            "activation_checkpoint": {
                "mode": "selective",
                "selective_ac_option": "2",
            },
            "float8": {
                "enable_float8_linear": False,
                "enable_fsdp_float8_all_gather": False,
                "precompute_float8_dynamic_scale_for_fsdp": "dynamic",
                "scaling_type_weight": "dynamic",
                "scaling_type_grad_output": "dynamic",
            },
            "comm": {
                "init_timeout_seconds": 300,
                "train_timeout_seconds": 100,
                "trace_buf_size": 20000,
                "scaling_type_weight": "dynamic",
                "scaling_type_grad_output": "dynamic",
            },
            "memory_estimation": {
                "enabled": True,
                "disable_fake_mode": False,
            },
            "bittensor": {
                "project": "aesop2",
                "netuid": 220,
                "bucket": "decis",
                "actual_batch_size": 8,
                "device": "cuda",
                "use_wandb": False,
                "remote": False,
                "debug": False,
                "trace": False,
                "random": False,
                "sync_state": False,
                "baseline": False,
            },
            "wallet": {
                "name": "Alice",
                "hotkey": "default",
            },
            "subtensor": {
                "network": "test",
                "chain_endpoint": "wss://test.finney.opentensor.ai:443/",
            },
        })
        return default_config

    def validate_config(self):
        """
        Validates the configuration to ensure required fields are present.
        """
        # Validate model configuration
        if 'model' not in self.config:
            logger.error("Model configuration is missing.")
            sys.exit(1)

        if 'name' not in self.config.model:
            logger.error("Model name is missing in the configuration.")
            sys.exit(1)

        if 'flavor' not in self.config.model:
            logger.error("Model flavor is missing in the configuration.")
            sys.exit(1)

        # TODO: Add more validations as necessary

    def __getattr__(self, name: str) -> Any:
        """
        Allows access to configuration sections as attributes.

        ## Parameters

        - `name` (str): The attribute name.

        ## Returns

        - `Any`: The requested configuration section.
        """
        if name in self.config:
            return self.config[name]
        else:
            raise AttributeError(f"'JobConfig' object has no attribute '{name}'")

    def __str__(self) -> str:
        """
        Returns a string representation of the configuration.

        ## Returns

        - `str`: The configuration as a YAML-formatted string.
        """
        return OmegaConf.to_yaml(self.config)