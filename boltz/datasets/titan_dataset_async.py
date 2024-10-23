# Optimized Data Loader with Asynchronous Loading and Batch Tokenization

import asyncio
import hashlib
import pickle
from typing import Any, Dict, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging import logger

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

from concurrent.futures import ThreadPoolExecutor

# Map from dataset name to a local directory, or a dataset repository on the HF hub
_supported_datasets = {
    "c4_test": "test/assets/c4_test",
    "c4": "allenai/c4",
}


class HuggingFaceDataset(IterableDataset, Stateful):
    """PyTorch Representation of the HuggingFace Dataset with Optimizations.

    Args:
        dataset_name (str): Name of the dataset to load.
        dataset_path (Optional[str]): Path to the dataset in the file system.
            If provided, data will be loaded from this path instead of downloaded.
        tokenizer (Tokenizer): Tokenizer used to encode data. Must implement
            `encode`, `decode`, and `batch_encode` methods.
        seq_len (int): Maximum sequence length.
        world_size (int): Number of data parallel processes participating in training.
        rank (int): Rank of the current data parallel process.
        infinite (bool): Whether to loop infinitely over the dataset.
        seed (Optional[int]): Seed used to deterministically select samples.
        sampling_probability (float): Probability of including a sample in each epoch.
        num_prefetch_batches (int): Number of batches to prefetch asynchronously.
        max_workers (int): Number of worker threads for asynchronous tasks.

    The dataset supports the C4 dataset and a subset for testing purposes:
    - c4_test (2K training entries)
    - c4 (177M training entries; streamed due to size)
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
        seed: Optional[int] = None,
        sampling_probability: float = 1.0,
        num_prefetch_batches: int = 10,
        max_workers: int = 4,
    ) -> None:
        # Allow user to pass in a (local or HF hub) path to use unsupported datasets
        if dataset_name not in _supported_datasets:
            if dataset_path:
                logger.warning(
                    f"Dataset {dataset_name} is not tested or verified. "
                    f"Recommended datasets are: {list(_supported_datasets.keys())}"
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_name} is not supported. "
                    f"Supported datasets are: {list(_supported_datasets.keys())}"
                )

        if not dataset_path:
            dataset_path = _supported_datasets[dataset_name]
        logger.info(f"Preparing {dataset_name} dataset from {dataset_path}")

        if dataset_name == "c4":
            # C4 is huge and requires both streaming and language selection (default to English)
            ds = load_dataset(dataset_path, name="en", split="train", streaming=True)
        else:
            ds = load_dataset(dataset_path, split="train")

        # TODO: support shuffling
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.seed = seed  # Store the seed
        self.sampling_probability = (
            sampling_probability  # Store the sampling probability
        )
        self.num_prefetch_batches = (
            num_prefetch_batches  # Number of batches to prefetch
        )
        self.max_workers = max_workers  # Number of worker threads for async tasks

        # Initialize an executor for asynchronous processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []

    def __iter__(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        queue = asyncio.Queue(maxsize=self.num_prefetch_batches)

        # Start the asynchronous producer
        producer_task = loop.create_task(self._producer(queue, loop))

        try:
            while True:
                batch = loop.run_until_complete(queue.get())
                if batch is None:
                    break  # End of data
                yield batch
        except Exception as e:
            logger.error(f"Error during data loading: {e}")
        finally:
            # Clean up
            producer_task.cancel()
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _producer(self, queue: asyncio.Queue, loop):
        sample_iter = self._get_data_iter()
        futures = []
        batch_texts = []
        batch_size = 32  # Adjust based on memory constraints

        idx = 0  # Local index for samples within the current iteration
        while True:
            try:
                sample = next(sample_iter)
                sample_index = self._sample_idx + idx  # Total sample index
                idx += 1

                if self._should_include_sample(sample_index):
                    batch_texts.append(sample["text"])

                    if len(batch_texts) >= batch_size:
                        # Submit tokenization task
                        future = loop.run_in_executor(
                            self.executor,
                            self._batch_tokenize,
                            batch_texts,
                        )
                        futures.append(future)
                        batch_texts = []

                    # Manage the queue size
                    while len(futures) >= self.num_prefetch_batches:
                        await self._consume_future(queue, futures.pop(0))

            except StopIteration:
                break  # End of data

        # Tokenize any remaining samples
        if batch_texts:
            future = loop.run_in_executor(
                self.executor,
                self._batch_tokenize,
                batch_texts,
            )
            futures.append(future)

        # Consume remaining futures
        for future in futures:
            await self._consume_future(queue, future)

        await queue.put(None)  # Signal that data loading is done

    async def _consume_future(self, queue: asyncio.Queue, future):
        tokens_list = await future
        for tokens in tokens_list:
            self._all_tokens.extend(tokens)
            while len(self._all_tokens) >= self.seq_len + 1:
                x = torch.LongTensor(self._all_tokens[: self.seq_len + 1])
                self._all_tokens = self._all_tokens[self.seq_len :]
                input = x[:-1]
                label = x[1:]
                await queue.put((input, label))

    def _batch_tokenize(self, texts: List[str]):
        # Tokenize a batch of texts
        tokenized_outputs = self._tokenizer.batch_encode(texts, bos=True, eos=True)
        return tokenized_outputs

    def _should_include_sample(self, sample_index):
        if self.seed is None:
            return True  # Include all samples if no seed is provided
        else:
            # Use a deterministic hash function based on seed and sample_index
            combined = f"{self.seed}_{sample_index}"
            hash_digest = hashlib.sha256(combined.encode()).hexdigest()
            hash_int = int(hash_digest, 16)
            # Map the hash to a float between 0 and 1
            probability = hash_int / (2**256 - 1)
            # Decide whether to include the sample based on the sampling probability
            return probability < self.sampling_probability

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        # As skipping to the end throws an error in case of map-style dataset, return an empty iterator
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_idx))

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "sample_idx": self._sample_idx}


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(
        self,
        dp_rank: int,
        hf_ds: IterableDataset,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__(hf_ds, batch_size, num_workers=num_workers)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}"
            )
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    infinite: bool = True,
    seed: Optional[int] = None,
    sampling_probability: float = 1.0,
    num_prefetch_batches: int = 10,
    max_workers: int = 4,
):
    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        world_size=world_size,
        rank=rank,
        infinite=infinite,
        seed=seed,
        sampling_probability=sampling_probability,
        num_prefetch_batches=num_prefetch_batches,
        max_workers=max_workers,
    )

    return DPAwareDataLoader(
        dp_rank=rank,
        hf_ds=hf_ds,
        batch_size=batch_size,
        num_workers=0,  # Set to 0 because we're handling async loading ourselves
    )
