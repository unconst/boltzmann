# The MIT License (MIT)
# © 2024 Chakana.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import typing
import random
import asyncio
import aiohttp
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler

# Import Stateful for checkpointing support
from torch.distributed.checkpoint.stateful import Stateful

class SubsetLoader(IterableDataset, Stateful):
    """
    Base class for data-specific subset loader classes.
    """

    def __init__(
            self,
            batch_size=None,
            sequence_length=None,
            num_pages=None,
            pages_info=None,
            tokenizer: AutoTokenizer = None,
            pack_samples: bool = False,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_pages = num_pages
        self.tokenizer = tokenizer
        self.pack_samples = pack_samples

        self.num_rows_per_page = 100

        # Buffers to hold data
        self.buffer = []
        self.used_buffer = []
        self.padded_buffer = []

        self.lock = asyncio.Lock()  # For thread-safe operations

        # Distributed properties
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # For checkpointing
        self._sample_idx = 0

    async def fetch_data_for_pages(self, pages):
        """
        Set the pages to be used to fill the buffer. Then fetch the page data
        to the buffer.
        """
        self.pages = pages

        # Empty the buffer if it is not.
        self.buffer = []

        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_data_for_page(page, session) for page in self.pages]
            await asyncio.gather(*tasks)

    async def _fetch_data_for_page(self, page, session):
        """
        Fetches data asynchronously for a single page, processes it without blocking the event loop,
        and appends the tokenized data to the buffer.
        Args:
            page: A tuple containing the config name, page number, and split.
            session: The HTTP session used for making requests.
        Raises:
            Exception: If the maximum number of retry attempts is exceeded.
        """
        retry_limit = 10
        attempt = 0
        while attempt < retry_limit:
            config_name, page_number, split = page

            # Create the request parameters
            params = {
                'dataset': self.name,
                'config': config_name,
                'split': split,
                'offset': page_number,
                'limit': self.num_rows_per_page
            }

            try:
                # Make an asynchronous HTTP GET request to fetch the data
                async with session.get(self.rows_base_url, params=params) as response:
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    data = await response.json()

                    # Prepare the data to append
                    buffer_to_append = []

                    # Asynchronously process each row without blocking the event loop
                    tasks = [
                        self._tokenize_content(row["row"]["text"]) for row in data["rows"]
                    ]

                    # Gather the tokenized results concurrently
                    row_input_ids = await asyncio.gather(*tasks)

                    # Flatten the list of input IDs and append them to the buffer
                    for input_ids in row_input_ids:
                        buffer_to_append.extend(input_ids)

                    # Safely append the processed data to the shared buffer
                    async with self.lock:
                        self.buffer.extend(buffer_to_append)
                        self.pages.append((config_name, page_number, split))
                    break  # Success, exit retry loop

            except aiohttp.ClientResponseError as e:
                # Handle HTTP client errors with a retry mechanism
                attempt += 1
                if attempt < retry_limit:
                    await asyncio.sleep(5)  # Wait before retrying
                else:
                    raise Exception(f"Maximum retry attempts exceeded for page {page}") from e

    async def _tokenize_content(self, content):
        """
        Asynchronously tokenizes a string of content using the tokenizer in a separate thread.
        Args:
            content: The text content to be tokenized.
        Returns:
            The list of token IDs for the content, including the EOS token.
        """
        # Offload the CPU-bound tokenization to a thread executor to prevent blocking the event loop
        input_ids = await asyncio.to_thread(
            self.tokenizer.encode, content, truncation=True, max_length=self.sequence_length
        )
        input_ids.append(self.tokenizer.eos_token_id)
        return input_ids

    def _get_pad_size(self, input_ids):
        """
        Get the number of tokens to be padded to the sample to match
        the max allowed sequence length.
        If sample packing is activated, then return 1
        """

        if self.pack_samples:
            return 1

        sample_size = len(input_ids)

        remainder = (sample_size % self.sequence_length)
        pad_size = (self.sequence_length - remainder)

        # Apply modulo again to guarantee a pad size of 0 if remainder is 0
        pad_size = pad_size % self.sequence_length

        return pad_size

    def _refill_padded_buffer(self):
        """
        This method pulls data from `self.buffer`, pads it, and pushes
        it to the `self.padded_buffer`.
        """

        while (
                self.buffer
                and len(self.padded_buffer) < self.sequence_length * self.batch_size
        ):
            input_ids = []

            # Search for EOS token index and cut the buffer at it.
            try:
                EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
                input_ids = self.buffer[:EOS_index+1]
                self.buffer = self.buffer[EOS_index+1:]
            except ValueError:
                # If EOS token is not found, take all data
                input_ids = self.buffer
                self.buffer = []

            self.used_buffer += input_ids

            # Add to padded buffer without the EOS token
            self.padded_buffer += input_ids[:-1]

            # Pad
            pad_size = self._get_pad_size(input_ids=input_ids[:-1])
            self.padded_buffer += [self.tokenizer.eos_token_id] * pad_size

    def __iter__(self):
        # Handling infinite looping
        while True:
            if not self.padded_buffer:
                self.buffer = self.used_buffer + self.buffer
                self.used_buffer = []
                self.padded_buffer = []

                # Pad and prepare data for batching
                self._refill_padded_buffer()

            # Yield batches
            while len(self.padded_buffer) >= self.sequence_length * self.batch_size:
                batch_input_ids = []
                batch_labels = []
                for _ in range(self.batch_size):
                    # Take sequence_length tokens from padded_buffer
                    input_ids = self.padded_buffer[:self.sequence_length]
                    self.padded_buffer = self.padded_buffer[self.sequence_length:]

                    # Generate labels from input_ids
                    labels = input_ids.copy()

                    batch_input_ids.append(input_ids)
                    batch_labels.append(labels)

                yield np.array(batch_input_ids), np.array(batch_labels)

            # If data is exhausted, break if not infinite looping
            if not self.buffer and not self.padded_buffer:
                break

    def __len__(self):
        # Estimate total number of batches
        total_samples = self.num_pages * self.num_rows_per_page
        total_tokens = total_samples * self.sequence_length
        return total_tokens // (self.sequence_length * self.batch_size * self.world_size)

    def state_dict(self):
        """
        Returns a dictionary containing the state of the dataset loader for checkpointing.
        """
        return {
            'pages': self.pages,
            'buffer': self.buffer,
            'used_buffer': self.used_buffer,
            'padded_buffer': self.padded_buffer,
            '_sample_idx': self._sample_idx
        }

    def load_state_dict(self, state_dict):
        """
        Loads the dataset loader state from a checkpoint.
        """
        self.pages = state_dict['pages']
        self.buffer = state_dict['buffer']
        self.used_buffer = state_dict['used_buffer']
        self.padded_buffer = state_dict['padded_buffer']
        self._sample_idx = state_dict['_sample_idx']


class DatasetLoader(SubsetLoader):
    name: str = "HuggingFaceFW/fineweb-edu-score-2"
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"

    retry_limit: int = 10  # Number of retries
    retry_delay: int = 5  # Seconds to wait between retries

    @classmethod
    async def create(
            cls,
            batch_size=None,
            sequence_length=None,
            num_pages=None,
            pages_info=None,
            tokenizer: AutoTokenizer = None,
            pack_samples: bool = False,
    ):
        self = cls(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_pages=num_pages,
            tokenizer=tokenizer,
            pack_samples=pack_samples
        )

        # Fetch dataset configs asynchronously
        self.configs_data = await cls.fetch_dataset_configs()

        if pages_info is not None:
            pages = pages_info
        elif self.num_pages:
            pages = self.get_random_pages(self.num_pages)
        else:
            pages = []

        await self.fetch_data_for_pages(pages)

        return self

    @staticmethod
    async def fetch_dataset_configs() -> typing.Dict[str, typing.Dict]:
        """
        Fetch the different configurations of the dataset.
        The returned value is a dictionary with config names as keys and
        a dict of the number of rows and the split as values.
        """
        params = dict(
            dataset=DatasetLoader.name
        )

        attempt = 0
        while attempt < DatasetLoader.retry_limit:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(DatasetLoader.size_base_url, params=params) as response:
                        response.raise_for_status()

                        data = await response.json()

                        # Extract the configs dict
                        configs_dict = data['size']['splits']

                        # Create a dict with config names as keys, and the number of rows as values
                        configs_data = {entry['config']: {'num_rows': entry['num_rows'],
                                                          'split': entry['split']}
                                        for entry in configs_dict
                                        if entry['config'] != 'default'
                                        }

                        return configs_data

            except aiohttp.ClientResponseError as e:
                attempt += 1
                if attempt < DatasetLoader.retry_limit:
                    await asyncio.sleep(DatasetLoader.retry_delay)
                else:
                    raise

    def get_random_pages(self, num_pages):
        """
        Randomly sample pages in a distributed manner.
        A page is a row number of a given split of a given dataset dump.
        """
        pages = []

        # Adjust the number of pages per rank
        pages_per_rank = num_pages // self.world_size
        start_index = self.rank * pages_per_rank
        end_index = start_index + pages_per_rank

        # In case num_pages is not divisible by world_size
        if self.rank == self.world_size - 1:
            end_index = num_pages

        # Seed the random number generator differently for each rank
        rng = np.random.default_rng(seed=42 + self.rank)

        # Each rank generates its own pages
        for _ in range(start_index, end_index):
            # Choose a random config
            config_name = rng.choice(list(self.configs_data.keys()))

            # Choose a random page (row)
            max_offset = self.configs_data[config_name]['num_rows'] - 1 - self.num_rows_per_page
            page = rng.integers(0, max(0, max_offset))

            split = self.configs_data[config_name]['split']

            pages.append((config_name, page, split))

        return pages

    @staticmethod
    async def next_pages(offset: int, n_pages: int, seed: str, num_rows_per_page: int = 100):
        configs_data = await DatasetLoader.fetch_dataset_configs()
        rng = np.random.default_rng(hash(seed) & 0xffffffff)  # Create a generator with a seed
        rng.bit_generator.advance(offset)  # Efficiently skip ahead `n` steps

        # Distributed settings
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        pages_per_rank = n_pages // world_size
        start_page = rank * pages_per_rank
        end_page = start_page + pages_per_rank

        # In case n_pages is not divisible by world_size
        if rank == world_size - 1:
            end_page = n_pages

        result = []
        configs_keys = list(configs_data.keys())
        for _ in range(start_page, end_page):
            config = rng.choice(configs_keys)
            max_offset = configs_data[config]['num_rows'] - 1 - num_rows_per_page
            choice = rng.integers(0, max(0, max_offset))
            result.append((str(config), int(choice), configs_data[config]['split']))
        return result

    @staticmethod
    async def next_pages_async(offset: int, n_pages: int, seed: str, num_rows_per_page: int = 100):
        # Alias for compatibility
        return await DatasetLoader.next_pages(offset, n_pages, seed, num_rows_per_page)
