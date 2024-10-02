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
import requests
import asyncio
import aiohttp
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset

class SubsetLoader(IterableDataset):
    """
    Base class for data-specific subset loader classes.
    """

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_pages = num_pages
        self.tokenizer = tokenizer
        self.pack_samples = pack_samples

        self.num_rows_per_page = 100

        # Buffer to hold pages loaded from the API
        self.buffer = []

        # Buffer to hold pages already loaded into a batch
        self.used_buffer = []

        # Buffer to hold padded sequences
        self.padded_buffer = []

        self.lock = asyncio.Lock()  # For thread-safe operations

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
        retry_limit = 10
        attempt = 0
        while attempt < retry_limit:
            config_name, page_number, split = page

            # Create the request parameters
            params = dict(dataset=self.name,
                          config=config_name,
                          split=split,
                          offset=page_number,
                          limit=self.num_rows_per_page
                          )

            try:
                async with session.get(self.rows_base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Prepare the data to append
                    buffer_to_append = []
                    for row in data["rows"]:
                        content = row["row"]["text"]
                        input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                        buffer_to_append.extend(input_ids)
                        buffer_to_append.append(self.tokenizer.eos_token_id)

                    async with self.lock:
                        self.buffer.extend(buffer_to_append)
                        self.pages.append((config_name, page_number, split))
                    break  # Success, exit retry loop

            except aiohttp.ClientResponseError as e:
                attempt += 1
                if attempt < retry_limit:
                    await asyncio.sleep(5)
                else:
                    raise

    def _get_pad_size(self, input_ids):
        """
        Get the number of tokens to be padded to the sample to match
        the max allowed sequence length.
        If sample packing is activated, then return 1
        """

        if self.pack_samples:
            return 1

        sample_size = len(input_ids)

        remainder = sample_size % self.sequence_length
        pad_size = self.sequence_length - remainder

        # Apply modulo again to guarantee a pad size of 0 if remainder is 0
        pad_size = pad_size % self.sequence_length

        return pad_size

    def _refill_padded_buffer(self):
        """
        This method pulls sequences from `self.buffer`, pads them, and pushes
        them to the `self.padded_buffer`.
        """

        while self.buffer and len(self.padded_buffer) < self.sequence_length:
            input_ids = []

            # Search for EOS token index and cut the buffer at it.
            try:
                EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
            except ValueError:
                # If EOS token is not found, take the entire buffer
                EOS_index = len(self.buffer) - 1

            input_ids = self.buffer[: EOS_index + 1]
            self.buffer = self.buffer[EOS_index + 1 :]

            self.used_buffer += input_ids

            # Add to padded buffer without the EOS token.
            input_ids_without_eos = (
                input_ids[:-1]
                if input_ids[-1] == self.tokenizer.eos_token_id
                else input_ids
            )
            self.padded_buffer += input_ids_without_eos

            # Pad
            pad_size = self._get_pad_size(input_ids=input_ids_without_eos)
            self.padded_buffer += [self.tokenizer.eos_token_id] * pad_size

    def __iter__(self):
        self.buffer = self.used_buffer + self.buffer
        self.padded_buffer = []

        # Pad and prepare data for batching
        self._refill_padded_buffer()

        return self

    def __next__(self):
        batch = []

        while True:
            # If we have enough data in padded_buffer, create batches
            while (
                len(self.padded_buffer) >= self.sequence_length
                and len(batch) < self.batch_size
            ):
                batch.append(self.padded_buffer[: self.sequence_length])
                self.padded_buffer = self.padded_buffer[self.sequence_length :]

            if len(batch) == self.batch_size:
                return np.stack(batch)

            # If padded_buffer is insufficient, try to refill it
            if len(self.padded_buffer) < self.sequence_length:
                if self.buffer:
                    self._refill_padded_buffer()
                elif not self.data_queue.empty():
                    # Get data from the queue
                    self.buffer.extend(self.data_queue.get())
                    self._refill_padded_buffer()
                elif self.stop_event.is_set():
                    if batch:
                        return np.stack(batch)
                    else:
                        raise StopIteration
                else:
                    # Wait for data to become available
                    time.sleep(0.1)
            else:
                if batch:
                    return np.stack(batch)
                else:
                    raise StopIteration


class AsyncSubsetFineWebEdu2Loader(SubsetLoader):

    # Class-level constants
    name: str = "HuggingFaceFW/fineweb-edu-score-2"
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"

    # Configuration parameters
    retry_limit: int = 10  # Number of retries
    retry_delay: int = 5  # Seconds to wait between retries
    num_rows_per_page: int = 100

    @staticmethod
    async def next_pages(offset: int, n_pages: int, seed: str, num_rows_per_page: int = 100):
        configs_data = await AsyncSubsetFineWebEdu2Loader.fetch_dataset_configs()
        rng = np.random.default_rng(hash(seed) & 0xffffffff)  # Create a generator with a seed
        rng.bit_generator.advance(offset)  # Efficiently skip ahead `n` steps
        result = []
        for _ in range(n_pages):
            config = rng.choice(list(configs_data.keys()))
            choice = rng.integers(0, configs_data[config]['num_rows'] - 1 - num_rows_per_page)
            result.append((str(config), int(choice), configs_data[config]['split']))
        return result

    def __init__(
            self,
            batch_size=None,
            sequence_length=None,
            num_pages=None,
            pages_info=None,
            tokenizer: AutoTokenizer = None,
            pack_samples: bool = False,
    ):
        # Call the parent class constructor
        super().__init__(
            batch_size, sequence_length, num_pages, tokenizer, pack_samples
        )

        # Initialize properties
        self.configs_data = None
        self.pages = []
        self.buffer = []
        self.lock = asyncio.Lock()  # For thread-safe operations

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
            await self._fetch(pages_info)
        elif self.num_pages:
            await self._fetch_data_to_buffer(self.num_pages)

        return self

    async def _fetch(self, page_info: typing.Tuple[str, int, str]):
        self.pages = page_info
        num_pages = len(self.pages)
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_data_for_page((config_name, page, split), session)
                     for (config_name, page, split) in self.pages]
            await asyncio.gather(*tasks)

    async def _fetch_data_to_buffer(self, num_pages):
        """
        Randomly sample pages and add their data to the buffer.
        If a page is inaccessible, another one is sampled.
        This method sets the `pages` property.
        """
        self.pages = []
        pages_to_fetch = self.get_random_pages(num_pages)

        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_data_for_page(page, session) for page in pages_to_fetch]
            await asyncio.gather(*tasks)

    async def fetch_data_to_rows(self, num_pages):
        rows = []
        pages_to_fetch = self.get_random_pages(num_pages)

        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_rows_for_page(page, session) for page in pages_to_fetch]
            results = await asyncio.gather(*tasks)
            for page_rows in results:
                rows.extend(page_rows)

        return rows

    async def _fetch_data_for_page(self, page, session):
        retry_limit = self.retry_limit
        attempt = 0
        while attempt < retry_limit:
            config_name, page_number, split = page

            # Create the request parameters
            params = dict(dataset=self.name,
                          config=config_name,
                          split=split,
                          offset=page_number,
                          limit=self.num_rows_per_page
                          )

            try:
                async with session.get(self.rows_base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Prepare the data to append
                    buffer_to_append = []
                    for row in data["rows"]:
                        content = row["row"]["text"]
                        input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                        buffer_to_append.extend(input_ids)
                        buffer_to_append.append(self.tokenizer.eos_token_id)

                    async with self.lock:
                        self.buffer.extend(buffer_to_append)
                        self.pages.append((config_name, page_number, split))
                    break  # Success, exit retry loop

            except aiohttp.ClientResponseError as e:
                attempt += 1
                if attempt < retry_limit:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise

    async def _fetch_rows_for_page(self, page, session):
        retry_limit = self.retry_limit
        attempt = 0
        while attempt < retry_limit:
            config_name, page_number, split = page

            # Create the request parameters
            params = dict(dataset=self.name,
                          config=config_name,
                          split=split,
                          offset=page_number,
                          limit=self.num_rows_per_page
                          )

            try:
                async with session.get(self.rows_base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Collect the rows
                    return [row["row"]["text"] for row in data["rows"]]

            except aiohttp.ClientResponseError as e:
                attempt += 1
                if attempt < retry_limit:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise

    def get_random_pages(self, num_pages):
        """
        Randomly sample pages.
        A page is a row number of a given split of a given dataset dump.
        """
        pages = []

        for _ in range(num_pages):
            # Choose a random config
            config_name = random.choice(list(self.configs_data.keys()))

            # Choose a random page (row)
            page = random.randint(0,
                                  self.configs_data[config_name]['num_rows'] - 1 - self.num_rows_per_page)

            split = self.configs_data[config_name]['split']

            pages.append((config_name, page, split))

        return pages

    def get_page_names(self):
        """
        This is a utility function that returns the page names that were used.
        Each page as a single string instead of a tuple.
        """
        page_names = []

        if hasattr(self, 'pages'):
            page_names = [f'{cfg_name}_{num_rows}_{split}' for
                          cfg_name, num_rows, split in self.pages]

        return page_names

    @staticmethod
    async def fetch_dataset_configs() -> typing.Dict[str, typing.Dict]:
        """
        Fetches the dataset configurations, including the number of rows and splits.
        """
        # Request parameters
        params = dict(
            dataset=AsyncSubsetFineWebEdu2Loader.name
        )

        # Initialize retry attempt counter
        attempt = 0
        while attempt < AsyncSubsetFineWebEdu2Loader.retry_limit:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(AsyncSubsetFineWebEdu2Loader.size_base_url, params=params) as response:
                        response.raise_for_status()

                        data = await response.json()

                        # Extract the configs dict
                        configs_dict = data['size']['splits']

                        # Now create a dict with config names (except 'default') as
                        # keys, and the number of rows as values
                        configs_data = {entry['config']: {'num_rows': entry['num_rows'],
                                                          'split': entry['split']}
                                        for entry in configs_dict
                                        if entry['config'] != 'default'
                                        }

                        return configs_data

            except aiohttp.ClientResponseError as e:
                attempt += 1
                if attempt < AsyncSubsetFineWebEdu2Loader.retry_limit:
                    await asyncio.sleep(AsyncSubsetFineWebEdu2Loader.retry_delay)
                else:
                    # Raise the exception if all retries are exhausted
                    raise

    @staticmethod
    async def next_pages_async(offset: int, n_pages: int, seed: str, num_rows_per_page: int = 100):
        configs_data = await AsyncSubsetFineWebEdu2Loader.fetch_dataset_configs()
        rng = np.random.default_rng(hash(seed) & 0xffffffff)  # Create a generator with a seed
        rng.bit_generator.advance(offset)  # Efficiently skip ahead `n` steps
        result = []
        for _ in range(n_pages):
            config = rng.choice(list(configs_data.keys()))
            choice = rng.integers(0, configs_data[config]['num_rows'] - 1 - num_rows_per_page)
            result.append((str(config), int(choice), configs_data[config]['split']))
        return result
