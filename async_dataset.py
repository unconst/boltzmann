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
from typing import List, Dict, Tuple
import requests
import numpy as np
from tqdm import tqdm
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import asyncio
import aiohttp
from queue import Queue, Empty
import threading
import time


class SubsetLoader(IterableDataset):
    """
    Base class for data-specific subset loader classes.

    # TODO: Make this class abstract
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

        # Buffer to hold pages loaded from the api
        self.buffer = []

        # Buffer to hold pages already loaded into a batch
        self.used_buffer = []

        # Buffer to hold padded pages
        self.padded_buffer = []

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
        This methods pulls one page from `self.buffer`, pads it and pushs
        it to the `self.padded_buffer`.
        """

        while self.buffer and len(self.padded_buffer) < self.sequence_length:

            input_ids = []

            # search for EOS token index and cut the buffer at it.
            EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
            input_ids = self.buffer[: EOS_index + 1]
            self.buffer = self.buffer[EOS_index + 1 :]

            self.used_buffer += input_ids

            # Add to padded buffer without the EOS token.
            self.padded_buffer += input_ids[:-1]

            # Pad
            self.padded_buffer += [self.tokenizer.eos_token_id] * self._get_pad_size(
                input_ids=input_ids[:-1]
            )

    def __iter__(self):
        self.buffer = self.used_buffer + self.buffer
        self.padded_buffer = []

        # Pad and prepare one page for batching
        self._refill_padded_buffer()

        return self

    def __next__(self):
        batch = []

        while len(self.padded_buffer) >= self.sequence_length:
            batch.append(self.padded_buffer[: self.sequence_length])
            self.padded_buffer = self.padded_buffer[self.sequence_length :]
            self._refill_padded_buffer()

            if len(batch) == self.batch_size:
                return np.stack(batch)

        raise StopIteration


class SubsetFineWebEdu2Loader(SubsetLoader):
    """
    A class for loading and managing subsets of the FineWebEdu2 dataset.

    This class extends SubsetLoader to provide specific functionality for the
    HuggingFaceFW/fineweb-edu-score-2 dataset, including asynchronous data fetching
    and management of dataset configurations.

    Attributes:
        name (str): The name of the dataset.
        rows_base_url (str): The base URL for fetching dataset rows.
        size_base_url (str): The base URL for fetching dataset size information.
        retry_limit (int): The maximum number of retry attempts for API requests.
        retry_delay (int): The delay between retry attempts in seconds.
        num_rows_per_page (int): The number of rows to fetch per page.

    Example usage:
        loader = SubsetFineWebEdu2Loader(batch_size=32, sequence_length=512, num_pages=10)
        for batch in loader:
            # Process the batch
    """

    # 1. Class-level constants
    name: str = "HuggingFaceFW/fineweb-edu-score-2"
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"

    # 2. Configuration parameters
    retry_limit: int = 10  # Number of retries
    retry_delay: int = 5  # Seconds to wait between retries
    num_rows_per_page: int = 100

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        pages_info=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
    ):
        """
        Initialize the SubsetFineWebEdu2Loader.

        Args:
            batch_size (int, optional): The size of each batch.
            sequence_length (int, optional): The length of each sequence.
            num_pages (int, optional): The number of pages to fetch.
            pages_info (List[Tuple[str, int, str]], optional): Specific pages to fetch.
            tokenizer (AutoTokenizer, optional): The tokenizer to use.
            pack_samples (bool, optional): Whether to pack samples. Defaults to False.
        """
        # 1. Call the parent class constructor
        super().__init__(
            batch_size, sequence_length, num_pages, tokenizer, pack_samples
        )

        # 2. Fetch dataset configurations
        self.configs_data = SubsetFineWebEdu2Loader.fetch_dataset_configs()

        # 3. Initialize buffers and tokenizer
        self.padded_buffer = []
        self.tokenizer = tokenizer

        # 4. Handle initialization based on pages_info
        if pages_info is not None:
            # 4a. Set pages and num_pages if pages_info is provided
            self.pages = pages_info
            self.num_pages = len(self.pages)
            # 4b. Initialize buffer and fetch data synchronously
            self.buffer = []
            asyncio.run(self.fetch_data_for_pages(self.pages))
            # 4c. Set stop event to indicate completion
            self.stop_event = threading.Event()
            self.stop_event.set()
        else:
            # 4d. Initialize for asynchronous fetching
            self.pages = None
            self.num_pages = num_pages
            self.data_queue = Queue(maxsize=1)
            self.stop_event = threading.Event()
            self.loop = asyncio.new_event_loop()
            # 4e. Start background fetching thread
            self.fetch_thread = threading.Thread(
                target=self._start_event_loop, daemon=True
            )
            self.fetch_thread.start()

    def _start_event_loop(self):
        """
        Start the asynchronous event loop for background data fetching.
        """
        # 1. Set the event loop for the current thread
        asyncio.set_event_loop(self.loop)
        # 2. Run the asynchronous background fetch method
        self.loop.run_until_complete(self._async_background_fetch())

    async def _async_background_fetch(self):
        """
        Asynchronously fetches data in the background and adds it to the data queue.
        """
        # 1. Continue fetching until stop event is set
        while not self.stop_event.is_set():
            # 2. Check if specific pages are provided
            if self.pages is not None:
                # 2a. Fetch data for specified pages
                pages_to_fetch: List[Tuple[str, int, str]] = self.pages
                await self.fetch_data_for_pages(pages_to_fetch)
                # 2b. Put fetched data in the queue
                self.data_queue.put(self.buffer)
                # 2c. Log fetching information
                num_pages_fetched: int = len(pages_to_fetch)
                print(f"[INFO] Fetched {num_pages_fetched} pages.")
                # 2d. Stop fetching
                self.stop_event.set()
            else:
                # 3. Fetch random pages if no specific pages are provided
                # 3a. Get random pages
                pages_to_fetch: List[Tuple[str, int, str]] = self.get_random_pages(
                    num_pages=self.num_pages
                )
                # 3b. Fetch data for selected pages
                await self.fetch_data_for_pages(pages_to_fetch)
                # 3c. Put fetched data in the queue
                self.data_queue.put(self.buffer)
                # 3d. Clear the buffer for the next batch
                self.buffer = []
                # 3e. Yield control to allow other tasks to run
                await asyncio.sleep(0)

    @staticmethod
    def fetch_dataset_configs() -> Dict[str, Dict]:
        """
        Fetches the dataset configurations, including the number of rows and splits.

        Returns:
            Dict[str, Dict]: A dictionary with config names as keys and dictionaries
                             of num_rows and split as values.

        Raises:
            requests.exceptions.RequestException: If the request fails after all retry attempts.
        """
        # 1. Set up request parameters
        params = {"dataset": SubsetFineWebEdu2Loader.name}

        # 2. Initialize retry attempt counter
        attempt = 0
        # 3. Retry loop
        while attempt < SubsetFineWebEdu2Loader.retry_limit:
            try:
                # 3a. Send GET request to fetch dataset size information
                response = requests.get(
                    SubsetFineWebEdu2Loader.size_base_url, params=params
                )
                # 3b. Raise an exception for bad status codes
                response.raise_for_status()

                # 4. Extract the configs dictionary from the response
                configs_dict = response.json()["size"]["splits"]

                # 5. Create a dictionary with config names as keys and number of rows as values
                configs_data = {
                    entry["config"]: {
                        "num_rows": entry["num_rows"],
                        "split": entry["split"],
                    }
                    for entry in configs_dict
                    if entry["config"] != "default"
                }

                # 6. Return the processed config data
                return configs_data

            except requests.exceptions.RequestException as e:
                # 7. Handle request exceptions
                attempt += 1
                if attempt < SubsetFineWebEdu2Loader.retry_limit:
                    # 7a. Wait before retrying
                    time.sleep(SubsetFineWebEdu2Loader.retry_delay)
                else:
                    # 7b. Raise the exception if all retries are exhausted
                    raise

    @staticmethod
    def next_pages(offset: int, n_pages: int, seed: str, num_rows_per_page: int = 100):
        """
        Generate a sequence of random pages based on a given seed and offset.

        Args:
            offset (int): The starting offset for the random number generator.
            n_pages (int): The number of pages to generate.
            seed (str): The seed for the random number generator.
            num_rows_per_page (int): The number of rows per page. Defaults to 100.

        Returns:
            List[Tuple[str, int, str]]: A list of tuples containing (config_name, page_number, split).

        Example:
            >>> next_pages(0, 2, "example_seed")
            [('config1', 42, 'train'), ('config2', 123, 'test')]
        """
        # 1. Fetch the dataset configurations
        configs_data = SubsetFineWebEdu2Loader.fetch_dataset_configs()

        # 2. Initialize the random number generator with the given seed
        rng = np.random.default_rng(hash(seed) & 0xFFFFFFFF)

        # 3. Advance the random number generator by the offset
        rng.bit_generator.advance(offset)

        # 4. Initialize the result list
        result = []

        # 5. Generate n_pages random pages
        for _ in range(n_pages):
            # 5a. Choose a random config
            config = rng.choice(list(configs_data.keys()))

            # 5b. Choose a random page number within the config's range
            choice = rng.integers(
                0, configs_data[config]["num_rows"] - 1 - num_rows_per_page
            )

            # 5c. Append the chosen page to the result
            result.append((str(config), int(choice), configs_data[config]["split"]))

        # 6. Return the list of generated pages
        return result

    async def fetch_data_for_pages(self, pages):
        """
        Asynchronously fetch data for multiple pages.

        Args:
            pages (List[Tuple[str, int, str]]): A list of pages to fetch data for.

        Note:
            This method populates the self.buffer with fetched data.
        """
        # 1. Clear the buffer before fetching new data
        self.buffer = []

        # 2. Use an asynchronous context manager for the client session
        async with aiohttp.ClientSession() as session:
            # 3. Create a list of tasks for fetching data from each page
            tasks = [self._fetch_data_for_page(session, page) for page in pages]

            # 4. Gather results from all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 5. Process the results
            for result in results:
                if isinstance(result, Exception):
                    # 5a. Handle exceptions if necessary
                    print(f"[ERROR] Exception occurred: {result}")
                else:
                    # 5b. Extend the buffer with fetched data
                    self.buffer.extend(result)

        # 6. Log information about the fetch operation
        print(f"[INFO] Fetched {len(pages)} pages asynchronously.")
        print(f"[DEBUG] Total tokens fetched: {len(self.buffer)}")

    async def _fetch_data_for_page(
        self, session: aiohttp.ClientSession, page: Tuple[str, int, str]
    ) -> List[int]:
        """
        Asynchronously fetches data for a single page.

        Args:
            session (aiohttp.ClientSession): The aiohttp client session.
            page (Tuple[str, int, str]): A tuple containing (config_name, offset, split).

        Returns:
            List[int]: A list of token IDs for the fetched page.

        Note:
            This method implements retry logic in case of failures.
        """
        # 1. Unpack the page tuple
        config_name, offset, split = page

        # 2. Prepare the request parameters
        params = {
            "dataset": self.name,
            "config": config_name,
            "split": split,
            "offset": offset,
            "limit": self.num_rows_per_page,
        }

        # 3. Initialize the attempt counter
        attempt = 0

        # 4. Retry loop
        while attempt < self.retry_limit:
            try:
                # 4a. Make an asynchronous GET request
                async with session.get(self.rows_base_url, params=params) as response:
                    # 4b. Raise an exception for bad status codes
                    response.raise_for_status()

                    # 4c. Parse the JSON response
                    json_response = await response.json()

                    # 4d. Initialize a list to store token IDs
                    input_ids_list = []

                    # 4e. Process each row in the response
                    for row in json_response.get("rows", []):
                        content = row["row"].get("text", "")
                        if content:
                            # 4f. Encode the content and add EOS token
                            input_ids = self.tokenizer.encode(content, truncation=True)
                            input_ids.append(self.tokenizer.eos_token_id)
                            input_ids_list.extend(input_ids)
                        else:
                            # 4g. Log a warning for empty content
                            print(f"[WARNING] Empty content in row: {row}")

                    # 4h. Return the list of token IDs
                    return input_ids_list

            except aiohttp.ClientError as e:
                # 5. Handle client errors (e.g., network issues)
                attempt += 1
                print(
                    f"[WARNING] Failed to fetch page {page} (attempt {attempt}/{self.retry_limit}): {e}"
                )
                if attempt < self.retry_limit:
                    # 5a. Wait before retrying
                    await asyncio.sleep(self.retry_delay)
                else:
                    # 5b. Log the error after all attempts fail
                    print(
                        f"[ERROR] Failed to fetch page {page} after {self.retry_limit} attempts."
                    )
                    return []

            except Exception as e:
                # 6. Handle unexpected exceptions
                print(f"[ERROR] Unexpected exception: {e}")
                return []

    def get_random_pages(self, num_pages):
        """
        Randomly sample pages from the dataset.

        Args:
            num_pages (int): Number of pages to sample.

        Returns:
            list: A list of tuples, each containing (config_name, page, split).

        Example:
            >>> loader = SubsetFineWebEdu2Loader()
            >>> random_pages = loader.get_random_pages(5)
            >>> print(random_pages)
            [('config1', 42, 'train'), ('config2', 123, 'test'), ...]
        """
        # 1. Initialize an empty list to store the sampled pages
        pages = []

        # 2. Iterate 'num_pages' times to sample pages
        for _ in range(num_pages):
            # 2a. Choose a random config name from the available configs
            config_name = np.random.choice(list(self.configs_data.keys()))

            # 2b. Choose a random page number within the range of available rows
            #     Subtract num_rows_per_page to ensure we don't exceed the dataset size
            page = np.random.randint(
                0,
                self.configs_data[config_name]["num_rows"] - 1 - self.num_rows_per_page,
            )

            # 2c. Get the split (e.g., 'train', 'test') for the chosen config
            split = self.configs_data[config_name]["split"]

            # 2d. Append the sampled page info as a tuple to the pages list
            pages.append((config_name, page, split))

        # 3. Return the list of sampled pages
        return pages

    def __next__(self):
        """
        Retrieve the next batch of data.

        Returns:
            np.ndarray: A batch of sequences, shape (batch_size, sequence_length).

        Raises:
            StopIteration: When there's no more data to yield.

        Note:
            This method handles the complex logic of building batches from the buffer
            and managing the asynchronous data fetching process.
        """
        # 1. Initialize an empty list to store the batch
        batch = []

        # 2. Main loop to build the batch
        while len(batch) < self.batch_size:
            # 2a. Check if there's enough data in the padded buffer
            if len(self.padded_buffer) >= self.sequence_length:
                # 3. Build batches from padded_buffer
                while (
                    len(self.padded_buffer) >= self.sequence_length
                    and len(batch) < self.batch_size
                ):
                    # 3a. Extract a sequence and add it to the batch
                    batch.append(self.padded_buffer[: self.sequence_length])
                    # 3b. Remove the extracted sequence from the padded buffer
                    self.padded_buffer = self.padded_buffer[self.sequence_length :]
                # 3c. If we have a full batch, return it
                if batch:
                    print(f"[DEBUG] Yielding batch of size: {len(batch)}")
                    return np.stack(batch)
            # 4. If padded buffer is insufficient, check the main buffer
            elif self.buffer:
                # 4a. Extend padded_buffer with new data from the main buffer
                self.padded_buffer.extend(self.buffer)
                print(
                    f"[DEBUG] Extended padded_buffer. New size: {len(self.padded_buffer)}"
                )
                # 4b. Clear the main buffer
                self.buffer = []
            # 5. Check if data fetching has stopped
            elif self.stop_event.is_set():
                # 5a. If padded buffer still has enough data, continue building batches
                if len(self.padded_buffer) >= self.sequence_length:
                    continue
                # 5b. If we have a partial batch, return it
                elif batch:
                    print(f"[DEBUG] Returning final batch of size: {len(batch)}")
                    return np.stack(batch)
                # 5c. If no more data, stop iteration
                else:
                    print("[DEBUG] No more data to yield. Stopping iteration.")
                    raise StopIteration
            # 6. Handle unexpected case (should not occur with pages_info provided)
            else:
                print("[ERROR] No data in buffer and stop_event not set.")
                raise StopIteration

        # 7. Return the full batch
        print(f"[DEBUG] Yielding batch of size: {len(batch)}")
        return np.stack(batch)

    def __iter__(self):
        """
        Initialize the iterator.

        Returns:
            self: The iterator object.
        """
        # 1. Call the parent class's __iter__ method
        super().__iter__()
        # 2. Return self as the iterator
        return self

    def __del__(self):
        """
        Cleanup method called when the object is about to be destroyed.

        This method ensures that all asynchronous operations are properly stopped
        and resources are released.
        """
        # 1. Check if stop_event attribute exists and set it
        if hasattr(self, "stop_event"):
            self.stop_event.set()

        # 2. Check if the event loop exists and is running
        if hasattr(self, "loop") and self.loop.is_running():
            # 2a. Stop the event loop
            self.loop.call_soon_threadsafe(self.loop.stop)
            # 2b. Wait for the fetch thread to complete
            self.fetch_thread.join()
            # 2c. Close the event loop
            self.loop.close()
