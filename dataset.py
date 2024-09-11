# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

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
import requests
import numpy as np
from tqdm import tqdm
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

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
            tokenizer: AutoTokenizer=None,
            pack_samples: bool=False,
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

    def fetch_data_for_pages(self, pages):
        """
        Set the pages to be used to fill the buffer. Then fetch the page data
        to the buffer.
        """

        self.pages = pages

        # Empty the buffer if it is not.
        self.buffer = []

        for page in self.pages:
            self._fetch_data_for_page(page)


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
        This methods pulls one page from `self.buffer`, pads it and pushs
        it to the `self.padded_buffer`.
        """

        while (
                self.buffer
                and len(self.padded_buffer) < self.sequence_length
        ):

            input_ids = []

            # search for EOS token index and cut the buffer at it.
            EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
            input_ids = self.buffer[:EOS_index+1]
            self.buffer =self.buffer[EOS_index+1:]

            self.used_buffer += input_ids

            # Add to padded buffer without the EOS token.
            self.padded_buffer += input_ids[:-1]

            # Pad
            self.padded_buffer += [self.tokenizer.eos_token_id] * self._get_pad_size(input_ids=input_ids[:-1])

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

    name: str = "HuggingFaceFW/fineweb-edu-score-2"
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"

    retry_limit: int = 10  # Number of retries
    retry_delay: int = 5  # Seconds to wait between retries
    num_rows_per_page: int = 100
    
    
    @staticmethod
    def next_pages( offset: int, n_pages: int, seed: str, num_rows_per_page:int = 100 ):
        configs_data = SubsetFineWebEdu2Loader.fetch_dataset_configs()
        rng = np.random.default_rng(hash(seed) & 0xffffffff)  # Create a generator with a seed
        rng.bit_generator.advance( offset )  # Efficiently skip ahead `n` steps
        result = []
        for _ in range( n_pages ):
            config = rng.choice( list(configs_data.keys() ))
            choice = rng.integers(0, configs_data[config]['num_rows'] - 1 - num_rows_per_page)
            result.append( (str(config), int(choice), configs_data[config]['split'] ) )
        return result

    def __init__(
            self,
            batch_size=None,
            sequence_length=None,
            num_pages = None,
            pages_info = None,
            tokenizer: AutoTokenizer=None,
            pack_samples: bool=False,
    ):
        super().__init__(batch_size,
                         sequence_length,
                         num_pages,
                         tokenizer,
                         pack_samples)

        # Get the dataset configs and their row sizes
        self.configs_data = SubsetFineWebEdu2Loader.fetch_dataset_configs()
        
        if pages_info != None:
            self._fetch( pages_info )
            
        elif self.num_pages:
            self._fetch_data_to_buffer(self.num_pages)
            
    def _fetch( self, page_info: typing.Tuple[ str, int, str ] ):
        
        self.pages = page_info
        for (config_name, page, split) in tqdm(self.pages):
            # Create the request parameters
            params = dict(dataset=self.name,
                        config=config_name,
                        split=split,
                        offset=page,
                        limit=self.num_rows_per_page
            )
            try:
                response = requests.get(self.rows_base_url, params=params)
                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
                for row in response.json()["rows"]:
                    content = row["row"]["text"]

                    # get the tokenized and encoded sample
                    input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += input_ids
                    self.buffer += [self.tokenizer.eos_token_id]

                response.close()

            except requests.exceptions.RequestException as e:

                response.close()
                attempts += 1
                if attempts < num_pages * self.retry_limit:
                    pass

                else:
                    raise
            

    def _fetch_data_to_buffer(self, num_pages):
        """
        Randomly sample pages and add their data to the buffer.
        If a page is inaccessible, another one is sampled.
        this method sets the `pages` property
        """

        self.pages = []
        attempts = 0

        while len(self.pages) < num_pages:

            # randomly sample one page
            config_name, page, split = self.get_random_pages(num_pages = 1)[0]

            # Create the request parameters
            params = dict(dataset=self.name,
                          config=config_name,
                          split=split,
                          offset=page,
                          limit=self.num_rows_per_page
            )

            try:
                response = requests.get(self.rows_base_url, params=params)

                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                # Add the page since the request was successful
                self.pages.append((config_name, page, split))

                for row in response.json()["rows"]:
                    content = row["row"]["text"]

                    # get the tokenized and encoded sample
                    input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += input_ids
                    self.buffer += [self.tokenizer.eos_token_id]

                response.close()

            except requests.exceptions.RequestException as e:

                response.close()
                attempts += 1
                if attempts < num_pages * self.retry_limit:
                    pass

                else:
                    raise

    def fetch_data_to_rows(self, num_pages):

        rows = []
        attempts = 0
        num_downloaded_pages = 0

        while num_downloaded_pages < num_pages:

            # randomly sample one page
            config_name, page, split = self.get_random_pages(num_pages = 1)[0]

            # Create the request parameters
            params = dict(dataset=self.name,
                          config=config_name,
                          split=split,
                          offset=page,
                          limit=self.num_rows_per_page
            )

            try:
                response = requests.get(self.rows_base_url, params=params)

                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                num_downloaded_pages += 1

                for row in response.json()["rows"]:
                    rows.append(row["row"]["text"])

            except requests.exceptions.RequestException as e:
                attempts += 1
                if attempts < num_pages * self.retry_limit:
                    pass

                else:
                    raise


        return rows

    def get_random_pages(self, num_pages):
        """
        Randomly sample one page.
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
        Each page as a single string instead of a tuple
        """

        page_names = []

        if hasattr(self, 'pages'):
            page_names = [f'{cfg_name}_{num_rows}_{split}' for
                           cfg_name, num_rows, split in self.pages]

        return page_names

    @staticmethod
    def fetch_dataset_configs() -> typing.Dict[str, typing.Dict]:
        """
        Fetch the different dump names, aka configs, aka samples, of the
        dataset.
        The returned value is a dictionary with dump names as keys and
        a dict of the number of rows and the split as values.
        """
        # Request parameters
        params = dict(
            dataset = SubsetFineWebEdu2Loader.name
            )

        attempt = 0
        while attempt < SubsetFineWebEdu2Loader.retry_limit:
            try:
                response = requests.get(SubsetFineWebEdu2Loader.size_base_url, params=params)
                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                # Extract the configs dict
                configs_dict = response.json()['size']['splits']

                # Now create a dict with config names (except 'default') as
                # keys, and the number of rows as values
                configs_data = {entry['config']: {'num_rows': entry['num_rows'] ,
                                                  'split': entry['split']}
                                for entry in configs_dict
                                if entry['config'] != 'default'
                                }

                return configs_data

            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt < SubsetFineWebEdu2Loader.retry_limit:
                    time.sleep(SubsetFineWebEdu2Loader.retry_delay)  # Wait before the next retry
                else:
                    raise

    def _fetch_data_for_page(self, page):

        retry_limit = 10

        attempt = 0
        while attempt < retry_limit:
            config_name, page, split = page

            # Create the request parameters
            params = dict(dataset=self.name,
                          config=config_name,
                          split=split,
                          offset=page,
                          limit=self.num_rows_per_page
            )

            try:

                response = requests.get(self.rows_base_url, params=params)

                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                for row in response.json()["rows"]:
                    content = row["row"]["text"]
                    input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += input_ids
                    self.buffer += [self.tokenizer.eos_token_id]

                break  # If the request was successful, break out of the retry loop

            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    raise