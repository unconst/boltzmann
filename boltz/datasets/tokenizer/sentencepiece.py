from typing import List

from sentencepiece import SentencePieceProcessor

from .tokenizer import Tokenizer
from boltz.common import logger


class SentencePieceTokenizer(Tokenizer):
    """
    Tokenizing and encoding/decoding text based on a SentencePiece model.

    Args:
        tokenizer_path (str): The path to the SentencePiece model file.
    """

    def __init__(self, tokenizer_path: str):
        super().__init__(tokenizer_path)
        # reload tokenizer
        self.sp_model = SentencePieceProcessor(model_file=tokenizer_path)

        # BOS / EOS token IDs
        self._n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"SentencePieceTokenizer built: #words {self._n_words}, BOS ID {self.bos_id}, EOS ID {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(
        self,
        s: str,
        truncation: bool = False,
        max_length: int = None,
        bos: bool = False,
        eos: bool = False
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            truncation (bool): Whether to truncate the sequence to `max_length`.
            max_length (int): The maximum length of the sequence after truncation.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if truncation and max_length is not None:
            t = t[:max_length]
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)

    @property
    def eos_token_id(self) -> int:
        return self.eos_id


    @property
    def n_words(self) -> int:
        return self._n_words

    @property
    def pad_token_id(self) -> int:
        return self.pad_id

    @pad_token_id.setter
    def pad_token_id(self, value: int):
        self.pad_id = value