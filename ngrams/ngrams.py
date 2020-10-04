from typing import List, Dict  # type hint
from collections import Counter, defaultdict  # frequency dictionary
from itertools import islice  # itertools for nth word start
import logging  # log status

from tqdm import tqdm  # for loop progress bar


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Ngrams:
    """
    Ngrams abstract class
    """

    def __init__(self,
                 seq_list: List[str] = None,
                 n: int = 2) -> None:
        """
        Initialize params and data
        :param seq_list: List of preprocessed string data to process ngram
        :param n: Integer to make ngram
        """
        self.n = n
        logger.info(f'Generate {self.n}-gram frequency table')
        self.numerator = self.count_ngrams(seq_list=seq_list,
                                           n=self.n)
        if self.n > 1:
            logger.info(f'Generate {self.n - 1}-gram frequency table')
            self.denominator = self.count_ngrams(seq_list=seq_list,
                                                 n=self.n - 1)

    def count_ngrams(self,
                     seq_list: List[str] = None,
                     n: int = 2) -> Dict:
        """
        Return ngram frequency table
        :param seq_list: List of preprocessed string data to process ngram
        :param n: Integer to make ngram
        :return: Ngram frequency table
        """
        raise NotImplementedError()

    def prob_ngrams(self,
                    seq: str = None,
                    eps: float = 1e-8) -> float:
        """
        Return probability of sequence with ngram model
        :param seq: Preprocessed string data to calculate probability
        :return: Probability of sequence with ngram
        """
        prob = 1
        logger.info(f'Calculate probability with {self.n}-gram model')
        if self.n == 1:
            for e in zip(*[islice(seq.split(' '), i, None) for i in range(self.n)]):
                prob *= self.numerator[e] / (sum(self.numerator.values()) + eps)
        else:
            for e in zip(*[islice(seq.split(' '), i, None) for i in range(self.n)]):
                prob *= self.numerator[e] / (self.denominator[tuple(list(e)[:-1])] + eps)
        return prob


class NgramsCounter(Ngrams):
    """
    Ngrams with Counter objects
    References
    1. https://stackoverflow.com/questions/43232519/python-count-tuples-occurence-in-list
    2. https://stackoverflow.com/questions/2522152/python-is-a-dictionary-slow-to-find-frequency-of-each-character
    """

    def __init__(self,
                 seq_list: List[str] = None,
                 n: int = 2) -> None:
        """
        Initialize params and data
        :param seq_list: List of preprocessed string data to process ngram
        :param n: Integer to make ngram
        """
        super().__init__(seq_list=seq_list, n=n)

    def count_ngrams(self,
                     seq_list: List[str] = None,
                     n: int = 2) -> Dict:
        """
        Return ngram frequency table with dictionary
        :param seq_list: List of preprocessed string data to process ngram
        :param n: Integer to make ngram
        :return: Ngram frequency table with dictionary
        """
        freq_table = Counter()
        for seq in tqdm(seq_list, desc='processing'):
            freq_table.update(Counter(zip(*[islice(seq.split(' '), i, None) for i in range(n)])))
        return freq_table


class NgramsDefaultdict(Ngrams):
    """
    Ngrams with Defualtdict objects
    References
    1. https://stackoverflow.com/questions/2522152/python-is-a-dictionary-slow-to-find-frequency-of-each-character
    """

    def __init__(self,
                 seq_list: List[str] = None,
                 n: int = 2) -> None:
        """
        Initialize params and data
        :param seq_list: List of preprocessed string data to process ngram
        :param n: Integer to make ngram
        """
        super().__init__(seq_list=seq_list, n=n)

    def count_ngrams(self,
                     seq_list: List[str] = None,
                     n: int = 2) -> Dict:
        """
        Return ngram frequency table with dictionary
        :param seq_list: List of preprocessed string data to process ngram
        :param n: Integer to make ngram
        :return: Ngram frequency table with dictionary
        """
        freq_table = defaultdict(int)
        for seq in tqdm(seq_list, desc='processing'):
            for ngram in zip(*[islice(seq.split(' '), i, None) for i in range(n)]):
                freq_table[ngram] += 1
        return freq_table
