import torch
from torch.utils.data import Dataset
from transformers.tokenization_bert import BertTokenizer
from csv import DictReader
import tokenizations


class VastReader:
    def __init__(self,
                 main_csv,
                 exclude_from_main=None,
                 word_importance_csv=None,
                 smoothing=None,
                 tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    ):
        """
        :param main_csv: Path to data CSV file
        :param exclude_from_main: Path to file containing new_id values of datapoints that should NOT be loaded from
        main_csv. If word_importance_csv is defined, all of its datapoints should be represented in this list
        :param word_importance_csv: Path to CSV file for data with word importance annotations. Word importance scores
        should be
        """
        self.main_csv = main_csv
        self.exclude_from_main = exclude_from_main  #
        self.word_importance_csv = word_importance_csv
        self.inputs, self.labels = [], []
        self.load_data()
        self.tokenizer = tokenizer

    def new_token_mapping(self, text, word_score_tuples):
        old_tokens = [item[0] for item in word_score_tuples]
        new_tokens = self.tokenizer.tokenize(text=text)
        old2new, new2old = tokenizations.get_alignments(old_tokens, new_tokens)
        new_token_scores = [word_score_tuples[new2old[i]][1] for i in range(len(new_tokens))]
        print(new_tokens, new_token_scores)
        return new_tokens, new_token_scores

    def load_data(self):
        exclude_from_main_ids = []
        if self.exclude_from_main:
            with open(self.exclude_from_main, "r") as f:
                for line in f:
                    exclude_from_main_ids.append(line.strip())


