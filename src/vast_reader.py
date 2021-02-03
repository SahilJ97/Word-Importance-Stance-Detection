import torch
from torch.utils.data import Dataset
from transformers.tokenization_bert import BertTokenizer
from csv import DictReader
import tokenizations
from math import log


class VastReader(Dataset):
    def __init__(self,
                 main_csv,
                 exclude_from_main=None,
                 word_importance_csv=None,
                 token_appearances_tsv=None,
                 smoothing=None,
                 smooth_param=.1,
                 tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
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
        self.token_appearances_tsv = token_appearances_tsv
        self.inputs, self.labels = [], []
        self.load_data()
        self.smoothing = smoothing
        self.smooth_param = smooth_param
        self.tokenizer = tokenizer

        # Count topics (for TF-IDF computations)
        all_topics = set()
        with open(main_csv, "r") as f:
            reader = DictReader(f)
            for row in reader:
                all_topics.add(row["new_topic"])
        self.n_topics = len(all_topics)

    def new_token_mapping(self, text, word_score_tuples):
        orig_tokens = [item[0] for item in word_score_tuples]
        new_tokens = self.tokenizer.tokenize(text)  # need to also convert to ids...
        old2new, new2old = tokenizations.get_alignments(orig_tokens, new_tokens)
        new_token_scores = []
        for i in range(len(new_tokens)):
            new_token_scores.append(
                max(
                    [target[1] for target in word_score_tuples[new2old[i]]]
                )
            )
        print(new_tokens, new_token_scores)
        return new_tokens, new_token_scores

    def tf_idfs(self, orig_tokens, topic):
        values = []
        for t in orig_tokens:
            with open(self.token_appearances_tsv, "r") as token_file:
                for line in token_file:
                    token, appearances = line.split("\t")
                    if token == t:
                        appearances = appearances.split(",")
                        idf = log(self.n_topics / len(set(appearances)))
                        tf = 0
                        for top in appearances:
                            if top == topic:
                                tf += 1
                        break
                values.append(tf * idf)

    def smooth(self, word_score_tuples):
        # TODO: implement simple additive smoothing and TF-IDF-weighted additive smoothing
        return word_score_tuples

    def load_data(self):
        exclude_from_main_ids = []
        if self.exclude_from_main:
            with open(self.exclude_from_main, "r") as f:
                for line in f:
                    exclude_from_main_ids.append(line.strip())

        with open(self.main_csv, "r") as f:
            reader = DictReader(f)
            for row in reader:
                if row["new_id"] in self.exclude_from_main:
                    continue
                self.labels.append(int(row["label"]))
                topic_tokens = self.tokenizer.tokenize("[CLS] " + row["topic"] + " [SEP]")
                topic_seq = self.tokenizer.convert_tokens_to_ids(topic_tokens)
                doc_tokens = self.tokenizer.tokenize(row["doc"])
                doc_seq = self.tokenizer.convert_tokens_to_ids(doc_tokens)
                input_dict = {"topic": topic_seq, "document": doc_seq, "weights": None}
                self.inputs.append(input_dict)

        with open(self.word_importance_csv, "r") as f:
            reader = DictReader(f)
            for row in reader:
                self.labels.append(int(row["label"]))
                topic_tokens = self.tokenizer.tokenize("[CLS] " + row["topic"] + " [SEP]")
                topic_seq = self.tokenizer.convert_tokens_to_ids(topic_tokens)
                doc_tokens, weights = self.new_token_mapping(row["doc"])
                doc_seq = self.tokenizer.convert_tokens_to_ids(doc_tokens)
                input_dict = {"topic": topic_seq, "document": doc_seq, "weights": weights}
                self.inputs.append(input_dict)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
