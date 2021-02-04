from torch.utils.data import Dataset
from transformers.tokenization_bert import BertTokenizer
from csv import DictReader
import tokenizations
from math import log
from sys import stderr


def contains_alpha(s):
    for c in s:
        if c.isalpha():
            return True
    return False


class VastReader(Dataset):
    def __init__(self,
                 main_csv,
                 token_appearances_tsv,
                 exclude_from_main=None,
                 word_importance_csv=None,
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
        if smoothing not in [None, "simple", "tf-idf"]:
            print("Error: smoothing must be one of: ['simple', 'tf-idf']", file=stderr)
            exit(1)
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

        self.load_data()

    def new_token_mapping(self, text, orig_tokens, orig_value_mapping):
        new_tokens = self.tokenizer.tokenize(text)
        old2new, new2old = tokenizations.get_alignments(orig_tokens, new_tokens)
        new_token_scores = []
        for i in range(len(new_tokens)):
            if len(new2old[i]) > 0:
                new_token_scores.append(
                    max(
                        [orig_value_mapping[j] for j in new2old[i]]
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
                        values.append(tf * idf)
                        break
        return values

    def relevance_scores(self, orig_tokens, tf_idfs):
        scores = list(tf_idfs)
        for i in range(len(orig_tokens)):
            if not contains_alpha(orig_tokens[i]):
                scores[i] = 0
        return scores

    def smooth(self, orig_value_mapping, tf_idfs):
        """Preserves the original token-value mapping (spacy tokens, rather than transformer tokens)"""
        new_values = []
        n = len(orig_value_mapping)
        if self.smoothing == "tf-idf":
            denom = 1
            for tf_idf in tf_idfs:
                denom += self.smooth_param * tf_idf
            denom *= n
            for v, tf_idf in zip(orig_value_mapping, tf_idfs):
                new_values.append(
                    (v + self.smooth_param * tf_idf * n) / denom
                )
        elif self.smoothing == "simple":
            denom = n*(1 + self.smooth_param*n)
            for v in orig_value_mapping:
                new_values.append(
                    (v + self.smooth_param * n) / denom
                )
        return new_values

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
                doc_tokens = self.tokenizer.tokenize(row["post"])
                doc_seq = self.tokenizer.convert_tokens_to_ids(doc_tokens)
                input_dict = {"topic": topic_seq, "document": doc_seq}
                self.inputs.append(input_dict)

        with open(self.word_importance_csv, "r") as f:
            reader = DictReader(f)
            for row in reader:
                self.labels.append(int(row["label"]))
                topic_tokens = self.tokenizer.tokenize("[CLS] " + row["topic"] + " [SEP]")
                topic_seq = self.tokenizer.convert_tokens_to_ids(topic_tokens)
                orig_word_weight_tuples = eval(row["weights"])
                orig_tokens = [t[0] for t in orig_word_weight_tuples]
                tf_idfs = self.tf_idfs(orig_tokens, row["topic"])
                orig_weight_mapping = [t[1] for t in orig_word_weight_tuples]
                if self.smoothing:
                    orig_weight_mapping = self.smooth(orig_weight_mapping, tf_idfs)
                doc_tokens, weights = self.new_token_mapping(row["argument"], orig_tokens, orig_weight_mapping)
                doc_seq = self.tokenizer.convert_tokens_to_ids(doc_tokens)
                relevance_scores = self.relevance_scores(orig_tokens, tf_idfs)
                relevance_scores = self.new_token_mapping(row["argument"], orig_tokens, relevance_scores)
                input_dict = {
                    "topic": topic_seq,
                    "document": doc_seq,
                    "weights": weights,
                    "relevance_scores": relevance_scores,
                }
                self.inputs.append(input_dict)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    dataset = VastReader(
        "../data/VAST/vast_train.csv",
        "../data/VAST_word_importance/token_appearances.tsv",
        exclude_from_main="../data/VAST_word_importance/special_datapoints.txt",
        word_importance_csv="../data/VAST_word_importance/processed_annotated.csv",
        smoothing="tf-idf",
    )
    print(len(dataset))
