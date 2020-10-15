from abc import ABC

import torch
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from csv import DictReader


@DatasetReader.register("vast_reader")
class VastReader(DatasetReader, ABC):
    def __init__(self, lazy=False, max_epoch_len=None, return_separate=False):
        super().__init__(lazy)
        self.token_indexers = {
            "tokens": PretrainedTransformerMismatchedIndexer(
                "bert-base-multilingual-cased",
            )
        }
        self.max_epoch_len = max_epoch_len
        self.return_separate = return_separate

    @overrides
    def _read(self, file_path):
        with open(file_path, 'r') as f:
            reader = DictReader(f)
            counter = 0
            for row in reader:
                counter += 1
                if self.max_epoch_len and counter > self.max_epoch_len:
                    break
                # Topic tokens
                tokenized_topic = eval(row["topic"])
                topic_tokens = ["[CLS]"]
                topic_tokens.extend(tokenized_topic[0] + ["[SEP]"])
                topic_tokens = [Token(t) for t in topic_tokens]

                # Document tokens
                tokenized_sentences = eval(row["text"])
                document_tokens = ["[CLS]"]
                for s in tokenized_sentences:
                    document_tokens.extend(s + ["."])
                document_tokens.append("[SEP]")
                document_tokens = [Token(t) for t in document_tokens]

                fields = {
                    'label': LabelField(row["label"])
                }
                if self.return_separate:  # produce topic and document as separate fields
                    fields['topic'] = TextField(topic_tokens, self.token_indexers)
                    fields['document'] = TextField(document_tokens, self.token_indexers)
                else:  # concatenate inputs
                    fields['text'] = TextField(topic_tokens + document_tokens[1:], self.token_indexers)

                yield Instance(fields)
