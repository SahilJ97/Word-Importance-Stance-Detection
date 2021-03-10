from abc import ABC
import torch
import torch.nn as nn
from src.models.vast_classifier import VastClassifier


class BertJoint(VastClassifier, ABC):
    def __init__(self, fix_bert=False, **kwargs):
        super(BertJoint, self).__init__(**kwargs)
        self.fix_bert = fix_bert
        self.dropout = nn.Dropout(p=.2046)
        self.hidden_layer = torch.nn.Linear(768*2, 283, bias=False)  # worked well with 566 instead of 283. make sure ok
        self.output_layer = torch.nn.Linear(283, 3)

    def to(self, *args, **kwargs):
        self.hidden_layer.to(*args, **kwargs)
        self.output_layer.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, pad_mask, doc_stopword_mask, inputs=None, inputs_embeds=None,
                use_dropout=True, token_type_ids=None):
        doc, topic = self.extract_co_embeddings(
            pad_mask=pad_mask,
            doc_stopword_mask=doc_stopword_mask,
            inputs=inputs,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids
        )
        both_embeds = torch.cat([doc, topic], dim=-1)
        if use_dropout:
            both_embeds = self.dropout(both_embeds)
        hl = self.hidden_layer(both_embeds)
        hl = torch.nn.functional.tanh(hl)
        ol = self.output_layer(hl)
        return ol
