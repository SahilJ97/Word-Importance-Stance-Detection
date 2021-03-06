from abc import ABC
import torch
import torch.nn as nn
from transformers import BertModel


class VastClassifier(nn.Module, ABC):
    num_labels = 3

    def __init__(self):
        super().__init__()


class BertJoint(VastClassifier, ABC):
    def __init__(self, pretrained_model="bert-base-uncased", doc_len=205, fix_bert=True):
        super(BertJoint, self).__init__()
        self.doc_len = doc_len
        self.fix_bert = fix_bert
        self.bert_model = BertModel.from_pretrained(
            pretrained_model,
            num_labels=self.num_labels,
        )
        self.dropout = nn.Dropout(p=.2046)
        self.hidden_layer = torch.nn.Linear(768*2, 566, bias=False)  # try: 283, 566,
        self.output_layer = torch.nn.Linear(566, 3)

    def to(self, *args, **kwargs):
        self.bert_model = self.bert_model.to(*args, **kwargs)
        self.hidden_layer.to(*args, **kwargs)
        self.output_layer.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, pad_mask, doc_stopword_mask, topic_stopword_mask, inputs=None, inputs_embeds=None,
                use_dropout=True, token_type_ids=None):
        # inputs/inputs_embeds are organized as "[CLS] document [SEP] topic [SEP]"
        if inputs is None and inputs_embeds is None:
            raise ValueError("Either inputs or inputs_embeds must be provided")
        if inputs is not None:
            inputs_embeds = self.get_inputs_embeds(inputs)
        if self.fix_bert:
            with torch.no_grad():
                last_hidden_state, pooler_outputs = self.bert_model.forward(
                    inputs_embeds=inputs_embeds,
                    attention_mask=pad_mask,
                    token_type_ids=token_type_ids
                )
        else:
            last_hidden_state, pooler_outputs = self.bert_model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=pad_mask,
                token_type_ids=token_type_ids
            )
        doc_token_counts = torch.sum(
            pad_mask[:, 1:self.doc_len + 1:] * doc_stopword_mask,
            dim=-1
        )
        topic_token_counts = torch.sum(
            pad_mask[:, self.doc_len + 2:-1] * topic_stopword_mask,
            dim=-1
        )
        last_hidden_state = torch.unsqueeze(pad_mask, dim=-1) * last_hidden_state
        doc_embeds = last_hidden_state[:, 1:self.doc_len + 1:]
        doc_embeds = torch.unsqueeze(doc_stopword_mask, dim=-1) * doc_embeds
        topic_embeds = last_hidden_state[:, self.doc_len + 2:-1]
        doc = torch.sum(doc_embeds, dim=1) / doc_token_counts[:, None]  # same idea for document embeddings
        topic = torch.sum(topic_embeds, dim=1) / topic_token_counts[:, None]  # avg of non-zeroed topic tokens
        both_embeds = torch.cat([doc, topic], dim=-1)
        if use_dropout:
            both_embeds = self.dropout(both_embeds)
        hl = self.hidden_layer(both_embeds)
        hl = torch.nn.functional.tanh(hl)
        ol = self.output_layer(hl)
        return ol

    def get_inputs_embeds(self, inputs):
        inputs = inputs.long()
        token_type_ids = torch.zeros_like(inputs, dtype=torch.long)
        return self.bert_model.embeddings(input_ids=inputs, token_type_ids=token_type_ids)
