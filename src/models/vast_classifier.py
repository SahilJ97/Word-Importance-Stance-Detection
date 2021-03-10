from abc import ABC
import torch.nn as nn
import torch
from transformers import BertModel


class VastClassifier(nn.Module, ABC):
    num_labels = 3

    def __init__(self, doc_len=250, pretrained_model="bert-base-uncased"):
        super().__init__()
        self.doc_len = doc_len
        self.bert_model = BertModel.from_pretrained(
            pretrained_model,
            num_labels=self.num_labels,
        )

    def to(self, *args, **kwargs):
        self.bert_model = self.bert_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def extract_co_embeddings(self, pad_mask, doc_stopword_mask, topic_stopword_mask, inputs=None, inputs_embeds=None,
                              token_type_ids=None):
        # Inputs/inputs_embeds are organized as "[CLS] document [SEP] topic [SEP]"
        if inputs is None and inputs_embeds is None:
            raise ValueError("Either inputs or inputs_embeds must be provided")
        if inputs is not None:
            inputs_embeds = self.get_inputs_embeds(inputs)
        if hasattr(self, "fix_bert") and self.fix_bert:
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
        topic_embeds = torch.unsqueeze(topic_stopword_mask, dim=-1) * topic_embeds
        doc = torch.sum(doc_embeds, dim=1) / doc_token_counts[:, None]  # same idea for document embeddings
        topic = torch.sum(topic_embeds, dim=1) / topic_token_counts[:, None]  # avg of non-zeroed topic tokens
        return doc, topic

    def get_inputs_embeds(self, inputs):
        inputs = inputs.long()
        token_type_ids = torch.zeros_like(inputs, dtype=torch.long)
        return self.bert_model.embeddings(input_ids=inputs, token_type_ids=token_type_ids)
