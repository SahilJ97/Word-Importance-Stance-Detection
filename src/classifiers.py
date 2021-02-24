from abc import ABC
import torch
import torch.nn as nn
from transformers import BertModel


class VastClassifier(nn.Module, ABC):  # attribution usage occurs in loss function!
    num_labels = 3

    def __init__(self):
        super().__init__()


class BaselineBert(VastClassifier, ABC):
    def __init__(self, pretrained_model="bert-base-uncased", topic_len=5):
        super(BaselineBert, self).__init__()
        self.topic_len = topic_len
        self.bert_model = BertModel.from_pretrained(
            pretrained_model,
            num_labels=self.num_labels,
        )
        self.dropout = nn.Dropout(p=.2046)
        self.hidden_layer = torch.nn.Linear(768*2, 283)
        self.output_layer = torch.nn.Linear(283, 3)

    def to(self, *args, **kwargs):
        self.bert_model = self.bert_model.to(*args, **kwargs)
        self.hidden_layer.to(*args, **kwargs)
        self.output_layer.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, pad_mask, inputs=None, inputs_embeds=None, use_dropout=True):
        if inputs is None and inputs_embeds is None:
            raise ValueError("Either inputs or inputs_embeds must be provided")
        if inputs is not None:
            inputs_embeds = self.get_inputs_embeds(inputs)
        with torch.no_grad():  # leave BERT fixed
            last_hidden_state, pooler_outputs = self.bert_model.forward(inputs_embeds=inputs_embeds)
        topic_token_counts = torch.sum(pad_mask[:, 1:1 + self.topic_len], dim=-1)  # skip first token ([CLS])
        doc_token_counts = torch.sum(pad_mask[:, 2+self.topic_len:], dim=-1)  # skip first token after topic ([SEP])
        pad_mask = torch.unsqueeze(pad_mask, dim=-1)
        last_hidden_state = pad_mask * last_hidden_state  # zero the embeddings corresponding to [PAD] tokens
        topic_embeds = last_hidden_state[:, 1:1+self.topic_len, :]
        doc_embeds = last_hidden_state[:, 2+self.topic_len:, :]
        topic = torch.sum(topic_embeds, dim=1)
        topic = topic / topic_token_counts[:, None]
        doc = torch.sum(doc_embeds, dim=1)
        doc = doc / doc_token_counts[:, None]
        both_embeds = torch.cat([topic, doc], dim=-1)
        if use_dropout:
            both_embeds = self.dropout(both_embeds)
        hl = self.hidden_layer(both_embeds)
        hl = torch.nn.functional.relu(hl)
        ol = self.output_layer(hl)
        #probs = torch.nn.functional.softmax(ol, dim=-1)
        return ol

    def get_inputs_embeds(self, inputs):
        inputs = inputs.long()
        token_type_ids = torch.zeros_like(inputs, dtype=torch.long)
        return self.bert_model.embeddings(input_ids=inputs, token_type_ids=token_type_ids)


"""class MemoryNetwork(VastClassifier, ABC):
    def __init__(
            self,
            vocab,
            num_hops,
            text_embedding_size,
            hidden_layer_size,
            init_topic_knowledge_file,
            knowledge_transfer_scheme="projection",
            pretrained_model="bert-base-multilingual-cased"
    ):
        super().__init__(vocab)
        self.embedder = BertModel.from_pretrained(
            pretrained_model,
        )#.to(DEVICE)
        self.num_hops = num_hops
        self.knowledge_transfer_scheme = knowledge_transfer_scheme
        self.M = torch.load(init_topic_knowledge_file,)# map_location=DEVICE)
        self.W1 = torch.rand((text_embedding_size, text_embedding_size),)# device=DEVICE)
        self.W2 = torch.rand((text_embedding_size, text_embedding_size),)# device=DEVICE)
        if self.knowledge_transfer_scheme == "parallel":
            hl_size = 2*text_embedding_size
        else:
            hl_size = text_embedding_size
        self.hidden_layer = torch.nn.Linear(hl_size, hidden_layer_size)#.to(DEVICE)
        self.output_layer = torch.nn.Linear(hidden_layer_size, self.num_labels)#.to(DEVICE)

    def knowledge_transfer(self, topic_embedding, doc_embedding):
        def shared_math(h_input):
            alphas = self.W1 @ h_input
            alphas = self.M @ alphas
            alphas = torch.softmax(alphas, dim=-1)  # shape: (num_memory_slots)
            o = torch.zeros(h_input.size())
            for mem_slot_n in range(len(alphas)):
                attended = alphas[mem_slot_n] * self.M[mem_slot_n]
                o += attended
            return self.W2 @ h_input + o

        if self.knowledge_transfer_scheme == "projection":
            h = topic_embedding
            for hop in range(self.num_hops):
                h = shared_math(h)
                h = torch.div(
                    torch.dot(h, doc_embedding),
                    torch.square(torch.norm(h))
                ) * h  # project document embedding onto h
            return h

        elif self.knowledge_transfer_scheme == "parallel":
            results = []
            for h in topic_embedding, doc_embedding:
                for hop in range(self.num_hops):
                    h = shared_math(h)
                results.append(h)
            return torch.cat(results)

    def forward(self, topic, document, label):  # also try with co-encoding, and using document in mem network
        topic_input = torch.unsqueeze(topic["tokens"]["token_ids"], 1)#.to(DEVICE)
        document_input = torch.unsqueeze(document["tokens"]["token_ids"], 1)#.to(DEVICE)
        batch_size = list(topic_input.size())[0]

        # Iterate through batch, generating embeddings
        topic_embeddings = []
        document_embeddings = []
        for i in range(batch_size):  # want co-embeddings! refactor...
            topic_embeddings.append(
                bert_embedding(self.embedder, topic_input[i])
            )
            document_embeddings.append(
                bert_embedding(self.embedder, document_input[i])
            )
        topic_embeddings = torch.stack(topic_embeddings)
        document_embeddings = torch.stack(document_embeddings)

        # Knowledge transfer component
        H = []
        for i in range(batch_size):
            H.append(self.knowledge_transfer(topic_embeddings[i], document_embeddings[i]))
        H = torch.stack(H)

        # Synthesizing component
        hl = torch.nn.functional.relu(
            self.hidden_layer(H)
        )
        probs = torch.nn.functional.softmax(
            self.output_layer(hl), dim=-1
        )
        loss = torch.nn.functional.cross_entropy(probs, label)
        for metric in self.metrics.values():
            metric(probs, label)
        return {'loss': loss, 'probs': probs}"""
