import torch
from src.models.vast_classifier import VastClassifier
from abc import ABC


class MemoryNetwork(VastClassifier, ABC):
    def __init__(
            self,
            num_hops,
            hidden_layer_size,
            init_topic_knowledge_file,
            knowledge_transfer_scheme="projection",
            **kwargs
    ):
        super(MemoryNetwork, self).__init__(**kwargs)
        self.num_hops = num_hops
        self.knowledge_transfer_scheme = knowledge_transfer_scheme
        self.M = torch.load(init_topic_knowledge_file,)  # unsqueeze?
        self.W1 = torch.rand((768, 768),)
        self.W2 = torch.rand((768, 768),)
        if self.knowledge_transfer_scheme == "parallel":
            hl_size = 2*768
        else:
            hl_size = 768
        self.hidden_layer = torch.nn.Linear(hl_size, hidden_layer_size)
        self.output_layer = torch.nn.Linear(hidden_layer_size, self.num_labels)

    def to(self, *args, **kwargs):
        self.M = self.M.to(*args, **kwargs)
        self.W1 = self.W1.to(*args, **kwargs)
        self.W2 = self.W2.to(*args, **kwargs)
        self.hidden_layer = self.hidden_layer.to(*args, **kwargs)
        self.output_layer = self.output_layer.to(*args, **kwargs)
        return super().to(*args, **kwargs)

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

        elif self.knowledge_transfer_scheme == "parallel":
            results = []
            for h in topic_embedding, doc_embedding:
                for hop in range(self.num_hops):
                    h = shared_math(h)
                results.append(h)
            return torch.cat(results)

    def forward(self, pad_mask, doc_stopword_mask, topic_stopword_mask, inputs=None, inputs_embeds=None,
                token_type_ids=None, **kwargs):
        doc_embeddings, topic_embeddings = self.extract_co_embeddings(
            pad_mask=pad_mask,
            doc_stopword_mask=doc_stopword_mask,
            topic_stopword_mask=topic_stopword_mask,
            inputs=inputs,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids
        )

        # Knowledge transfer component
        H = []
        for doc_emb, topic_emb in zip(doc_embeddings, topic_embeddings):
            H.append(self.knowledge_transfer(topic_emb, doc_emb))
        H = torch.stack(H)

        # Synthesizing component
        hl = torch.nn.functional.relu(
            self.hidden_layer(H)  # add dropout before this? requires some hyperparam searching.
        )
        return self.output_layer(hl)
