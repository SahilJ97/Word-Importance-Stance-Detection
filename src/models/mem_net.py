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
        self.M = torch.load(init_topic_knowledge_file,)
        self.W1 = torch.empty((768, 768), dtype=torch.float)
        torch.nn.init.kaiming_uniform_(self.W1, mode='fan_in', nonlinearity='relu')
        self.W2 = torch.empty((768, 768), dtype=torch.float)
        torch.nn.init.kaiming_uniform_(self.W2, mode='fan_in', nonlinearity='relu')
        if self.knowledge_transfer_scheme == "parallel":
            mem_output_size = 4*768
        else:
            mem_output_size = 768
        self.hidden_layer = torch.nn.Linear(mem_output_size, hidden_layer_size)
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
            alphas = torch.einsum('ij,bj->bi', self.W1, h_input)
            alphas = torch.einsum('ij,bj->bi', self.M, alphas)
            alphas = torch.softmax(alphas, dim=-1)  # shape: (batch_size, num_memory_slots)
            o = torch.einsum('bi,ij->bj', alphas, self.M)  # shape: (batch_size, 768)
            return torch.einsum('ij,bj->bi', self.W2, h_input) + o

        """if self.knowledge_transfer_scheme == "projection":
            h = topic_embedding
            for hop in range(self.num_hops):
                h = shared_math(h)
                h = torch.div(
                    torch.dot(h, doc_embedding),
                    torch.square(torch.norm(h))
                ) * h  # project document embedding onto h. DEFINITELY REVISE THIS SCHEME!!!
            return h"""

        if self.knowledge_transfer_scheme == "parallel":
            results = [topic_embedding, doc_embedding]
            for h in topic_embedding, doc_embedding:
                for hop in range(self.num_hops):
                    h = shared_math(h)
                results.append(h)
            return torch.cat(results, dim=-1)

    def forward(self, pad_mask, doc_stopword_mask, inputs=None, inputs_embeds=None,
                token_type_ids=None, **kwargs):
        doc_embeddings, topic_embeddings = self.extract_co_embeddings(
            pad_mask=pad_mask,
            doc_stopword_mask=doc_stopword_mask,
            inputs=inputs,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids
        )

        # Knowledge transfer component
        H = self.knowledge_transfer(doc_embeddings, topic_embeddings)

        # Synthesizing component
        hl = torch.nn.functional.relu(
            self.hidden_layer(H)  # add dropout before this? requires some hyperparam searching.
        )
        return self.output_layer(hl)
