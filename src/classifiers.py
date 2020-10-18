from abc import ABC
import torch
from allennlp.models import Model
from transformers import BertForSequenceClassification, BertModel
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
from src.utils import bert_embedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VastClassifier(Model, ABC):
    def __init__(self, vocab):
        super().__init__(vocab)
        print("Using CUDA? ", DEVICE == "cuda")
        self.num_labels = vocab.get_vocab_size("labels")
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "f": FBetaMeasure(),
        }

    def get_metrics(self, reset=False):
        metric_vals = {}
        for metric_name, metric in self.metrics.items():
            val = metric.get_metric()
            if isinstance(val, dict):
                for sub_metric_name, sub_val in val.items():
                    if isinstance(sub_val, list):
                        for i in range(len(sub_val)):
                            metric_vals[f"{sub_metric_name}_{i}"] = sub_val[i]
                    else:
                        metric_vals[sub_metric_name] = sub_val
            else:
                metric_vals[metric_name] = val
        return metric_vals


@Model.register('baseline_mbert')
class BaselineMBert(VastClassifier, ABC):
    def __init__(self, vocab, pretrained_model="bert-base-multilingual-cased"):
        super().__init__(vocab)
        self.bert_classifier = BertForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=self.num_labels,
        )
        self.bert_classifier = self.bert_classifier.to(DEVICE)

    def forward(self, text, label):
        inputs = text["tokens"]["token_ids"]
        logits = self.bert_classifier(inputs)
        probs = torch.nn.functional.softmax(logits[0], dim=-1)
        loss = torch.nn.functional.cross_entropy(probs, label)
        for metric in self.metrics.values():
            metric(probs, label)
        return {'loss': loss, 'probs': probs}


@Model.register('memory_network')
class MemoryNetwork(VastClassifier, ABC):
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
        ).to(DEVICE)
        self.num_hops = num_hops
        self.knowledge_transfer_scheme = knowledge_transfer_scheme
        self.M = torch.load(init_topic_knowledge_file, map_location=DEVICE)
        self.W1 = torch.rand((text_embedding_size, text_embedding_size), device=DEVICE)
        self.W2 = torch.rand((text_embedding_size, text_embedding_size), device=DEVICE)
        if self.knowledge_transfer_scheme == "parallel":
            hl_size = 2*text_embedding_size
        else:
            hl_size = text_embedding_size
        self.hidden_layer = torch.nn.Linear(hl_size, hidden_layer_size).to(DEVICE)
        self.output_layer = torch.nn.Linear(hidden_layer_size, self.num_labels).to(DEVICE)

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
        topic_input = torch.unsqueeze(topic["tokens"]["token_ids"], 1)
        document_input = torch.unsqueeze(document["tokens"]["token_ids"], 1)
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
        return {'loss': loss, 'probs': probs}
