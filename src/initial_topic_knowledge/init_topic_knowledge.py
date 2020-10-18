from sys import argv
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from csv import DictReader
import torch
from kmeans_pytorch import kmeans
import os
from src import utils

BERT_MODEL = "bert-base-multilingual-cased"
EMBEDDING_TYPE, N_SLOTS, INPUT_FILE, OUTPUT_FILE = argv[1:5]
N_SLOTS = int(N_SLOTS)


def get_text():
    topic_only = (EMBEDDING_TYPE == "topic")
    with open(INPUT_FILE, 'r') as f:
        topics_seen = set()
        reader = DictReader(f)
        for row in reader:
            txt = "[CLS] " + row["new_topic"] + " [SEP]"
            if not topic_only:
                txt += " " + row["post"] + " [SEP]"
            txt = txt.lower()
            if not topic_only or txt not in topics_seen:
                topics_seen.add(txt)
                yield txt


if __name__ == "__main__":
    if EMBEDDING_TYPE == "topic":
        embeddings_file = "all_topic_embeddings.pt"
    else:
        embeddings_file = "all_top_doc_embeddings.pt"

    # Load embeddings if embeddings_file exists, otherwise compute them
    if os.path.exists(embeddings_file):
        embeddings = torch.load(embeddings_file)
    else:
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        model = BertModel.from_pretrained(BERT_MODEL)
        print("Embedding topics...")
        embeddings = []
        for text in get_text():
            true_tokens = tokenizer.tokenize(text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(true_tokens)
            input_ids = torch.tensor(indexed_tokens).unsqueeze(0)
            """with torch.no_grad():
                outputs = model(input_ids)
            last_hidden_state = outputs[0][-1].squeeze()  # use utils.bert_embedding???
            text_embedding = torch.mean(last_hidden_state, 0)"""
            text_embedding = utils.bert_embedding(model, input_ids)  # may need to rerun clustering now...
            embeddings.append(text_embedding)
        embeddings = torch.stack(embeddings)
        torch.save(embeddings, embeddings_file)

    # Cluster
    cluster_ids_x, cluster_centers = kmeans(
        X=embeddings, num_clusters=N_SLOTS, distance='euclidean'
    )
    torch.save(cluster_centers, OUTPUT_FILE)

    # Compute SSE
    cluster_ids_x = list(cluster_ids_x.numpy())
    sse = torch.tensor(0.)
    for i in range(len(cluster_ids_x)):
        center = cluster_centers[cluster_ids_x[i]]
        embedding = embeddings[i]
        sse += torch.dist(embedding, center).pow(2)
    print(f"SSE: {sse.numpy()}")
