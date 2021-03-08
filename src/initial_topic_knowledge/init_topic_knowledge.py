from sys import argv
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from csv import DictReader
import torch
from kmeans_pytorch import kmeans
import os
from src.models.bert_joint import BertJoint
from src.vast_reader import VastReader
from src.utils import get_pad_mask

N_SLOTS, OUTPUT_FILE = argv[1], argv[2]
N_SLOTS = int(N_SLOTS)
EMBEDDINGS_FILE = "all_topic_embeddings.pt"

train_set = VastReader(
        "../../data/VAST/vast_train.csv",
        "../../data/VAST_word_importance/token_appearances.tsv",
        exclude_from_main="../../data/VAST_word_importance/special_datapoints.txt",
        word_importance_csv="../../data/VAST_word_importance/processed_annotated.csv",
        smoothing=None,
        relevance_type="binary"
    )
model = BertJoint(doc_len=train_set.doc_len, fix_bert=True)

token_type_ids = [0 for _ in range(train_set.doc_len + 2)] + [1 for _ in range(train_set.topic_len + 1)]
token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)


if __name__ == "__main__":
    # Load embeddings if embeddings_file exists, otherwise compute them
    if os.path.exists(EMBEDDINGS_FILE):
        topic_embeddings = torch.load(EMBEDDINGS_FILE)
    else:
        topic_embeddings = []
        for item in train_set:
            print(item)
            inputs, _, doc_stopword_mask, topic_stopword_mask, _ = item
            inputs = torch.unsqueeze(inputs, dim=0)
            with torch.no_grad():
                doc, topic = model.extract_co_embeddings(
                    get_pad_mask(inputs),
                    torch.unsqueeze(doc_stopword_mask, dim=0),
                    torch.unsqueeze(topic_stopword_mask, dim=0),
                    inputs=inputs,
                    token_type_ids=token_type_ids
                )
            topic_embeddings.append(topic)
        topic_embeddings = torch.stack(topic_embeddings)
        torch.save(topic_embeddings, EMBEDDINGS_FILE)

    # Cluster
    cluster_ids_x, cluster_centers = kmeans(
        X=topic_embeddings, num_clusters=N_SLOTS, distance='euclidean'
    )
    torch.save(cluster_centers, OUTPUT_FILE)

    # Compute SSE
    cluster_ids_x = list(cluster_ids_x.numpy())
    sse = torch.tensor(0.)
    for i in range(len(cluster_ids_x)):
        center = cluster_centers[cluster_ids_x[i]]
        embedding = topic_embeddings[i]
        sse += torch.dist(embedding, center).pow(2)
    print(f"SSE: {sse.numpy()}")
