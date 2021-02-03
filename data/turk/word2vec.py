from gensim.models import Word2Vec
from gensim import downloader
from csv import DictReader
from scipy.spatial.distance import cosine
from numpy import sum
import numpy as np

# first filter out bad workers?

glove_vectors = downloader.load('glove-twitter-50')
print(glove_vectors)

if __name__ == "__main__":
    topics = {}
    with open("processed_annotated.csv", "r") as in_file:
        reader = DictReader(in_file)
        for row in reader:
            weights = eval(row["weights"])
            topic = row["topic"]
            selected_embedding = None
            for word, weight in weights.values():
                word = word.lower()
                if weight == 0:
                    continue
                try:
                    word_embedding = np.array(glove_vectors[word])
                except KeyError:
                    continue
                if selected_embedding is not None:
                    selected_embedding += word_embedding * weight
                else:
                    selected_embedding = word_embedding * weight
            topics.setdefault(topic, []).append(selected_embedding)

    topic_embeddings = []
    selected_embeddings = []

    for topic, selected_embedding in topics.items():
        if len([s for s in selected_embedding if s is not None]) == 0:
            continue
        selected_embedding = sum([s for s in selected_embedding if s is not None], axis=0)
        topic = topic.lower().split()
        topic_word_embeddings = []
        for word in topic:
            try:
                topic_word_embeddings.append(glove_vectors[word])
            except KeyError:
                continue
        if len(topic_word_embeddings) > 0:
            topic_embedding = sum(topic_word_embeddings, axis=0)
            topic_embeddings.append(topic_embedding)
            selected_embeddings.append(selected_embedding)

    topic_distances = []
    selected_distances = []
    for i in range(len(topic_embeddings)):
        for j in range(i+1, len(topic_embeddings)):
            topic_distances.append(cosine(topic_embeddings[i], topic_embeddings[j]))
            selected_distances.append(cosine(selected_embeddings[i], selected_embeddings[j]))
    print(topic_distances)
    print(selected_distances)
    print(np.corrcoef(topic_distances, selected_distances))