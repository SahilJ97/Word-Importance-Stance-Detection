import torch


def bert_embedding(bert_model, x):
    with torch.no_grad():
        outputs = bert_model(x)
    return outputs[1].squeeze()


"""def bert_co_embeddings(bert_model, X):
    embeddings = []
    for x in X:
"""