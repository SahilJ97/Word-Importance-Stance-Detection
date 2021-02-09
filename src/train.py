import torch
from attributionpriors.pytorch_ops import AttributionPriorExplainer
from sys import argv
from .vast_reader import VastReader
from .classifiers import BaselineBert
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy, one_hot
from torch.optim import Adam

smoothing, smooth_param, relevance_type, use_prior, batch_size, learn_rate, k = argv[1:8]
if smoothing == "none":
    smoothing = None

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 20


def train():
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    for epoch in range(NUM_EPOCHS):
        print(f"\tBeginning epoch {epoch}...")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            input_seq = inputs["input_seq"]
            input_seq = input_seq.to(DEVICE)
            labels = labels.to(DEVICE)
            labels = one_hot(labels, num_classes=3).float()
            optimizer.zero_grad()
            outputs = model(input_seq)
            loss = binary_cross_entropy(outputs, labels)
            if use_prior:
                for i in range(len(input_seq)):
                    if inputs["weights"][i]:
                        ip_seq = inputs["input_seq"][i]
                        ip_seq = torch.unsqueeze(ip_seq, dim=0)
                        expected_gradients = explainer.shap_values(model, ip_seq)[0]
                        print(expected_gradients)

            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    train_set = VastReader(
        "../data/VAST/vast_train.csv",
        "../data/VAST_word_importance/token_appearances.tsv",
        exclude_from_main="../data/VAST_word_importance/special_datapoints.txt",
        word_importance_csv="../data/VAST_word_importance/processed_annotated.csv",
        smoothing=smoothing,
        smooth_param=smooth_param,
        relevance_type=relevance_type
    )
    explainer = AttributionPriorExplainer(train_set, batch_size=1, k=k)  # don't use k = 1 because few examples with labels?
    dev_set = VastReader("../data/VAST/vast_dev.csv")
    model = BaselineBert()
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-4)

