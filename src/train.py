import torch
from attributionpriors.attributionpriors.pytorch_ops import AttributionPriorExplainer
from sys import argv
from src.vast_reader import VastReader
from src.classifiers import BaselineBert
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy, one_hot
from torch.optim import Adam

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # switch to CUDA_VISIBLE_DEVICES=2 python3 train.py
NUM_EPOCHS = 20

# Parse arguments
smoothing, smooth_param, relevance_type, use_prior, batch_size, learn_rate, k = argv[1:8]
if smoothing == "none":
    smoothing = None
else:
    smooth_param = float(smooth_param)
batch_size = int(batch_size)
learn_rate = float(learn_rate)
k = int(k)
use_prior = use_prior in ["true", "True", "t"]


def train():
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    for epoch in range(NUM_EPOCHS):
        print(f"\tBeginning epoch {epoch}...")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels, attribution_info = data
            use_attributions, weights, relevance_scores = attribution_info
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            labels = one_hot(labels, num_classes=3).float()
            optimizer.zero_grad()
            print(model(inputs).size())
            outputs = model(inputs)
            loss = binary_cross_entropy(outputs, labels)
            if use_prior:
                for i in range(len(inputs)):
                    if use_attributions[i]:
                        ip_seq = inputs[i]
                        expected_gradients = explainer.shap_values(model, ip_seq)
                        print(expected_gradients)

            loss.backward()
            optimizer.step()

            # val: visualize attributions! track change!


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
    train()
