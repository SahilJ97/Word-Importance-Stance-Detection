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


def expected_gradients(x, references):
    input_length = len(x)
    x = x.float()
    references = references.float()
    alphas = torch.rand(len(references))
    attributions = torch.zeros((input_length,))
    for r, alpha in zip(references, alphas):
        keep_r_indices = torch.cat([torch.bernoulli(alpha) for _ in range(input_length)])
        keep_x_indices = torch.ones((input_length,), dtype=torch.float) - keep_r_indices
        shifted_input = x * keep_x_indices + r * keep_r_indices
        shifted_output = model(shifted_input)
        shifted_output.backward()
        derivatives = shifted_input.grad
        attributions += (x - r) * derivatives
    return attributions / k  # return mean of sample results


def train():
    train_loader = DataLoader(train_set, batch_size + k, shuffle=True)  # k examples are used to compute attributions
    for epoch in range(NUM_EPOCHS):
        print(f"\tBeginning epoch {epoch}...")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels, attribution_info = data
            use_attributions, weights, relevance_scores = attribution_info
            inputs = inputs.to(DEVICE)
            inputs, reference_inputs = inputs[:batch_size], inputs[batch_size:]
            labels = labels[:batch_size]
            labels = one_hot(labels, num_classes=3).float()
            labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = binary_cross_entropy(outputs, labels)
            if use_prior:
                for i in range(len(inputs)):
                    if use_attributions[i]:
                        attributions = expected_gradients(inputs, reference_inputs)
                        attributions = torch.abs(attributions)
                        print(attributions)
                        weight_tensor, relevance_tensor = weights[i].to(DEVICE), relevance_scores[i].to(DEVICE)

                        print(expected_gradients[i])

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
    explainer = AttributionPriorExplainer(train_set, batch_size=batch_size, k=k)  # don't use k = 1 because few examples with labels? SWITCH TO PATH-EXPLAIN!!!
    dev_set = VastReader("../data/VAST/vast_dev.csv")
    model = BaselineBert()
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-4)
    train()
