import torch
from attributionpriors.attributionpriors.pytorch_ops import AttributionPriorExplainer
from sys import argv
from src.vast_reader import VastReader
from src.classifiers import BaselineBert
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy, one_hot
from torch.optim import Adam

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"  # use CUDA_VISIBLE_DEVICES=i python3 train.py?
NUM_EPOCHS = 20

# Parse arguments
smoothing, smooth_param, relevance_type, use_prior, batch_size, learn_rate, k, lda = argv[1:9]
if smoothing == "none":
    smoothing = None
else:
    smooth_param = float(smooth_param)
batch_size = int(batch_size)
learn_rate = float(learn_rate)
k = int(k)
use_prior = use_prior in ["true", "True", "t"]
lda = float(lda)  # lambda (prior loss coefficient)


def expected_gradients(x, y, references):
    input_length = len(x)
    references = references
    alphas = torch.rand(len(references), device=DEVICE)
    attributions = torch.zeros((input_length,), device=DEVICE)
    for r, alpha in zip(references, alphas):
        keep_r_indices = torch.stack([torch.bernoulli(alpha) for _ in range(input_length)])
        keep_x_indices = torch.ones((input_length,), dtype=torch.float, device=DEVICE) - keep_r_indices
        shifted_input = x * keep_x_indices + r * keep_r_indices
        shifted_input = torch.unsqueeze(shifted_input, dim=0)
        shifted_input = shifted_input
        shifted_output, hidden_states = model.forward_with_hidden_states(shifted_input)
        first_hidden_state = hidden_states[0]
        shifted_loss = binary_cross_entropy(shifted_output, y)
        derivatives = torch.autograd.grad(
            outputs=shifted_loss,
            inputs=first_hidden_state,
            grad_outputs=torch.ones_like(shifted_loss).to(DEVICE),
            create_graph=True  # needed to differentiate prior loss term
        )[0]
        derivative_norms = torch.norm(derivatives, dim=-1)  # aggregate token-level derivatives
        derivative_norms = torch.squeeze(derivative_norms, dim=0)
        attributions = attributions + (x - r) * derivative_norms
    return attributions / k  # return mean of sample results


def train():
    train_loader = DataLoader(train_set, batch_size + k, shuffle=True)  # k examples are used to compute attributions
    for epoch in range(NUM_EPOCHS):
        print(f"\tBeginning epoch {epoch}...")
        running_correctness_loss, running_prior_loss = 0., 0.
        for i, data in enumerate(train_loader, 0):
            inputs, labels, attribution_info = data
            use_attributions, weights, relevance_scores = attribution_info
            inputs, reference_inputs = inputs[:batch_size], inputs[batch_size:]
            inputs = inputs.to(DEVICE)
            reference_inputs = reference_inputs.to(DEVICE)
            labels = labels[:batch_size]
            labels = one_hot(labels, num_classes=3).float()
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = binary_cross_entropy(outputs, labels)
            if use_prior:
                for j in range(len(inputs)):
                    if use_attributions[j]:
                        attributions = expected_gradients(inputs[j], labels[j], reference_inputs)
                        attributions = torch.abs(attributions)
                        scores = attributions / torch.sum(attributions, dim=-1)
                        print(scores[:20])
                        weight_tensor, relevance_tensor = weights[j].to(DEVICE), relevance_scores[j].to(DEVICE)
                        prior_loss = sum((weight_tensor - scores)**2 * relevance_tensor) / sum(relevance_tensor)
                        loss = loss + lda*prior_loss
            print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

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
