import torch
from attributionpriors.attributionpriors.pytorch_ops import AttributionPriorExplainer
from sys import argv
from src.vast_reader import VastReader
from src.classifiers import BaselineBert
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy, one_hot
from torch.optim import Adam
from pytorch_lightning.metrics.functional import f1
from src import visualize

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"  # use CUDA_VISIBLE_DEVICES=i python3 train.py?
NUM_EPOCHS = 20
SAMPLES_TO_VISUALIZE = 5  # NO! Just do them all!!!

"""
Key difference from original formulation: separate optimizer step for prior loss.
"""

# Parse arguments
smoothing, smooth_param, relevance_type, use_prior, batch_size, learn_rate, k, lda, model_name = argv[1:10]
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
    x_embeds = model.get_inputs_embeds(torch.unsqueeze(x, dim=0))
    references_embeds = model.get_inputs_embeds(references)
    alphas = torch.rand(len(references), device=DEVICE)
    attributions = torch.zeros((input_length,), device=DEVICE)
    for r_embeds, alpha in zip(references_embeds, alphas):
        r_embeds = torch.unsqueeze(r_embeds, dim=0)
        shifted_inputs_embeds = r_embeds + alpha * (x_embeds - r_embeds)
        shifted_output = model.forward(inputs_embeds=shifted_inputs_embeds)
        shifted_loss = binary_cross_entropy(shifted_output, y)
        derivatives = torch.autograd.grad(
            outputs=shifted_loss,
            inputs=shifted_inputs_embeds,
            grad_outputs=torch.ones_like(shifted_loss).to(DEVICE),
            create_graph=True  # needed to differentiate prior loss term
        )[0]
        derivative_norms = torch.norm(derivatives, dim=-1)  # aggregate token-level derivatives
        attributions = attributions + torch.squeeze(
            torch.norm(x_embeds-r_embeds, dim=-1) * derivative_norms,
            dim=0
        )
    return attributions / k  # return mean of sample results


def train():
    train_loader = DataLoader(train_set, batch_size + k, shuffle=True)  # k examples are used to compute attributions
    for epoch in range(NUM_EPOCHS):

        # Prepare attribution visualization file
        html_file = f"../output/{model_name}-{epoch}.html"
        with open(html_file, "w") as out_file:
            out_file.write(visualize.header)

        # Train
        print(f"\nBeginning epoch {epoch}...")
        running_correctness_loss, running_prior_loss = 0., 0.
        num_prior_losses = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels, attribution_info = data
            use_attributions, weights, relevance_scores = attribution_info
            inputs, reference_inputs = inputs[:batch_size], inputs[batch_size:]
            inputs = inputs.to(DEVICE)
            reference_inputs = reference_inputs.to(DEVICE)
            labels = labels[:batch_size]
            labels = one_hot(labels, num_classes=3).float()
            labels = labels.to(DEVICE)
            dev_outputs = model.forward(inputs=inputs)
            correctness_loss = binary_cross_entropy(dev_outputs, labels)
            running_correctness_loss += correctness_loss.item()
            correctness_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if use_prior:
                for j in range(len(inputs)):
                    if use_attributions[j]:
                        # Compute prior loss and back-propagate
                        attributions = expected_gradients(inputs[j], labels[j], reference_inputs)
                        attributions = torch.abs(attributions)
                        scores = attributions / torch.sum(attributions, dim=-1)
                        weight_tensor, relevance_tensor = weights[j].to(DEVICE), relevance_scores[j].to(DEVICE)
                        prior_loss = lda * sum((weight_tensor - scores)**2 * relevance_tensor) / sum(relevance_tensor)
                        running_prior_loss += prior_loss.item()
                        num_prior_losses += 1
                        prior_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        with torch.cuda.device(DEVICE):
                            torch.cuda.empty_cache()

                        # Output visualization to file
                        tokens = train_set.tokenizer.convert_ids_to_tokens(inputs[j])
                        att_word_weights = attributions * relevance_tensor
                        gold_word_weights = weight_tensor * relevance_tensor
                        attributions_html = visualize.get_words_html(
                            tokens,
                            att_word_weights.tolist()
                        )
                        weights_html = visualize.get_words_html(
                            tokens,
                            gold_word_weights.tolist()
                        )
                        with open(html_file, "a") as out_file:
                            out_file.write(f"Model attributions:\n{attributions_html}\n")
                            out_file.write(f"Attribution labels:\n{weights_html}\n")

            if i % 10 == 0 and i != 0:
                print(f"Epoch {epoch} iteration {i}")
                print(f"\tRunning correctness loss: {running_correctness_loss/i}")
                if num_prior_losses > 0:
                    print(f"\tRunning prior loss: {running_prior_loss/num_prior_losses}")

        # Save
        print("Saving model...")
        torch.save(model, f"../output/{model_name}.pt")

        # Validate
        print("Validating...")
        dev_inputs, dev_labels, _ = dev_set[:]
        dev_inputs = dev_inputs.to(DEVICE)
        dev_labels = one_hot(dev_labels, num_classes=3).float()
        dev_labels = dev_labels.to(DEVICE)
        dev_outputs = model.forward(inputs=dev_inputs)
        correctness_loss = binary_cross_entropy(dev_outputs, dev_labels)
        f = f1(dev_outputs, dev_labels, num_classes=3, average="macro", multilabel=True)
        print(f"\tLoss: {correctness_loss.item()}")
        print(f"\tF1: {f}")


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
    dev_set = VastReader("../data/VAST/vast_dev.csv")
    explainer = AttributionPriorExplainer(train_set, batch_size=batch_size, k=k)
    model = BaselineBert()
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-4)
    train()
