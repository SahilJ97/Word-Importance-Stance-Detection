import torch
from attributionpriors.attributionpriors.pytorch_ops import AttributionPriorExplainer
from sys import argv
from src.vast_reader import VastReader
from src.classifiers import BaselineBert
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.metrics import f1_score
from src import visualize
import numpy as np
from nltk.corpus import stopwords
import string

sw = stopwords.words("english")
punc = [c for c in string.punctuation]
sw_and_punc = sw + punc

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"  # use CUDA_VISIBLE_DEVICES=i python3 train.py? causes issue
NUM_EPOCHS = 20

"""
Key difference from original formulation: separate optimizer step for prior loss.
"""

"""CLASS_WEIGHTS = torch.tensor(
    [2.413433908045977, 2.528316086547507, 5.2594911937377695],
    device=DEVICE
)  # inverse label frequency"""
CLASS_WEIGHTS = None
loss = CrossEntropyLoss(weight=CLASS_WEIGHTS)

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


def empty_cache():
    if "cuda" in DEVICE:
        with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()


def get_mask(inputs, tokenizer):
    """Used to zero embeddings corresponding to [PAD] tokens before pooling BERT embeddings"""
    inputs = inputs.tolist()
    mask = np.ones_like(inputs)
    for i in range(len(inputs)):
        #tokens = tokenizer.convert_ids_to_tokens(inputs[i])
        for j in range(len(inputs[i])):
            #if tokens[j] == "[PAD]" or (j > model.topic_len+1 and tokens[j] in sw_and_punc):
            if inputs[i][j] == 0:
                mask[i][j] = 0
    return torch.tensor(mask, dtype=torch.float, device=DEVICE)


def expected_gradients(x, y, references, x_mask):
    input_length = len(x)
    x_embeds = model.get_inputs_embeds(torch.unsqueeze(x, dim=0))
    mask = torch.unsqueeze(x_mask, dim=0)
    references_embeds = model.get_inputs_embeds(references)
    alphas = torch.rand(len(references), device=DEVICE)
    attributions = torch.zeros((input_length,), device=DEVICE)
    for r_embeds, alpha in zip(references_embeds, alphas):
        r_embeds = torch.unsqueeze(r_embeds, dim=0)
        shifted_inputs_embeds = r_embeds + alpha * (x_embeds - r_embeds)
        shifted_output = model.forward(
            mask,
            inputs_embeds=shifted_inputs_embeds,
            use_dropout=False,
            token_type_ids=token_type_ids[0]
        )
        shifted_loss = loss(shifted_output, torch.unsqueeze(y, dim=-1))
        derivatives = torch.autograd.grad(
            outputs=shifted_loss,
            inputs=shifted_inputs_embeds,
            grad_outputs=torch.ones_like(shifted_loss).to(DEVICE),
            create_graph=True,  # needed to differentiate prior loss term
        )[0]
        derivative_norms = torch.norm(derivatives, dim=-1)  # aggregate token-level derivatives
        attributions = attributions + torch.squeeze(
            torch.norm(x_embeds-r_embeds, dim=-1) * derivative_norms,
            dim=0
        )
    return attributions / k  # return mean of sample results


def train():
    train_loader_batch_size = batch_size
    if use_prior:
        train_loader_batch_size += k  # k examples are used to compute attributions
    train_loader = DataLoader(train_set, train_loader_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size, shuffle=False)
    for epoch in range(NUM_EPOCHS):

        # Prepare attribution visualization file
        if use_prior:
            html_file = f"../output/{model_name}-{epoch}.html"
            with open(html_file, "w") as out_file:
                out_file.write(visualize.header)

        # Train
        print(f"\nBeginning epoch {epoch}...")
        running_correctness_loss, running_prior_loss = 0., 0.
        num_prior_losses = 0
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            empty_cache()
            inputs, labels, attribution_info = data
            has_att_labels, weights, relevance_scores = attribution_info
            inputs, reference_inputs = inputs[:batch_size], inputs[batch_size:]
            mask = get_mask(inputs, train_set.tokenizer)
            inputs = inputs.to(DEVICE)
            reference_inputs = reference_inputs.to(DEVICE)
            labels = labels[:batch_size]
            labels = labels.to(DEVICE)
            outputs = model.forward(mask, inputs=inputs, use_dropout=True, token_type_ids=token_type_ids[:len(inputs)])
            correctness_loss = loss(outputs, labels)
            running_correctness_loss += correctness_loss.item()
            correctness_loss.backward()

            if use_prior:
                for j in range(len(inputs)):
                    if has_att_labels[j]:
                        empty_cache()
                        # Compute prior loss and back-propagate
                        attributions = expected_gradients(inputs[j], labels[j], reference_inputs, x_mask=mask[j])
                        attributions = torch.abs(attributions)
                        scores = attributions / torch.sum(attributions, dim=-1)
                        weight_tensor, relevance_tensor = weights[j].to(DEVICE), relevance_scores[j].to(DEVICE)
                        prior_loss = lda * sum((weight_tensor - scores)**2 * relevance_tensor) / sum(relevance_tensor)
                        running_prior_loss += prior_loss.item()
                        num_prior_losses += 1
                        try:
                            prior_loss.backward()
                        except RuntimeError as e:  # Once occurred unexpectedly during 8th epoch
                            print("RuntimeError while backpropagating prior loss. Skipping...")
                            continue

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
                            out_file.write(f"<p>Model attributions:</p>\n{attributions_html}\n")
                            out_file.write(f"<p>Attribution labels:</p>\n{weights_html}\n")
                            out_file.write(f"<p>predicted, actual: {outputs[j].tolist(), labels[j].tolist()}</p>\n")

            optimizer.step()  # Use accumulated gradients from correctness loss term and prior loss term(s)

            # Print running losses every 10 batches
            if i % 10 == 0 and i != 0:
                print(f"Epoch {epoch} iteration {i}")
                print(f"\tRunning correctness loss: {running_correctness_loss/(i+1)}")
                if num_prior_losses > 0:
                    print(f"\tRunning prior loss: {running_prior_loss/num_prior_losses}")

        # Save
        #print("Saving model...")
        #torch.save(model, f"../output/{model_name}.pt")

        # Validate
        print("Validating...")
        with torch.no_grad():
            all_labels = []
            all_outputs = []
            for i, data in enumerate(dev_loader, 0):
                empty_cache()
                inputs, labels, _ = data
                mask = get_mask(inputs, train_set.tokenizer)
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                all_labels.append(labels)
                outputs = model.forward(
                    mask,
                    inputs=inputs,
                    use_dropout=False,
                    token_type_ids=token_type_ids[:len(inputs)]
                )
                all_outputs.append(outputs)
            all_labels = torch.cat(all_labels, dim=0)
            all_outputs = torch.cat(all_outputs, dim=0)
            correctness_loss = loss(all_outputs, all_labels)
            _, all_preds = torch.max(all_outputs, dim=-1)
            class_f1 = f1_score(all_labels.tolist(), all_preds.tolist(), labels=[0, 1, 2], average=None)
        print(f"\tLoss: {correctness_loss.item()}")
        print(f"\tF1: {class_f1[0], class_f1[1], np.sum(class_f1)/3}")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)  # helps with debugging
    print("Loading data...")
    train_set = VastReader(
        "../data/VAST/vast_train.csv",
        "../data/VAST_word_importance/token_appearances.tsv",
        exclude_from_main="../data/VAST_word_importance/special_datapoints.txt",
        word_importance_csv="../data/VAST_word_importance/processed_annotated.csv",
        smoothing=smoothing,
        smooth_param=smooth_param,
        relevance_type=relevance_type
    )

    token_type_ids = [0 for _ in range(train_set.topic_len + 2)] + [1 for _ in range(train_set.doc_len + 1)]
    token_type_ids = [token_type_ids for _ in range(batch_size)]
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=DEVICE)
    print(token_type_ids.size(), token_type_ids[0])

    dev_set = VastReader("../data/VAST/vast_dev.csv")

    first_input, first_label, _ = train_set[0]
    print(train_set.tokenizer.convert_ids_to_tokens(first_input))
    print(first_label)

    if use_prior:
        explainer = AttributionPriorExplainer(train_set, batch_size=batch_size, k=k)

    print("Loading model...")
    model = BaselineBert(topic_len=train_set.topic_len, fix_bert=True)  # cannot fix BERT if using attribution prior
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=learn_rate)

    train()
