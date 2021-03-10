import torch
from src.vast_reader import VastReader
from src.models.bert_joint import BertJoint
from src.models.mem_net import MemoryNetwork
from src.utils import get_pad_mask
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.metrics import f1_score
from src import visualize
import numpy as np
from nltk.corpus import stopwords
import string
import argparse
import sys

NUM_EPOCHS = 20
CLASS_WEIGHTS = None
loss = CrossEntropyLoss(weight=CLASS_WEIGHTS)

# Parse arguments
true_strings = ['t', 'true', '1', 'yes', 'y', ]
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--relevance_type', help='Type of relevance scores ("binary" or "tf-idf")',
                    required=False)
parser.add_argument('-u', '--use_prior', help='Use attribution prior? y or n',
                    type=lambda x: (str(x).lower() in true_strings), required=True)
parser.add_argument('-b', '--batch_size', type=int, required=True)
parser.add_argument('-l', '--learning_rate', type=float, required=True)
parser.add_argument('-k', '--n_references', help='Number of references used to compute Expected Gradients', type=int,
                    required=False)
parser.add_argument('--lambda', help='Prior loss coefficient', type=float, required=False)
parser.add_argument('-o', '--output_name', help='Filename (without suffix) to which the best model will be saved',
                    required=True)
parser.add_argument('-m', '--model_type', help='bert-joint or mem-net', required=True)
parser.add_argument('-s', '--random_seed', type=int, required=True)
parser.add_argument('--num_hops', help='Number of hops (only applies if model_type is mem-net)', type=int,
                    required=False)
parser.add_argument('--topic_knowledge', help='Topic knowledge file (only applies if model_type is mem-net)',
                    required=False)
parser.add_argument('--knowledge_transfer',
                    help='Knowledge transfer scheme (parallel or projection). Only applies if model_type is mem-net',
                    required=False)
parser.add_argument('--gpu', required=False)
args = vars(parser.parse_args())
relevance_type = args['relevance_type']
if not relevance_type:
    relevance_type = "binary"
use_prior = args['use_prior']
batch_size = args['batch_size']
learning_rate = args['learning_rate']
k = args['n_references']
lda = args['lambda']
output_name = args['output_name']
model_type = args['model_type']
seed = args['random_seed']
num_hops = args['num_hops']
topic_knowledge = args['topic_knowledge']
knowledge_transfer = args['knowledge_transfer']
gpu = args['gpu']

DEVICE = f"cuda:{gpu}" if gpu else "cpu"


def empty_cache():
    if "cuda" in DEVICE:
        with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()


def expected_gradients(x, y, references, x_pad_mask, doc_stopword_mask):
    input_length = len(x)
    x_embeds = model.get_inputs_embeds(torch.unsqueeze(x, dim=0))
    x_pad_mask = torch.unsqueeze(x_pad_mask, dim=0)
    doc_stopword_mask = torch.unsqueeze(doc_stopword_mask, dim=0)
    references_embeds = model.get_inputs_embeds(references)
    alphas = torch.rand(len(references), device=DEVICE)
    attributions = torch.zeros((input_length,), device=DEVICE)
    for r_embeds, alpha in zip(references_embeds, alphas):
        r_embeds = torch.unsqueeze(r_embeds, dim=0)
        shifted_inputs_embeds = r_embeds + alpha * (x_embeds - r_embeds)
        shifted_output = model.forward(
            x_pad_mask,
            doc_stopword_mask,  # note: simply using stopword masks for x
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
            torch.norm(x_embeds - r_embeds, dim=-1) * derivative_norms,
            dim=0
        )
    return attributions / k  # return mean of sample results


def train():
    train_loader_batch_size = batch_size
    if use_prior:
        train_loader_batch_size += k  # k examples are used to compute attributions
    train_loader = DataLoader(train_set, train_loader_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size, shuffle=True)
    for epoch in range(NUM_EPOCHS):

        # Prepare attribution visualization file
        if use_prior:
            html_file = f"../output/{output_name}-{epoch}.html"
            with open(html_file, "w") as out_file:
                out_file.write(visualize.header)

        # Train
        print(f"\nBeginning epoch {epoch}...")
        running_correctness_loss, running_prior_loss = 0., 0.
        num_prior_losses = 0
        epoch_losses = []
        epoch_combined_f1s = []
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            empty_cache()
            inputs, labels, doc_stopword_mask, attribution_info = data
            has_att_labels, weights, relevance_scores = attribution_info
            inputs, reference_inputs = inputs[:batch_size], inputs[batch_size:]
            pad_mask = get_pad_mask(inputs).to(DEVICE)
            inputs = inputs.to(DEVICE)
            reference_inputs = reference_inputs.to(DEVICE)
            labels = labels[:batch_size]
            labels = labels.to(DEVICE)
            doc_stopword_mask = doc_stopword_mask[:batch_size].to(DEVICE)
            outputs = model.forward(
                pad_mask,
                doc_stopword_mask,
                inputs=inputs,
                use_dropout=True,
                token_type_ids=token_type_ids[:len(inputs)]
            )
            correctness_loss = loss(outputs, labels)
            running_correctness_loss += correctness_loss.item()
            correctness_loss.backward()

            if use_prior:
                for j in range(len(inputs)):
                    if has_att_labels[j]:
                        empty_cache()
                        # Compute prior loss and back-propagate
                        attributions = expected_gradients(
                            inputs[j],
                            labels[j],
                            reference_inputs,
                            x_pad_mask=pad_mask[j],
                            doc_stopword_mask=doc_stopword_mask[j],
                        )
                        attributions = torch.abs(attributions)
                        scores = attributions / torch.sum(attributions, dim=-1)
                        weight_tensor, relevance_tensor = weights[j].to(DEVICE), relevance_scores[j].to(DEVICE)
                        prior_loss = lda * sum((weight_tensor - scores) ** 2 * relevance_tensor) / sum(relevance_tensor)
                        running_prior_loss += prior_loss.item()
                        num_prior_losses += 1
                        try:
                            prior_loss.backward()
                        except RuntimeError as e:  # once occurred unexpectedly during 8th epoch
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

            optimizer.step()  # use accumulated gradients from correctness loss term and prior loss term(s)

            # Print running losses every 10 batches
            if i % 10 == 0 and i != 0:
                print(f"Epoch {epoch} iteration {i}")
                print(f"\tRunning correctness loss: {running_correctness_loss / (i + 1)}")
                if num_prior_losses > 0:
                    print(f"\tRunning prior loss: {running_prior_loss / num_prior_losses}")

        # Validate
        print("Validating...")
        with torch.no_grad():
            all_labels = []
            all_outputs = []
            for i, data in enumerate(dev_loader, 0):
                empty_cache()
                inputs, labels, doc_stopword_mask, _ = data
                pad_mask = get_pad_mask(inputs).to(DEVICE)
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                doc_stopword_mask = doc_stopword_mask.to(DEVICE)
                all_labels.append(labels)
                outputs = model.forward(
                    pad_mask,
                    doc_stopword_mask,
                    inputs=inputs,
                    use_dropout=False,
                    token_type_ids=token_type_ids[:len(inputs)]
                )
                all_outputs.append(outputs)
            all_labels = torch.cat(all_labels, dim=0)
            all_outputs = torch.cat(all_outputs, dim=0)
            correctness_loss = loss(all_outputs, all_labels).item()
            _, all_preds = torch.max(all_outputs, dim=-1)
            class_f1 = f1_score(all_labels.tolist(), all_preds.tolist(), labels=[0, 1, 2], average=None)
            combined_f1 = np.sum(class_f1) / 3
        print(f"\tLoss: {correctness_loss}")
        print(f"\tF1: {class_f1[0], class_f1[1], combined_f1}")

        # Early stopping
        epoch_losses.append(correctness_loss)
        epoch_combined_f1s.append(combined_f1)
        if len(epoch_losses) >= 3:
            if epoch_losses[-1] > epoch_losses[-2] and epoch_losses[-1] > epoch_losses[-3] \
            and epoch_combined_f1s[-1] < epoch_combined_f1s[-2] and epoch_combined_f1s[-1] < epoch_combined_f1s[-3]:
                print("Stopping early!")
                break

        # Save model with best combined F1 on dev set
        if len(epoch_losses) >= 2:
            if epoch_combined_f1s[-1] == max(epoch_combined_f1s):
                print("Saving model...")
                torch.save(model, f"../output/{output_name}.pt")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)  # helps with debugging
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Loading data...")
    train_set = VastReader(
        "../data/VAST/vast_train.csv",
        "../data/VAST_word_importance/token_appearances.tsv",
        exclude_from_main="../data/VAST_word_importance/special_datapoints.txt",
        word_importance_csv="../data/VAST_word_importance/processed_annotated.csv",
        smoothing=None,
        relevance_type=relevance_type
    )
    dev_set = VastReader("../data/VAST/vast_dev.csv")

    token_type_ids = [0 for _ in range(train_set.doc_len + 2)] + [1 for _ in range(train_set.topic_len + 1)]
    token_type_ids = [token_type_ids for _ in range(batch_size)]
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=DEVICE)

    first_input, first_label, first_doc_stopword_mask, _ = train_set[0]
    print(train_set.tokenizer.convert_ids_to_tokens(first_input))
    print(first_label)
    print(first_doc_stopword_mask)

    print("Loading model...")
    if model_type == 'bert-joint':
        model = BertJoint(doc_len=train_set.doc_len, fix_bert=False)  # cannot fix BERT if using attribution prior!
    elif model_type == 'mem-net':
        model = MemoryNetwork(doc_len=train_set.doc_len, num_hops=num_hops, hidden_layer_size=283,
                              init_topic_knowledge_file=topic_knowledge, knowledge_transfer_scheme=knowledge_transfer)
    else:
        print("Please specify a valid model type.", file=sys.stderr)
        exit(1)
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train()
