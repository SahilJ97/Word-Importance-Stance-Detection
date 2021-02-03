import csv
import random
import spacy

nlp = spacy.load("en_core_web_sm")


def tokenize(s):
    doc = nlp(s)
    return " ".join([token.text for token in doc])


if __name__ == "__main__":
    with open("type1_for_ann.csv", "w") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=["topic", "argument", "label", "tokenized_argument", "new_id"])
        writer.writeheader()

        with open("../VAST/vast_train.csv", "r") as in_file:
            for entry in csv.DictReader(in_file):
                if entry["type_idx"] != "1" or entry["label"] == "2":
                    continue
                row = {
                    "topic": entry["new_topic"],
                    "argument": entry["post"],
                    "label": entry["label"],
                    "tokenized_argument": tokenize(entry["post"]),
                    "new_id": entry["new_id"]
                }
                writer.writerow(rowdict=row)




