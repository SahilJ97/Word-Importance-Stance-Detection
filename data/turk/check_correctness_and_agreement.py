from csv import DictReader
from statistics import mean
from data.turk import print_weights_html


def get_key_for_string_pair(a, b):
    return " ".join([a, b].sort())


if __name__ == "__main__":

    wrong_by_worker = {}
    worker_totals = {}
    wrong_by_hit = {}
    all_selections = {}
    #min_distance_from_topic = []
    contains_topic = 0
    doesnt_contain = 0
    with open("two_batch_results.csv", "r") as f:
        reader = DictReader(f)
        for row in reader:
            if row["Input.topic"] in row["Input.argument"] or row["Input.topic"] in row["Input.tokenized_argument"]:
                contains_topic += 1
            else:
                doesnt_contain += 1
            all_selections.setdefault(row["HITId"], []).append(row["Answer.selected_words"].split(","))
            predicted_pro = row["Answer.stance-prediction.pro"] == "TRUE"
            if predicted_pro and row["Input.label"] == "0" or not predicted_pro and row["Input.label"] == "1":
                wrong_by_worker.setdefault(row["WorkerId"], []).append(
                    row["Input.topic"] + "----" + row["Input.argument"]
                )
                wrong_by_hit.setdefault(row["HITId"], []).append(row["WorkerId"])
            worker_totals[row["WorkerId"]] = worker_totals.setdefault(row["WorkerId"], 0) + 1

    print(f"{contains_topic/(contains_topic+doesnt_contain)} of examples contain topic in argument")

    print("HITs for which all 3 workers selected the wrong label:")
    for hit, workers_wrong in wrong_by_hit.items():
        if len(workers_wrong) == 3:
            print(hit)

    print("\nHITs for which 2 workers selected the wrong label:")
    for hit, workers_wrong in wrong_by_hit.items():
        if len(workers_wrong) == 2:
            print(hit)

    # Compute WQSs

    documents = {}  # stores valid responses, indexed by document
    doc_level_WWAs = {}  # stores document-level worker-worker agreement scores, indexed by document.
    worker_ids = set()  # holds ids of all workers with valid responses

    # populate documents dict
    with open("two_batch_results.csv", "r") as f:
        reader = DictReader(f)
        for row in reader:
            predicted_pro = row["Answer.stance-prediction.pro"] == "TRUE"
            if predicted_pro and row["Input.label"] == "1" or not predicted_pro and row["Input.label"] == "0":
                documents.setdefault(row["HITId"], []).append(
                    (row["WorkerId"], set(row["Answer.selected_words"].split(",")))
                )
    # populate doc_level_WWAs
    for hit_id, responses in documents.items():
        doc_level_WWAs[hit_id] = {}
        for i in range(len(responses)):
            for j in range(len(responses)):
                if i == j:
                    continue
                r1 = responses[i]
                worker_ids.add(r1[0])
                r2 = responses[j]
                wwa = len(r1[1].intersection(r2[1])) / len(r1[1])
                doc_level_WWAs[hit_id][f"{r1[0]} {r2[0]}"] = wwa

    # compute WQSs
    wqs = {}
    for worker_id in worker_ids:
        numerator, denominator = 0, 0
        for doc in doc_level_WWAs.values():
            if len(doc.values()) > 0:
                avg_wwa = mean(wwa for wwa in doc.values())
            else:
                continue
            for pair, wwa in doc.items():
                if pair.startswith(worker_id):
                    numerator += wwa*avg_wwa
                    denominator += avg_wwa
        wqs[worker_id] = numerator/denominator

    print("\nWorker ID \t # of responses \t Agreement w/ labels \t WQS \t squared WQS")
    for id, total in worker_totals.items():
        n_wrong = 0
        if id in wrong_by_worker.keys():
            n_wrong = len(wrong_by_worker[id])
        if id in wqs.keys():
            print(f"{id}\t{total}\t\t\t\t{round(1-n_wrong/total, 3)}  \t\t\t\t\t  {round(wqs[id], 3)} \t  {round(wqs[id]**2, 3)}")

    print("\nWeights and raw aggregation scores (latter includes responses with incorrect stance selections):")

    # Print HITs
    for hit_id, all_selections_for_hit in all_selections.items():
        if hit_id in documents.keys():
            accepted_workers = documents[hit_id]
        else:
            accepted_workers = []
        print(hit_id)
                # Initialize word weight data structure
        with open("two_batch_results.csv", "r") as f:
            reader = DictReader(f)
            for row in reader:
                if row["HITId"] == hit_id:
                    break
        words = {}
        words_for_raw = {}
        i = 0
        while f"Answer.word_{i}" in row.keys():
            words[f"word_{i}"] = (row[f"Answer.word_{i}"], 0)
            words_for_raw[f"word_{i}"] = (row[f"Answer.word_{i}"], 0)
            i += 1

        # Compute
        for worker_id, selections in accepted_workers:
            quality_score = wqs[worker_id]
            for word in selections:
                words[word] = (words[word][0], words[word][1] + quality_score)
        print(words)

        for selections in all_selections_for_hit:
            for word in selections:
                words_for_raw[word] = (words_for_raw[word][0], words_for_raw[word][1] + .33)

        # Print HTML
        print(" ".join([w[0] for w in words.values()]))
        html = []
        for i in range(len(words.keys())):
            text, weight = words[f"word_{i}"]
            html.append(print_weights_html.get_word_html(text, weight))
        print(" ".join(html))

        html = []
        for i in range(len(words_for_raw.keys())):
            text, weight = words_for_raw[f"word_{i}"]
            html.append(print_weights_html.get_word_html(text, weight))
        print(" ".join(html))

