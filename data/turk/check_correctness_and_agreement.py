from csv import DictReader
from statistics import mean


def get_key_for_string_pair(a, b):
    return " ".join([a, b].sort())


if __name__ == "__main__":

    wrong_by_worker = {}
    worker_totals = {}
    wrong_by_hit = {}
    with open("two_batch_results.csv", "r") as f:
        reader = DictReader(f)
        for row in reader:
            predicted_pro = row["Answer.stance-prediction.pro"] == "TRUE"
            if predicted_pro and row["Input.label"] == "0" or not predicted_pro and row["Input.label"] == "1":
                wrong_by_worker.setdefault(row["WorkerId"], []).append(
                    row["Input.topic"] + "----" + row["Input.argument"]
                )
                wrong_by_hit.setdefault(row["HITId"], []).append(row["WorkerId"])
            worker_totals[row["WorkerId"]] = worker_totals.setdefault(row["WorkerId"], 0) + 1

    # Compute WQSs

    documents = {}  # stores valid responses, indexed by document
    doc_level_WWAs = {}  # stores document-level worker-worker agreement scores, indexed by document.
    worker_ids = set()  # holds ids of all workers with valid responses

    # populate documents dict
    with open("first_batch_results.csv", "r") as f:
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

    print("Worker ID \t # of responses \t Agreement w/ labels \t WQS \t squared WQS")
    for id, total in worker_totals.items():
        n_wrong = 0
        if id in wrong_by_worker.keys():
            n_wrong = len(wrong_by_worker[id])
        if id in wqs.keys():
            print(f"{id}\t{total}\t\t\t\t{round(1-n_wrong/total, 3)}  \t\t\t\t\t  {round(wqs[id], 3)} \t  {round(wqs[id]**2, 3)}")
