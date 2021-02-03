from csv import DictReader

if __name__ == "__main__":
    workers = {}

    with open("two_batch_results.csv", "r") as f:
        reader = DictReader(f)
        for row in reader:
            n_words = len(row["Input.tokenized_argument"].split(" "))
            min_selections = round(n_words / 11)
            max_selections = round(n_words / 5.5)
            worker_id = row["WorkerId"]
            n_selections = len(row["Answer.selected_words"].split(","))
            value = (n_selections - min_selections) / (max_selections - min_selections)
            workers.setdefault(worker_id, []).append(value)

    print("Worker ID\t# HITS\tAvg # Selections")
    for worker_id, values in workers.items():
        print(worker_id, "\t", len(values), "  \t", sum(values)/len(values))
