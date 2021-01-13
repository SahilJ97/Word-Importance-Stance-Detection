from csv import DictReader


class Worker:
    def __init__(self, id, words_selected, prediction):
        self.id = id
        self.words_selected = words_selected
        self.prediction = prediction


required_words = {"word_45", "word_47"}
desired_words = {"word_42", "word_44", "word_46", "word_22", "word_24", "word_26", "word_27", "word_34", "word_36",
                 "word_40"}


def select(workers):
    workers_accepted = []
    selections_seen = {}
    for worker in workers:
        words_selected = " ".join([word_map[w] for w in sorted(worker.words_selected)])
        if words_selected in selections_seen.keys():
            if selections_seen[words_selected]:
                workers_accepted.append(worker)
            print("Automatically handling duplicate...")
            continue
        print(words_selected)
        while True:
            choice = input("Accept? y or n...")
            if "y" in choice:
                workers_accepted.append(worker)
                selections_seen[words_selected] = True
                break
            elif "n" in choice:
                selections_seen[words_selected] = False
                break

    print([worker.id for worker in workers_accepted])


if __name__ == "__main__":

    with open("qual_results.csv", "r") as f:
        workers = []
        reader = DictReader(f)
        word_map = None
        for row in reader:

            # Populate word_map on the first iteration
            if not word_map:
                word_map = {}
                for i in range(len(row["Input.tokenized_argument"].split())):
                    if f"Answer.word_{i}" in row.keys():
                        word_map[f"word_{i}"] = row[f"Answer.word_{i}"]
                print(word_map)

            # Save worker info
            workers.append(
                Worker(row["WorkerId"], set(row["Answer.selected_words"].split(",")), row["Answer.stance-prediction.pro"])
            )
        print(f"{len(workers)} workers in total")

        got_correct = []
        for worker in workers:
            if worker.prediction == "FALSE":
                got_correct.append(worker)
        print(f"{len(got_correct)} selected the correct stance")

        selected_required = []
        for worker in got_correct:
            if required_words.issubset(worker.words_selected):
                selected_required.append(worker)
        print(f"{len(selected_required)}/{len(got_correct)} selected all of the required words")

        desired_selected = {i: [] for i in range(len(desired_words)+1)}
        for worker in selected_required:
            n_selected = len(desired_words.intersection(worker.words_selected))
            desired_selected[n_selected].append(worker)

        print(f"For those {len(selected_required)}, here is the distribution of how many DESIRED (but not required) "
              f"words were selected:")
        for i in range(len(desired_words) + 1):
            print(f"\n{i}: {len(desired_selected[i])}")
            for worker in desired_selected[i]:
                all_selected = [word_map[w] for w in sorted(worker.words_selected)]
                print(all_selected)

    select(selected_required)
