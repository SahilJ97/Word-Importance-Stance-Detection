from lazyme.string import color_print
import matplotlib.pyplot as plt
from csv import DictReader

visual_dict = {
    0: ("white", False),
    1: ("gray", False),
    2: ("gray", True),
    3: ("gray", True)
}

if __name__ == "__main__":
    # Visualize
    
    word_weights = {}
    topics = {}
    
    with open("two_batch_results.csv", "r") as f:
        reader =DictReader(f)
        for row in reader:
            hit_id = row["HITId"]
            tokens = row["Input.tokenized_argument"].split(" ")
            word_weights.setdefault(hit_id, [[token, 0] for token in tokens])
            topics.setdefault(hit_id, row["Input.topic"])
            selected = row["Answer.selected_words"].split(",")
            for item in selected:
                try:
                    token_number = int(item.replace("word_", ""))
                except ValueError:
                    print(item, row["HITId"])
                word_weights[hit_id][token_number][1] += 1

    for hit_id in topics.keys():
        topic = topics[hit_id]
        print(topic)
        for word, weight in word_weights[hit_id]:
            visuals = visual_dict[weight]
            color_print(f"{word}", end=" ", color=visuals[0], underline=visuals[1])
        print("\n")

    # Collect statistics
    with open("Batch_300480_batch_results.csv", "r") as f:
        durations = []
        word_spaces = []

        reader =DictReader(f)
        for row in reader:
            selected = row["Answer.selected_words"].split(",")
            durations.append(int(row["WorkTimeInSeconds"]))
            last_selected = None
            for item in selected:
                token_number = int(item.replace("word_", ""))
                if last_selected:
                    word_spaces.append(token_number - last_selected)
                last_selected = token_number

        plt.hist(durations, bins=50)
        plt.title("Work Time (seconds)")
        plt.show()

        plt.hist(word_spaces, bins=range(1, 30))
        plt.title("Space between selected words")
        plt.show()


