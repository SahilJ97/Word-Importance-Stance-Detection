from csv import DictWriter, DictReader

csv_files = ["first_batch_results.csv", "second_batch_results.csv"]

if __name__ == "__main__":
    columns = []
    for f in csv_files:
        with open(f, "r") as in_file:
            columns.append(set(in_file.readline().strip().split(",")))

    all_columns = set.union(*columns)
    all_columns = list(all_columns)
    all_columns.sort()

    with open("two_batch_results.csv", "w+") as out_file:
        writer = DictWriter(out_file, fieldnames=all_columns)
        writer.writeheader()
        for f in csv_files:
            with open(f, "r") as in_file:
                reader = DictReader(in_file)
                for in_row in reader:
                    out_row = {col: "" for col in all_columns}
                    for key, value in in_row.items():
                        out_row[key] = value
                    writer.writerow(out_row)
