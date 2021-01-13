from csv import DictReader, DictWriter


if __name__ == "__main__":

    with open("first_700_fixed_unprocessed.csv", "r") as in_file:
        with open("first_700_processed.csv", "w+") as out_file:
            reader = DictReader(in_file)
            output_fields = ["topic", "argument", "label", "tokenized_argument", "new_id"]
            writer = DictWriter(out_file, output_fields)
            writer.writeheader()
            for in_row in reader:
                if "x" in in_row["Remove?"]:
                    continue
                out_row = {field: in_row[field] for field in output_fields}
                if in_row["Updated topic"] != "":
                    out_row["topic"] = in_row["Updated topic"]
                if in_row["Updated label"] != "":
                    out_row["label"] = in_row["Updated label"]
                writer.writerow(out_row)
