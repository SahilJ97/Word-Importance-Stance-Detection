from google.cloud import translate_v2 as translate
from csv import DictReader, DictWriter
from glob import glob
from stop_words import get_stop_words
from nltk.tokenize import PunktSentenceTokenizer
import ro_core_news_sm
from string import punctuation
from unidecode import unidecode

translate_client = translate.Client()
spacy_model = ro_core_news_sm.load()
stop_words = get_stop_words("ro")


def translate_and_tokenize(document):
    """
    param document: string to be translated
    return: (tokens of translation output, non-tokenized translation output, tokens in sentence form)
    """
    sentences = PunktSentenceTokenizer().tokenize(document)
    translated_sentences = []
    filtered_trans_sentences = []
    tokens = []
    for sentence in sentences:
        translated = translate_client.translate(
            sentence,
            target_language="ro"
        )["translatedText"]
        translated_sentences.append(translated)

        # Tokens
        word_tokenized = [token.text for token in spacy_model(translated)]
        filtered_tokens = []
        for token in word_tokenized:
            token = token.lower()
            if token not in punctuation and token not in ["...", "--"] and unidecode(token) not in stop_words:
                filtered_tokens.append(token)
        tokens.append(filtered_tokens)
        filtered_trans_sentences.append(" ".join(filtered_tokens) + ".")
    return tokens, " ".join(translated_sentences), " ".join(filtered_trans_sentences)


def strip_quot(s):
    for p in "\"\'":
        s = s.replace(p, "")
    return s


if __name__ == "__main__":
    total_chars = 0
    topics_seen = {}  # {english: (romanian_tokenized, romanian_non_tokenized, romanian_filtered), ...}
    for csv_name in glob("../VAST/*.csv"):
        print(f"Processing {csv_name}...")
        with open(csv_name, "r") as csv_file:
            reader = DictReader(csv_file)
            rows_sorted = sorted(reader, key=lambda r: (r['post']))
            last_post = (
                None,
                (None, None, None)
            )  # (english, (romanian_tokenized, romanian_non_tokenized, romanian_filtered))

            translated_rows = []
            for row in rows_sorted:
                #if total_chars > 1000:  # Comment out for actual run
                #    continue  # Comment out for actual run
                post = strip_quot(row["post"])  # Strip quotation marks and asterisks for neater translation
                topic = strip_quot(row["new_topic"])
                if post != last_post[0]:
                    total_chars += len(post)
                    last_post = (
                        post,
                        translate_and_tokenize(post)
                    )
                if topic not in topics_seen.keys():
                    total_chars += len(topic)
                    topics_seen[topic] = translate_and_tokenize(topic)
                translated_row = row
                translated_row.pop('pos_text', None)  # remove pos_text
                translated_row["post"] = last_post[1][1]
                translated_row["new_topic"] = topics_seen[topic][1]
                translated_row["topic_str"] = topics_seen[topic][1]
                translated_row["text"] = last_post[1][0]
                translated_row["topic"] = topics_seen[topic][0]
                translated_row["text_s"] = last_post[1][2]
                translated_rows.append(translated_row)

            output_filename = csv_name.split("/")[-1]
            with open(output_filename, "w+") as output_file:
                writer = DictWriter(output_file, fieldnames=translated_rows[0].keys())
                writer.writeheader()
                for row in translated_rows:
                    writer.writerow(row)
