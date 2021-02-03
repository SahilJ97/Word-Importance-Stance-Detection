from csv import DictReader
from data.turk import print_weights_html

hits = [
    "31D0ZWOD0CAIHENX6X5OJ4WTE4O0AI",
    "32204AGAADNU256WQ54JLI6DYG3HGD",
    "32FESTC2NJ1MQFWBOIFZE6AOUWDCUZ",
    "33P2GD6NRP3Z4R2FGWPYJAY65XKHKG",
    "34ZTTGSNJZZLFU0AT19VFWAF43CHQ8",
    "36JW4WBR08VOO5A1FLY13X61X13FHC",
    "375VSR8FVYK7IGF4C242KTOPWFUZRG",
    "37ZQELHEQ29M4BOT46XR37N6GEOMNZ",
    "38O9DZ0A64YOQME8V7SQMNNMFJI26E",
    "39XCQ6V3K0FGPJXVDQUOKUQYZT956I",
    "3EHIMLB7F9AF9Q0E77SVT5PFAXQH8X",
    "3EN4YVUOUE31FXWC3F78LHMNEY2XJP",
    "3H781YYV6VTMN010K4VKZYOPWUJET3",
    "3HEA4ZVWVFXK6FXH87KJSOCY3YY552",
    "3I4E7AFQ2MAUYGNSY56BWE1H12FJTT",
    "3LCXHSGDLVHL80OLV4QUPTS3UB5ESE",
    "3P6ENY9P7B78YWCRGT4M6UJTDTVHI5",
    "3PA41K45VPF3MTTJGF8UOEVBL4U7P5",
    "3TC2K6WK9IDBU0XWOMPO61JDLE8280",
    "3TFJJUELSJ0D63N95Y5T8NFWB5T2C6",
    "3WJGKMRWVKK09DLP2NPA46608WICDP",
    "3YD0MU1NC4CWZPNHRWQXYJ6GMIUA7S",
    "31YWE12TE2N8V2VA26IBIRHEN3Q7XG",
    "33Q5P9PUSRX3CEQLU5E7VYP3WEFCZB",
    "351S7I5UGB7W33I819KZNFVPCKGNJJ",
    "366FYU4PTI0NKHBDQVHSY71TDHFKEQ",
    "375VSR8FVYK7IGF4C242KTOPWFIRZW",
    "37SDSEDINBD45FF05IPA1I98T8H81C",
    "3CKVGCS3PIGMHA9RNXRF44QZ0AIS0T",
    "3CVDZS288JBAGJBC6IEEV2EXENPFM8",
    "3EKZL9T8YAXHLTQROJ80VRLY0OYCHL",
    "3HFWPF5AKBUIY28O6K9Y62O63M2S3O",
    "3JGHED38EF2XV7LK724D15ONP9K7YM",
    "3JYPJ2TAYKJG1M8PYY66C68SJS0PFE",
    "3K2CEDRACDCKZOTD12J3C6TY5WXMTN",
    "3P7RGTLO6GOKU4U1AQF62OB2S2HKAV",
    "3TKXBROM5VL4P27H4T77I7Z8QLMIJY",
    "3TLFH2L6YBZUSKPKRK0JXXF0ZRG2TC",
    "3UEBBGULPHZTTRGLFK50MCVRR9SFU0",
    "3XWUWJ18TN1IC9DBA90Z8CWU1W7UUE",
    "3Y3N5A7N4IKGGAS7B862KNLDKYFMYC",
]

if __name__ == "__main__":
    with open("2_wrong.html", "w+") as out_f:
        out_f.write(print_weights_html.header)
        reached_section = False
        with open("output.txt", "r") as f:
            for line in f:
                if "Weights and raw aggregation scores" in line:
                    reached_section = True
                if reached_section and line.strip() in hits:
                    hit_id = line.strip()
                    out_f.write("<p>" + hit_id + "</p>")

                    # find topic and label
                    with open("two_batch_results.csv", "r") as result_f:
                        r = DictReader(result_f)
                        for row in r:
                            if row["HITId"] == hit_id:
                                out_f.write("<p>" + row["Input.topic"] + "</p>")
                                out_f.write("<p>label: " + row["Input.label"] + "</p>")
                                break

                    for i in range(3):
                        out_f.write(next(f))
                        out_f.write("<hr>\n")


