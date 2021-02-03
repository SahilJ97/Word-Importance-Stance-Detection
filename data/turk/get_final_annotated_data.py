from csv import DictReader, DictWriter

delete_hits = {'33Q5P9PUSRX3CEQLU5E7VYP3WEFCZB', '3TFJJUELSJ0D63N95Y5T8NFWB5T2C6', '3M47JKRKCZCZC1GXBO1RWE0P1JA86N', '3JMQI2OLF1GJ0HIGRDYOKH0C70JND5', '3WRAAIUSBLAQE4T899UVSIJ4OBWAXA', '3P7RGTLO6GOKU4U1AQF62OB2S2HKAV', '3H781YYV6VTMN010K4VKZYOPWUJET3', '3KLL7H3EGFCDK2WRF035LJ9NLVEHVP', '3B6F54KMR4NMOZU9JGVFI99952OS1W', '374UMBUHN707Q2HMODPP9AOMBI6CT8', '3NSCTNUR21YHCL9Y33B00ZSKVCZ5A8', '36KM3FWE3TN0YCPZZUT49DU3DWI07R', '37ZQELHEQ29M4BOT46XR37N6GEOMNZ', '3JYPJ2TAYKJG1M8PYY66C68SJS0PFE', '3WJGKMRWVKK09DLP2NPA46608WICDP', '386T3MLZLP64YQUI9ZLANDQYOGO80M', '375VSR8FVYK7IGF4C242KTOPWFIRZW', '3S1L4CQSFZG9EI2N9FO9MBJXK6DAFW', '3I4E7AFQ2MAUYGNSY56BWE1H12FJTT', '3JMNNNO3B3FMK1TEMP5UKHMZ3LY2W2', '3Y3N5A7N4IKGGAS7B862KNLDKYFMYC', '3JTPR5MTZUNNOYIKFUOPUDVG0OG5KA', '3N5YJ55YXIEL2AF4737MUGS6MXZAN3', '37SDSEDINBD45FF05IPA1I98T8U81P', '3ZUE82NE0CCVVU98CH4VQKD2UI08FD', '32FESTC2NJ1MQFWBOIFZE6AOUWDCUZ', '3UEBBGULPHZTTRGLFK50MCVRR9SFU0', '366FYU4PTI0NKHBDQVHSY71TDHFKEQ', '3EN4YVUOUE31FXWC3F78LHMNEY2XJP', '3EKZL9T8YAXHLTQROJ80VRLY0OYCHL', '37VHPF5VYEEOSIPYLFEUIESZX588C2', '3IKMEYR0LY6VP1ZXXBMRW6S4GEPK2O', '3YD0MU1NC4CWZPNHRWQXYJ6GMIUA7S', '33P2GD6NRP3Z4R2FGWPYJAY65XKHKG', '3HEA4ZVWVFXK6FXH87KJSOCY3YY552', '36D1BWBEHPCQFPZ0INZ7BJK85JX2MO', '3VIVIU06FMNUQAD27QKZYZRPUFHIM2', '3CVDZS288JBAGJBC6IEEV2EXENPFM8', '3EHIMLB7F9AF9Q0E77SVT5PFAXQH8X'}
flip_hits = {'3TZ0XG8CBWVMUKIV9V43GORRRKI89Y', '3H4IKZHALDTSMT9TG7CRGMC6L4LNNN', '3DGDV62G7QKW0SWDA3RZSNE3AOV2P7', '356TQKY9XH8IG1U2UY6J2S0LWUU87M', '3BDORL6HKMOUAN4UKOKU9VN4NXWCRI', '3ULIZ0H1VCGLIXWY7KI7PYZHRIG517', '3M0556243UVGY1WCIJTE8H3XJDZFN8', '36FQTHX3Z52JHGRMVYN3GDE6ZYGB36', '3X878VYTIGTRVG4P2CCPCKH7JY7F7A', '3KWGG5KP6LD3D7R2FGODNWTUHAACM7', '33CLA8O0MKM1DZO4BQAVH187KAOFRW', '3OYHVNTV5V99W8O2K9KO3JV7HJLKOU', '30ZKOOGW2YHIO38F2G9OX9F3JWUA1V', '3PEG1BH7AG2PDEZTG4F2XZV8AKFKB3'}
wqs = """
A2JP9IKRHNLRPI	0.568 0.323
A1198W1SPF1R4	0.437 0.191
ATJVY9O4CY5EX	0.585 0.342
A3IRHO9RKEV0L7	0.648 0.42
A416NAWAPDAMV	0.603 0.364
A70TBKPGWQZO4	0.471 0.221
AKSJ3C5O3V9RB	0.655 0.429
A249LDVPG27XCE	0.624 0.389
A2WPHVMLLEV5ZB	0.299 0.09
A3O1I9MATO3ZZN	0.562 0.316
A3GMRPF5MCQVGV	0.51 0.26
ADJ9I7ZBFYFH7	0.595 0.354
A35C0II2FFV18S	0.565 0.319
A2NAKIXS3DVGAA	0.777 0.604
A3O4WYGKH31OX	0.486 0.236
ACBK30I7R42NI	0.665 0.442
A1FGKIKJYSL1MI	0.667 0.444
A2GO0EY5FW18AG	0.551 0.304
A3V9JFVZQ2XF4Y	0.312 0.098
A3TUJHF9LW3M8N	0.806 0.65
AYXDE087J2KO	0.326 0.106
A2BCK6M4SPHRRH	0.451 0.204
A3HP7L563UF5DW	0.708 0.502
ANONYKAVCEV2H	0.459 0.211
ABN9IUIM1YA7Y	0.481 0.232
A5BCSSPC180AG	0.481 0.231
"""  # Worker ID, WQS, squared WQS

if __name__ == "__main__":

    with open("output.txt", "r") as f:
        start_looking = False
        all_weights = {}
        for line in f:
            if "Weights and raw aggregation scores" in line:
                start_looking = True
            if start_looking and len(line.strip().split()) == 1:
                hit_id = line.strip().split()[0]
                word_weights = eval(next(f))
                all_weights[hit_id] = word_weights
                # weights ignores tokens that lack an alphabetic character. need to re-map to tokenized_argument.
                # when loading data for training, compute alignments between spacy tokens and transformer tokens
    with open("two_batch_results.csv", "r") as in_f:
        with open("special_datapoints.txt", "w") as special_f:
            # list of VAST/vast_train.csv datapoints to NOT load
            # (any datapoint that is either deleted or endowed with word importance labels)
            with open("processed_annotated.csv", "w") as ann_f:
                fieldnames = ["new_id", "argument", "tokenized_argument", "topic", "label", "weights"]
                reader = DictReader(in_f)
                writer = DictWriter(ann_f, fieldnames=fieldnames)
                writer.writeheader()
                seen_hits = []
                for in_row in reader:
                    hit_id, new_id = in_row["HITId"], in_row["Input.new_id"]
                    if hit_id in seen_hits:
                        continue
                    seen_hits.append(hit_id)
                    special_f.write(new_id + "\n")
                    if hit_id in delete_hits:
                        continue
                    out_row = {}
                    for fieldname in ["new_id", "argument", "tokenized_argument", "topic", "label"]:
                        out_row[fieldname] = in_row[f"Input.{fieldname}"]
                    if hit_id in flip_hits:
                        if out_row["label"] == "0":
                            out_row["label"] = "1"
                        else:
                            out_row["label"] = "0"
                    out_row["weights"] = all_weights[hit_id]
                    writer.writerow(out_row)
