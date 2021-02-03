from csv import DictWriter, DictReader

hits = [
    "30ZKOOGW2YHIO38F2G9OX9F3JWUA1V",
    "356TQKY9XH8IG1U2UY6J2S0LWUU87M",
    "36D1BWBEHPCQFPZ0INZ7BJK85JX2MO",
    "36KM3FWE3TN0YCPZZUT49DU3DWI07R",
    "374UMBUHN707Q2HMODPP9AOMBI6CT8",
    "37SDSEDINBD45FF05IPA1I98T8U81P",
    "386T3MLZLP64YQUI9ZLANDQYOGO80M",
    "3KWGG5KP6LD3D7R2FGODNWTUHAACM7",
    "3M47JKRKCZCZC1GXBO1RWE0P1JA86N",
    "3N5YJ55YXIEL2AF4737MUGS6MXZAN3",
    "3NSCTNUR21YHCL9Y33B00ZSKVCZ5A8",
    "3PEG1BH7AG2PDEZTG4F2XZV8AKFKB3",
    "3S1L4CQSFZG9EI2N9FO9MBJXK6DAFW",
    "3TZ0XG8CBWVMUKIV9V43GORRRKI89Y",
    "3X878VYTIGTRVG4P2CCPCKH7JY7F7A",
    "33CLA8O0MKM1DZO4BQAVH187KAOFRW",
    "36FQTHX3Z52JHGRMVYN3GDE6ZYGB36",
    "37VHPF5VYEEOSIPYLFEUIESZX588C2",
    "3B6F54KMR4NMOZU9JGVFI99952OS1W",
    "3BDORL6HKMOUAN4UKOKU9VN4NXWCRI",
    "3DGDV62G7QKW0SWDA3RZSNE3AOV2P7",
    "3H4IKZHALDTSMT9TG7CRGMC6L4LNNN",
    "3IKMEYR0LY6VP1ZXXBMRW6S4GEPK2O",
    "3JMNNNO3B3FMK1TEMP5UKHMZ3LY2W2",
    "3JMQI2OLF1GJ0HIGRDYOKH0C70JND5",
    "3JTPR5MTZUNNOYIKFUOPUDVG0OG5KA",
    "3KLL7H3EGFCDK2WRF035LJ9NLVEHVP",
    "3M0556243UVGY1WCIJTE8H3XJDZFN8",
    "3OYHVNTV5V99W8O2K9KO3JV7HJLKOU",
    "3ULIZ0H1VCGLIXWY7KI7PYZHRIG517",
    "3VIVIU06FMNUQAD27QKZYZRPUFHIM2",
    "3WRAAIUSBLAQE4T899UVSIJ4OBWAXA",
    "3ZUE82NE0CCVVU98CH4VQKD2UI08FD",
]

if __name__ == "__main__":
    with open("3_wrong.csv", "w+") as out_f:
        writer = DictWriter(out_f, fieldnames=["HITId", "topic", "argument", "label"])
        writer.writeheader()
        for in_file in ["first_batch_results.csv", "second_batch_results.csv"]:
            with open(in_file, "r") as f:
                reader = DictReader(f)
                for row in reader:
                    if row["HITId"] not in hits:
                        continue
                    hits.remove(row["HITId"])
                    out_row = {
                        "HITId": row["HITId"],
                        "topic": row["Input.topic"],
                        "argument": row["Input.argument"],
                        "label": row["Input.label"]
                    }
                    writer.writerow(out_row)

