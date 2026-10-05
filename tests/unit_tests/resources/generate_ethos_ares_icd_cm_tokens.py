"""Writes ethos_ares_icd_cm_tokens.csv, the tokens that ethos-ares gives to ICD codes.

It runs DiagnosesData.split_icd_codes of ethos-ares itself, so it needs the ethos-ares environment (polars):

    python generate_ethos_ares_icd_cm_tokens.py <ethos-ares>/src <codes.csv> ethos_ares_icd_cm_tokens.csv

codes.csv has the columns vocabulary_id and concept_code (the ICD codes of an OMOP concept table). The edge cases below
are always added.
"""
import csv
import json
import sys

import polars as pl

ethos_ares_src, codes_csv, output_csv = sys.argv[1:4]
sys.path.insert(0, ethos_ares_src)
from ethos.tokenize.omop.preprocessors import DiagnosesData  # noqa: E402

EDGE_CASES = [
    ("ICD10CM", c) for c in [
        "I10", "I10.00", "E11.90", "E11.9", "E11.40", "R10.90", "M54.50", "M54.5", "I21.9", "S72.001A", "I50", "I5A",
        "K31.A0", "U09.9", "Z09", "F32", "F32.A", "C56.0", "Z59.00", "G92.00", "QQQ.1", "A", "I1", "i25.10", "T36.0X1A",
    ]
] + [
    ("ICD9CM", c) for c in [
        "585", "250.00", "250", "428.22", "428", "V45.1", "V45", "E849.0", "001.0", "0010", "9999", "XYZ", "799.9", "78650",
    ]
]

codes = {(r["vocabulary_id"], r["concept_code"]) for r in csv.DictReader(open(codes_csv))} | set(EDGE_CASES)
codes = sorted(codes)
df = pl.DataFrame(
    {
        "subject_id": list(range(len(codes))),
        "time": [0] * len(codes),
        "code": [f"{vocabulary}/{code}" for vocabulary, code in codes],
    }
)
result = DiagnosesData.split_icd_codes(df)
tokens = {i: [] for i in range(len(codes))}
for subject_id, code in zip(result["subject_id"], result["code"]):
    tokens[subject_id].append(code)

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["vocabulary_id", "concept_code", "tokens"])
    for i, (vocabulary, code) in enumerate(codes):
        writer.writerow([vocabulary, code, json.dumps(tokens[i])])
print(f"{len(codes)} codes, {sum(1 for t in tokens.values() if not t)} of them without tokens")
