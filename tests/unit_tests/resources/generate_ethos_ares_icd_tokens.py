"""Writes ethos_ares_icd_cm_tokens.csv and ethos_ares_icd_pcs_tokens.csv, the tokens that ethos-ares gives to ICD codes.

It runs DiagnosesData.split_icd_codes (ICD10CM, ICD9CM) and ProcedureData.split_icd_codes (ICD10PCS, ICD9Proc) of
ethos-ares itself, so it needs the ethos-ares environment (polars):

    python generate_ethos_ares_icd_tokens.py <ethos-ares>/src <codes.csv> <output directory>

codes.csv has the columns vocabulary_id and concept_code (the ICD codes of an OMOP concept table). The edge cases below
are always added. The ICD-PCS file has the seven characters of the code, as every token is ICD//PCS//<character>.
"""
import csv
import json
import os
import sys

import polars as pl

ethos_ares_src, codes_csv, output_dir = sys.argv[1:4]
sys.path.insert(0, ethos_ares_src)
from ethos.tokenize.omop.preprocessors import DiagnosesData, ProcedureData  # noqa: E402

EDGE_CASES = {
    "ICD10CM": [
        "I10", "I10.00", "E11.90", "E11.9", "E11.40", "R10.90", "M54.50", "M54.5", "I21.9", "S72.001A", "I50", "I5A",
        "K31.A0", "U09.9", "Z09", "F32", "F32.A", "C56.0", "Z59.00", "G92.00", "QQQ.1", "A", "I1", "i25.10", "T36.0X1A",
    ],
    "ICD9CM": [
        "585", "250.00", "250", "428.22", "428", "V45.1", "V45", "E849.0", "001.0", "0010", "9999", "XYZ", "799.9", "78650",
    ],
    "ICD10PCS": ["0DTJ4ZZ", "0dtj4zz", "02HV33Z", "NoPCS", "0DTJ4Z", "0DTJ4ZZZ", "0DTJ4Z!", "ABCDEFG", "0000000"],
    "ICD9Proc": ["89.02", "8902", "00.10", "0010", "00.1", "99.99", "9999", "3.0", "36.1", "3610", "36.10", "XYZ", "00", "0"],
}

codes = {(r["vocabulary_id"], r["concept_code"]) for r in csv.DictReader(open(codes_csv))}
codes |= {(vocabulary, code) for vocabulary, edge_codes in EDGE_CASES.items() for code in edge_codes}


def tokens_of(split_function, vocabularies):
    selected = sorted(c for c in codes if c[0] in vocabularies)
    df = pl.DataFrame(
        {
            "subject_id": list(range(len(selected))),
            "time": [0] * len(selected),
            "code": [f"{vocabulary}/{code}" for vocabulary, code in selected],
        }
    )
    result = split_function(df)
    tokens = {i: [] for i in range(len(selected))}
    for subject_id, code in zip(result["subject_id"], result["code"]):
        tokens[subject_id].append(code)
    return [(vocabulary, code, tokens[i]) for i, (vocabulary, code) in enumerate(selected)]


cm = tokens_of(DiagnosesData.split_icd_codes, {"ICD10CM", "ICD9CM"})
with open(os.path.join(output_dir, "ethos_ares_icd_cm_tokens.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["vocabulary_id", "concept_code", "tokens"])
    for vocabulary, code, tokens in cm:
        writer.writerow([vocabulary, code, json.dumps(tokens)])

pcs = tokens_of(ProcedureData.split_icd_codes, {"ICD10PCS", "ICD9Proc"})
with open(os.path.join(output_dir, "ethos_ares_icd_pcs_tokens.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["vocabulary_id", "concept_code", "characters"])
    for vocabulary, code, tokens in pcs:
        assert all(t.startswith("ICD//PCS//") and len(t) == len("ICD//PCS//") + 1 for t in tokens)
        writer.writerow([vocabulary, code, "".join(t[-1] for t in tokens)])

print(f"ICD-CM: {len(cm)} codes, {sum(1 for _, _, t in cm if not t)} without tokens | "
      f"ICD-PCS: {len(pcs)} codes, {sum(1 for _, _, t in pcs if not t)} without tokens")
