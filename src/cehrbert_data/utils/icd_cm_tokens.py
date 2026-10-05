"""ICD-CM tokens of ETHOS, the same as in ethos-ares (DiagnosesData.split_icd_codes in
ethos/tokenize/omop/preprocessors.py).

A diagnosis code is converted to ICD-10-CM and split into the category (named after its description), the characters
4-6 and the suffix:

    ICD10CM/I25.10 -> ICD//CM//CHRONIC_ISCHEMIC_HEART_DISEASE, ICD//CM//3-6//10
    ICD10CM/S72.001A -> ICD//CM//FRACTURE_OF_FEMUR, ICD//CM//3-6//001, ICD//CM//SFX//A

- ICD-9-CM codes are converted to ICD-10-CM with the ICD-9 to ICD-10 mapping of ethos-ares. A code with several
  ICD-10-CM codes gets the category of the first one, and a code without an exact match gets the category of its first
  subcode.
- The dots are removed from an ICD-10-CM code and the trailing zeros are removed until the code is a known one, because
  some sources pad the codes with zeros (I10.00 -> I10). The known codes are those of the January 2021 list of
  ethos-ares, so a code that was added later and ends with a zero (M54.50) is also shortened (M54.5).
- A code with an unknown category, or with no ICD-10-CM code, gives no tokens, which drops the diagnosis.

The code names and the mapping are the files of ethos-ares in `cehrbert_data/resources`.
"""

import bisect
import csv
import functools
import gzip
import os
import re
from typing import Container, Dict, List, Optional, Tuple

_RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")

# The codes that are not in the code list of ethos-ares
_COMPLEMENTARY_CODE_TO_NAME = {
    "F32": "Major depressive disorder, single episode",
    "F32A": "Depression, unspecified",
    "G928": "Other toxic encephalopathy",
    "G929": "Unspecified toxic encephalopathy",
    "I5A": "Non-ischemic myocardial injury (non-traumatic)",
    "K31A": "Gastric intestinal metaplasia",
    "L24A9": "Irritant contact dermatitis due to friction or contact with body fluids",
    "L24B": "Irritant contact dermatitis related to stoma or fistula",
    "M350": "Sicca syndrome [Sjogren]",
    "P099": "Abnormal findings on neonatal screening, unspecified",
    "R051": "Acute cough",
    "R052": "Subacute cough",
    "R053": "Chronic cough",
    "R054": "Cough syncope",
    "R058": "Other specified cough",
    "R059": "Cough, unspecified",
    "U09": "Post COVID-19 condition",
    "U099": "Post COVID-19 condition, unspecified",
    "W458X": "Lid of can entering through skin",
    "Z09": "Encounter for follow-up examination after completed treatment",
    "Z2252": "Carrier of viral hepatitis",
}


@functools.lru_cache(maxsize=1)
def _code_to_name() -> Dict[str, str]:
    """ICD-10-CM code (without the dot) to the description of the code."""
    code_to_name = {}
    with gzip.open(os.path.join(_RESOURCES_DIR, "icd10cm-order-Jan-2021.csv.gz"), "rt", newline="") as f:
        for row in csv.DictReader(f):
            code_to_name.setdefault(row["code"], row["long"])
    code_to_name.update(_COMPLEMENTARY_CODE_TO_NAME)
    return code_to_name


@functools.lru_cache(maxsize=1)
def _icd9_to_icd10() -> Tuple[Dict[str, str], List[str]]:
    """ICD-9-CM code to ICD-10-CM code, and the ICD-9-CM codes in order."""
    icd10_codes: Dict[str, List[str]] = {}
    with gzip.open(os.path.join(_RESOURCES_DIR, "icd_cm_9_to_10_mapping.csv.gz"), "rt", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            icd9, icd10 = row[0], row[1] if len(row) > 1 else ""
            if icd10 in ("", "NoDx", "NoPCS"):
                continue
            icd10_codes.setdefault(icd9, []).append(icd10)
    mapping = {
        icd9: sorted(codes)[0][:3] if len(codes) > 1 else codes[0] for icd9, codes in icd10_codes.items()
    }
    return mapping, sorted(mapping)


def _normalize(raw_code: str, known_codes: Container[str]) -> str:
    code = raw_code.replace(".", "").upper()
    while code not in known_codes and code.endswith("0") and len(code) > 3:
        code = code[:-1]
    return code


def _convert_icd9_to_icd10(raw_code: str) -> Optional[str]:
    mapping, icd9_codes = _icd9_to_icd10()
    code = _normalize(raw_code, mapping)
    if code in mapping:
        return mapping[code]
    # The ICD-10-CM category of the first of the subcodes, e.g. 585 -> N18
    start = bisect.bisect_left(icd9_codes, code)
    subcodes = []
    while start < len(icd9_codes) and icd9_codes[start].startswith(code):
        subcodes.append(mapping[icd9_codes[start]])
        start += 1
    return sorted(subcodes)[0][:3] if subcodes else None


def _unify_code_name(token: str) -> str:
    return re.sub(r"[,.]", "", token.upper()).replace(" ", "_")


@functools.lru_cache(maxsize=None)
def icd_cm_tokens(vocabulary_id: Optional[str], concept_code: Optional[str]) -> Tuple[str, ...]:
    """The tokens of an ICD10CM or ICD9CM code; none when the code is dropped."""
    if not concept_code:
        return ()
    if vocabulary_id == "ICD9CM":
        icd_code = _convert_icd9_to_icd10(concept_code)
    else:
        icd_code = _normalize(concept_code, _code_to_name())
    if icd_code is None:
        return ()
    category = _code_to_name().get(icd_code[:3])
    if not category:
        return ()
    parts = [("", category), ("3-6//", icd_code[3:6]), ("SFX//", icd_code[6:])]
    return tuple(_unify_code_name(f"ICD//CM//{prefix}{part}") for prefix, part in parts if part != "")


def get_spark_udf():
    """The Spark UDF of (vocabulary_id, concept_code) to the array of the tokens."""
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    return F.udf(lambda vocabulary_id, concept_code: list(icd_cm_tokens(vocabulary_id, concept_code)),
                 T.ArrayType(T.StringType()))
