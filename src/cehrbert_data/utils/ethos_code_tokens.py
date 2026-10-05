"""ETHOS style tokens for ICD-10-CM, ICD-10-PCS and ATC codes.

The codes are split into the same parts as in ethos-ares (ethos/tokenize/omop/preprocessors.py), but
the category is written as its code and not as its description:

    ICD10CM/I21.9     -> ICD//CM//I21, ICD//CM//3-6//9
    ICD10CM/S72.001A  -> ICD//CM//S72, ICD//CM//3-6//001, ICD//CM//SFX//A
    ICD10PCS/0DTJ4ZZ  -> ICD//PCS//0, ICD//PCS//D, ICD//PCS//T, ICD//PCS//J, ICD//PCS//4,
                         ICD//PCS//Z, ICD//PCS//Z
    ATC/N02BE01       -> ATC//N02, ATC//4//B, ATC//SFX//E01

The tokens come from rules on the code alone and no mapping file is needed. ethos-ares has
`ICD//CM//ACUTE_MYOCARDIAL_INFARCTION` and `ATC//N02//ANALGESICS` for the first token. The parts after
it are the same, and each of those category tokens stands for exactly one category here.

A malformed ICD-10-CM or ICD-10-PCS code gives no tokens, as in ethos-ares, which drops them.

ICD-9 codes are not converted to ICD-10 here and keep the tokens of cehrbert_data.
"""

import re
from typing import List, Optional, Tuple

# an ICD-10-CM category: a letter, a digit and a digit or letter (I21, F32, I5A)
_ICD10CM_CATEGORY = re.compile(r"[A-Z][0-9][0-9A-Z]")


def icd10cm_tokens(concept_code: Optional[str]) -> List[str]:
    """The category (the first three characters), the characters 4-6 and the rest, by rules only.

    There is no list of ICD-10-CM codes, so a code padded with zeros (I10.00) keeps them
    (ICD//CM//I10, ICD//CM//3-6//00), where ethos-ares removes them until the code is a known one
    (ICD//CM//I10). A code that does not start with a category gives no tokens."""
    if not concept_code:
        return []
    code = concept_code.replace(".", "").upper()
    if not _ICD10CM_CATEGORY.match(code):
        return []
    tokens = [f"ICD//CM//{code[:3]}"]
    if code[3:6]:
        tokens.append(f"ICD//CM//3-6//{code[3:6]}")
    if code[6:]:
        tokens.append(f"ICD//CM//SFX//{code[6:]}")
    return tokens


def icd10pcs_tokens(concept_code: Optional[str]) -> List[str]:
    """One token per character of the 7 character code; placeholders such as NoPCS give none."""
    if not concept_code:
        return []
    code = concept_code.upper()
    if not re.fullmatch(r"[0-9A-Z]{7}", code):
        return []
    return [f"ICD//PCS//{c}" for c in code]


def atc_tokens(concept_code: Optional[str]) -> List[str]:
    """The first three characters, the fourth character and the rest."""
    if not concept_code:
        return []
    code = concept_code.upper()
    tokens = [f"ATC//{code[:3]}"]
    if code[3:4]:
        tokens.append(f"ATC//4//{code[3:4]}")
    if code[4:]:
        tokens.append(f"ATC//SFX//{code[4:]}")
    return tokens


def get_spark_udfs() -> Tuple:
    """The Spark UDFs (icd10cm, icd10pcs, atc), each from a concept code to an array of tokens."""
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    array_of_strings = T.ArrayType(T.StringType())
    return (
        F.udf(icd10cm_tokens, array_of_strings),
        F.udf(icd10pcs_tokens, array_of_strings),
        F.udf(atc_tokens, array_of_strings),
    )
