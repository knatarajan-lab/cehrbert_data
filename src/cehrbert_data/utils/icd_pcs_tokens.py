"""ICD-PCS tokens of ETHOS, the same as in ethos-ares (ProcedureData.split_icd_codes in
ethos/tokenize/omop/preprocessors.py).

A procedure code is converted to ICD-10-PCS and split into its seven characters:

    ICD10PCS/0DTJ4ZZ -> ICD//PCS//0, ICD//PCS//D, ICD//PCS//T, ICD//PCS//J, ICD//PCS//4, ICD//PCS//Z, ICD//PCS//Z

- ICD-9 procedure codes (ICD9Proc) are converted to ICD-10-PCS with the ICD-9 to ICD-10-PCS mapping of ethos-ares, a
  code with several ICD-10-PCS codes gets the first one. The dot is removed and the trailing zeros are removed until the
  code is in the mapping.
- A code that is not made of seven digits or letters, e.g. the placeholder NoPCS, or an ICD-9 code that is not in the
  mapping gives no tokens, which drops the procedure.

The mapping is the file of ethos-ares in `cehrbert_data/resources`.
"""

import csv
import functools
import gzip
import os
import re
from typing import Dict, Optional, Tuple

from cehrbert_data.utils.icd_cm_tokens import _RESOURCES_DIR, normalize_icd_code

_ICD10PCS_CODE = re.compile(r"[0-9A-Z]{7}")


@functools.lru_cache(maxsize=1)
def _icd9_to_icd10_pcs() -> Dict[str, str]:
    icd10_codes: Dict[str, list] = {}
    with gzip.open(os.path.join(_RESOURCES_DIR, "icd_pcs_9_to_10_mapping.csv.gz"), "rt", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            icd9, icd10 = row[0], row[1] if len(row) > 1 else ""
            if icd10 in ("", "NoDx", "NoPCS"):
                continue
            icd10_codes.setdefault(icd9, []).append(icd10)
    return {icd9: sorted(codes)[0] for icd9, codes in icd10_codes.items()}


@functools.lru_cache(maxsize=None)
def icd_pcs_tokens(vocabulary_id: Optional[str], concept_code: Optional[str]) -> Tuple[str, ...]:
    """The tokens of an ICD10PCS or ICD9Proc code; none when the code is dropped."""
    if not concept_code:
        return ()
    if vocabulary_id == "ICD9Proc":
        mapping = _icd9_to_icd10_pcs()
        icd_code = mapping.get(normalize_icd_code(concept_code, mapping))
    else:
        icd_code = concept_code.upper()
    if icd_code is None or not _ICD10PCS_CODE.fullmatch(icd_code):
        return ()
    return tuple(f"ICD//PCS//{character}" for character in icd_code)


def get_spark_udf():
    """The Spark UDF of (vocabulary_id, concept_code) to the array of the tokens."""
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    return F.udf(lambda vocabulary_id, concept_code: list(icd_pcs_tokens(vocabulary_id, concept_code)),
                 T.ArrayType(T.StringType()))
