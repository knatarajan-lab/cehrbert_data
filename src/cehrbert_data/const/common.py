PERSON = "person"
VISIT_OCCURRENCE = "visit_occurrence"
CONDITION_OCCURRENCE = "condition_occurrence"
PROCEDURE_OCCURRENCE = "procedure_occurrence"
DRUG_EXPOSURE = "drug_exposure"
DEVICE_EXPOSURE = "device_exposure"
OBSERVATION = "observation"
MEASUREMENT = "measurement"
PROCESSED_MEASUREMENT = "processed_measurement"
PROCESSED_OBSERVATION = "processed_observation"
PROCESSED_DEVICE = "processed_device"
CATEGORICAL_MEASUREMENT = "categorical_measurement"
OBSERVATION_PERIOD = "observation_period"
DEATH = "death"
CDM_TABLES = [
    PERSON,
    VISIT_OCCURRENCE,
    CONDITION_OCCURRENCE,
    PROCEDURE_OCCURRENCE,
    DRUG_EXPOSURE,
    DEVICE_EXPOSURE,
    OBSERVATION,
    MEASUREMENT,
    CATEGORICAL_MEASUREMENT,
    OBSERVATION_PERIOD,
    DEATH,
]
REQUIRED_MEASUREMENT = "required_measurement"
NUMERIC_MEASUREMENT_STATS = "numeric_measurement_stats"
UNKNOWN_CONCEPT = "[UNKNOWN]"
NA = "N/A"
CONCEPT = "concept"
CONCEPT_ANCESTOR = "concept_ancestor"
MEASUREMENT_QUESTION_PREFIX = "1-Question:"
MEASUREMENT_ANSWER_PREFIX = "2-Answer:"
