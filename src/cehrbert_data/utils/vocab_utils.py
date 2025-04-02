from pyspark.sql import functions as F, types as T


def build_ancestry_table_for(spark, concept_ids):
    initial_query = """
    SELECT
        cr.concept_id_1 AS ancestor_concept_id,
        cr.concept_id_2 AS descendant_concept_id,
        1 AS distance
    FROM global_temp.concept_relationship AS cr
    WHERE cr.concept_id_1 in ({concept_ids}) AND cr.relationship_id = 'Subsumes'
    """

    recurring_query = """
    SELECT
        i.ancestor_concept_id AS ancestor_concept_id,
        cr.concept_id_2 AS descendant_concept_id,
        i.distance + 1 AS distance
    FROM global_temp.ancestry_table AS i
    JOIN global_temp.concept_relationship AS cr
        ON i.descendant_concept_id = cr.concept_id_1 AND cr.relationship_id = 'Subsumes'
    LEFT JOIN global_temp.ancestry_table AS i2
        ON cr.concept_id_2 = i2.descendant_concept_id
    WHERE i2.descendant_concept_id IS NULL
    """

    union_query = """
    SELECT
        *
    FROM global_temp.ancestry_table

    UNION

    SELECT
        *
    FROM global_temp.candidate
    """

    ancestry_table = spark.sql(initial_query.format(concept_ids=",".join([str(c) for c in concept_ids])))
    ancestry_table.createOrReplaceGlobalTempView("ancestry_table")

    candidate_set = spark.sql(recurring_query)
    candidate_set.createOrReplaceGlobalTempView("candidate")

    while candidate_set.count() != 0:
        spark.sql(union_query).createOrReplaceGlobalTempView("ancestry_table")
        candidate_set = spark.sql(recurring_query)
        candidate_set.createOrReplaceGlobalTempView("candidate")

    ancestry_table = spark.sql(
        """
    SELECT
        *
    FROM global_temp.ancestry_table
    """
    )

    spark.sql(
        """
    DROP VIEW global_temp.ancestry_table
    """
    )

    return ancestry_table


def get_descendant_concept_ids(spark, concept_ids):
    """
    Query concept_ancestor table to get all descendant_concept_ids for the given list of concept_ids.

    :param spark:
    :param concept_ids:
    :return:
    """
    sanitized_concept_ids = [int(c) for c in concept_ids]
    # Join the sanitized IDs into a string for the query
    concept_ids_str = ",".join(map(str, sanitized_concept_ids))
    # Construct and execute the SQL query using the sanitized string
    descendant_concept_ids = spark.sql(
        f"""
        SELECT DISTINCT
            c.*
        FROM global_temp.concept_ancestor AS ca
        JOIN global_temp.concept AS c
            ON ca.descendant_concept_id = c.concept_id
        WHERE ca.ancestor_concept_id IN ({concept_ids_str})
    """
    )
    return descendant_concept_ids


def roll_up_to_drug_ingredients(drug_exposure, concept, concept_ancestor):
    # lowercase the schema fields
    drug_exposure = drug_exposure.select([F.col(f_n).alias(f_n.lower()) for f_n in drug_exposure.schema.fieldNames()])

    drug_ingredient = (
        drug_exposure.select("drug_concept_id")
        .distinct()
        .join(concept_ancestor, F.col("drug_concept_id") == F.col("descendant_concept_id"))
        .join(concept, F.col("ancestor_concept_id") == F.col("concept_id"))
        .where(concept["concept_class_id"] == "Ingredient")
        .select(F.col("drug_concept_id"), F.col("concept_id").alias("ingredient_concept_id"))
    )

    drug_ingredient_fields = [
        F.coalesce(F.col("ingredient_concept_id"), F.col("drug_concept_id")).alias("drug_concept_id")
    ]
    drug_ingredient_fields.extend(
        [F.col(field_name) for field_name in drug_exposure.schema.fieldNames() if field_name != "drug_concept_id"]
    )

    drug_exposure = drug_exposure.join(drug_ingredient, "drug_concept_id", "left_outer").select(drug_ingredient_fields)

    return drug_exposure


def roll_up_diagnosis(condition_occurrence, concept, concept_relationship):
    list_3dig_code = [
        "3-char nonbill code",
        "3-dig nonbill code",
        "3-char billing code",
        "3-dig billing code",
        "3-dig billing E code",
        "3-dig billing V code",
        "3-dig nonbill E code",
        "3-dig nonbill V code",
    ]

    condition_occurrence = condition_occurrence.select(
        [F.col(f_n).alias(f_n.lower()) for f_n in condition_occurrence.schema.fieldNames()]
    )

    condition_icd = (
        condition_occurrence.select("condition_source_concept_id")
        .distinct()
        .join(concept, (F.col("condition_source_concept_id") == F.col("concept_id")))
        .where(concept["domain_id"] == "Condition")
        .where(concept["vocabulary_id"] != "SNOMED")
        .select(
            F.col("condition_source_concept_id"),
            F.col("vocabulary_id").alias("child_vocabulary_id"),
            F.col("concept_class_id").alias("child_concept_class_id"),
        )
    )

    condition_icd_hierarchy = (
        condition_icd.join(
            concept_relationship,
            F.col("condition_source_concept_id") == F.col("concept_id_1"),
        )
        .join(
            concept,
            (F.col("concept_id_2") == F.col("concept_id")) & (F.col("concept_class_id").isin(list_3dig_code)),
            how="left",
        )
        .select(
            F.col("condition_source_concept_id").alias("source_concept_id"),
            F.col("child_concept_class_id"),
            F.col("concept_id").alias("parent_concept_id"),
            F.col("concept_name").alias("parent_concept_name"),
            F.col("vocabulary_id").alias("parent_vocabulary_id"),
            F.col("concept_class_id").alias("parent_concept_class_id"),
        )
        .distinct()
    )

    condition_icd_hierarchy = condition_icd_hierarchy.withColumn(
        "ancestor_concept_id",
        F.when(
            F.col("child_concept_class_id").isin(list_3dig_code),
            F.col("source_concept_id"),
        ).otherwise(F.col("parent_concept_id")),
    ).dropna(subset="ancestor_concept_id")

    condition_occurrence_fields = [
        F.col(f_n).alias(f_n.lower())
        for f_n in condition_occurrence.schema.fieldNames()
        if f_n != "condition_source_concept_id"
    ]
    condition_occurrence_fields.append(
        F.coalesce(F.col("ancestor_concept_id"), F.col("condition_source_concept_id")).alias(
            "condition_source_concept_id"
        )
    )

    condition_occurrence = (
        condition_occurrence.join(
            condition_icd_hierarchy,
            condition_occurrence["condition_source_concept_id"] == condition_icd_hierarchy["source_concept_id"],
            how="left",
        )
        .select(condition_occurrence_fields)
        .withColumn("condition_concept_id", F.col("condition_source_concept_id"))
    )
    return condition_occurrence


def roll_up_procedure(procedure_occurrence, concept, concept_ancestor):
    def extract_parent_code(concept_code):
        return concept_code.split(".")[0]

    parent_code_udf = F.udf(extract_parent_code, T.StringType())

    procedure_code = (
        procedure_occurrence.select("procedure_source_concept_id")
        .distinct()
        .join(concept, F.col("procedure_source_concept_id") == F.col("concept_id"))
        .where(concept["domain_id"] == "Procedure")
        .select(
            F.col("procedure_source_concept_id").alias("source_concept_id"),
            F.col("vocabulary_id").alias("child_vocabulary_id"),
            F.col("concept_class_id").alias("child_concept_class_id"),
            F.col("concept_code").alias("child_concept_code"),
        )
    )

    # cpt code rollup
    cpt_code = procedure_code.where(F.col("child_vocabulary_id") == "CPT4")

    cpt_hierarchy = (
        cpt_code.join(
            concept_ancestor,
            cpt_code["source_concept_id"] == concept_ancestor["descendant_concept_id"],
        )
        .join(concept, concept_ancestor["ancestor_concept_id"] == concept["concept_id"])
        .where(concept["vocabulary_id"] == "CPT4")
        .select(
            F.col("source_concept_id"),
            F.col("child_concept_class_id"),
            F.col("ancestor_concept_id").alias("parent_concept_id"),
            F.col("min_levels_of_separation"),
            F.col("concept_class_id").alias("parent_concept_class_id"),
        )
    )

    cpt_hierarchy_level_1 = (
        cpt_hierarchy.where(F.col("min_levels_of_separation") == 1)
        .where(F.col("child_concept_class_id") == "CPT4")
        .where(F.col("parent_concept_class_id") == "CPT4 Hierarchy")
        .select(F.col("source_concept_id"), F.col("parent_concept_id"))
    )

    cpt_hierarchy_level_1 = cpt_hierarchy_level_1.join(
        concept_ancestor,
        (cpt_hierarchy_level_1["source_concept_id"] == concept_ancestor["descendant_concept_id"])
        & (concept_ancestor["min_levels_of_separation"] == 1),
        how="left",
    ).select(
        F.col("source_concept_id"),
        F.col("parent_concept_id"),
        F.col("ancestor_concept_id").alias("root_concept_id"),
    )

    cpt_hierarchy_level_1 = cpt_hierarchy_level_1.withColumn(
        "isroot",
        F.when(
            cpt_hierarchy_level_1["root_concept_id"] == 45889197,
            cpt_hierarchy_level_1["source_concept_id"],
        ).otherwise(cpt_hierarchy_level_1["parent_concept_id"]),
    ).select(F.col("source_concept_id"), F.col("isroot").alias("ancestor_concept_id"))

    cpt_hierarchy_level_0 = (
        cpt_hierarchy.groupby("source_concept_id")
        .max()
        .where(F.col("max(min_levels_of_separation)") == 0)
        .select(F.col("source_concept_id").alias("cpt_level_0_concept_id"))
    )

    cpt_hierarchy_level_0 = cpt_hierarchy.join(
        cpt_hierarchy_level_0,
        cpt_hierarchy["source_concept_id"] == cpt_hierarchy_level_0["cpt_level_0_concept_id"],
    ).select(
        F.col("source_concept_id"),
        F.col("parent_concept_id").alias("ancestor_concept_id"),
    )

    cpt_hierarchy_rollup_all = cpt_hierarchy_level_1.union(cpt_hierarchy_level_0).drop_duplicates()

    # ICD code rollup
    icd_list = ["ICD9CM", "ICD9Proc", "ICD10CM"]

    procedure_icd = procedure_code.where(F.col("vocabulary_id").isin(icd_list))

    procedure_icd = (
        procedure_icd.withColumn("parent_concept_code", parent_code_udf(F.col("child_concept_code")))
        .withColumnRenamed("procedure_source_concept_id", "source_concept_id")
        .withColumnRenamed("concept_name", "child_concept_name")
        .withColumnRenamed("vocabulary_id", "child_vocabulary_id")
        .withColumnRenamed("concept_code", "child_concept_code")
        .withColumnRenamed("concept_class_id", "child_concept_class_id")
    )

    procedure_icd_map = (
        procedure_icd.join(
            concept,
            (procedure_icd["parent_concept_code"] == concept["concept_code"])
            & (procedure_icd["child_vocabulary_id"] == concept["vocabulary_id"]),
            how="left",
        )
        .select("source_concept_id", F.col("concept_id").alias("ancestor_concept_id"))
        .distinct()
    )

    # ICD10PCS rollup
    procedure_10pcs = procedure_code.where(F.col("vocabulary_id") == "ICD10PCS")

    procedure_10pcs = (
        procedure_10pcs.withColumn("parent_concept_code", F.substring(F.col("child_concept_code"), 1, 3))
        .withColumnRenamed("procedure_source_concept_id", "source_concept_id")
        .withColumnRenamed("concept_name", "child_concept_name")
        .withColumnRenamed("vocabulary_id", "child_vocabulary_id")
        .withColumnRenamed("concept_code", "child_concept_code")
        .withColumnRenamed("concept_class_id", "child_concept_class_id")
    )

    procedure_10pcs_map = (
        procedure_10pcs.join(
            concept,
            (procedure_10pcs["parent_concept_code"] == concept["concept_code"])
            & (procedure_10pcs["child_vocabulary_id"] == concept["vocabulary_id"]),
            how="left",
        )
        .select("source_concept_id", F.col("concept_id").alias("ancestor_concept_id"))
        .distinct()
    )

    # HCPCS rollup --- keep the concept_id itself
    procedure_hcpcs = procedure_code.where(F.col("child_vocabulary_id") == "HCPCS")
    procedure_hcpcs_map = (
        procedure_hcpcs.withColumn("ancestor_concept_id", F.col("source_concept_id"))
        .select("source_concept_id", "ancestor_concept_id")
        .distinct()
    )

    procedure_hierarchy = (
        cpt_hierarchy_rollup_all.union(procedure_icd_map)
        .union(procedure_10pcs_map)
        .union(procedure_hcpcs_map)
        .distinct()
    )
    procedure_occurrence_fields = [
        F.col(f_n).alias(f_n.lower())
        for f_n in procedure_occurrence.schema.fieldNames()
        if f_n != "procedure_source_concept_id"
    ]
    procedure_occurrence_fields.append(
        F.coalesce(F.col("ancestor_concept_id"), F.col("procedure_source_concept_id")).alias(
            "procedure_source_concept_id"
        )
    )

    procedure_occurrence = (
        procedure_occurrence.join(
            procedure_hierarchy,
            procedure_occurrence["procedure_source_concept_id"] == procedure_hierarchy["source_concept_id"],
            how="left",
        )
        .select(procedure_occurrence_fields)
        .withColumn("procedure_concept_id", F.col("procedure_source_concept_id"))
    )
    return procedure_occurrence
