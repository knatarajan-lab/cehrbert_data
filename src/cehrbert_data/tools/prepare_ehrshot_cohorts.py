import os
import copy
from cehrbert_data.tools.extract_features import create_feature_extraction_args, main

if __name__ == "__main__":
    args = create_feature_extraction_args()
    cohort_dir = args.cohort_dir
    output_folder = args.output_folder
    for individual_cohort in os.listdir(cohort_dir):
        if individual_cohort.endswith('/'):
            individual_cohort = individual_cohort[:-1]
        cohort_name = os.path.basename(individual_cohort)
        args_copy = copy.copy(args)
        individual_cohort_dir = os.path.join(cohort_dir, individual_cohort, "labeled_patients.csv")
        if os.path.exists(individual_cohort_dir):
            if os.path.exists(os.path.join(individual_cohort_dir, cohort_name)):
                continue
            args_copy.cohort_dir = individual_cohort_dir
            args_copy.cohort_name = cohort_name
            main(
                args_copy
            )
