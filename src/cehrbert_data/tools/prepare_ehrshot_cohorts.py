import os
import copy
from cehrbert_data.tools.extract_features import create_feature_extraction_args, main

if __name__ == "__main__":
    args = create_feature_extraction_args()
    cohort_dir = args.cohort_dir
    output_folder = args.output_folder
    for individual_cohort in os.listdir(cohort_dir):
        args_copy = copy.copy(args)
        individual_cohort_dir = os.path.join(cohort_dir, individual_cohort)
        individual_output_folder = os.path.join(output_folder, individual_cohort)
        args_copy.cohort_dir = individual_cohort_dir
        args_copy.output_folder = individual_output_folder
        main(
            args_copy
        )
