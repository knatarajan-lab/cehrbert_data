# cehrbert_data

cehrbert_data is the ETL tool that generates the pretraining and finetuning datasets for CEHRbERT, which is a large language model developed for the structured EHR data, the work has been published
at https://proceedings.mlr.press/v158/pang21a.html.

## Patient Representation
For each patient, all medical codes were aggregated and constructed into a sequence chronologically.
In order to incorporate temporal information, we inserted an artificial time token (ATT) between two neighboring visits
based on their time interval.
The following logic was used for creating ATTs based on the following time intervals between visits, if less than 28
days, ATTs take on the form of $W_n$ where n represents the week number ranging from 0-3 (e.g. $W_1$); 2) if between 28
days and 365 days, ATTs are in the form of **$M_n$** where n represents the month number ranging from 1-11 e.g $M_{11}$;

3) beyond 365 days then a **LT** (Long Term) token is inserted. In addition, we added two more special tokens — **VS**
   and **VE** to represent the start and the end of a visit to explicitly define the visit segment, where all the
   concepts
   associated with the visit are subsumed by **VS** and **VE**.

!["patient_representation"](https://raw.githubusercontent.com/cumc-dbmi/cehr-bert/main/images/tokenization_att_generation.png)

## Pre-requisite
The project is built in python 3.10, and project dependency needs to be installed

Create a new Python virtual environment

```console
python3.10 -m venv .venv;
source .venv/bin/activate;
```

Build the project

```console
pip install -e .
```

Download [jtds-1.3.1.jar](jtds-1.3.1.jar) into the spark jars folder in the python environment
```console
cp jtds-1.3.1.jar .venv/lib/python3.10/site-packages/pyspark/jars/
```
## Instructions for Use

### 1. Download OMOP tables as parquet files

We created a spark app to download OMOP tables from SQL Server as parquet files. You need adjust the properties
in `db_properties.ini` to match with your database setup.

```console
PYTHONPATH=./: spark-submit tools/download_omop_tables.py -c db_properties.ini -tc person visit_occurrence condition_occurrence procedure_occurrence drug_exposure measurement observation_period concept concept_relationship concept_ancestor -o ~/Documents/omop_test/
```

We have prepared a synthea dataset with 1M patients for you to test, you could download it
at [omop_synthea.tar.gz](https://drive.google.com/file/d/1k7-cZACaDNw8A1JRI37mfMAhEErxKaQJ/view?usp=share_link)

```console
tar -xvf omop_synthea.tar ~/Document/omop_test/
```

### 2. Generate training data for CEHR-BERT
We order the patient events in chronological order and put all data points in a sequence. We insert artificial tokens
VS (visit start) and VE (visit end) to the start and the end of the visit. In addition, we insert artificial time
tokens (ATT) between visits to indicate the time interval between visits. This approach allows us to apply BERT to
structured EHR as-is.
The sequence can be seen conceptually as [VS] [V1] [VE] [ATT] [VS] [V2] [VE], where [V1] and [V2] represent a list of
concepts associated with those visits.

```console
PYTHONPATH=./: spark-submit spark_apps/generate_training_data.py -i ~/Documents/omop_test/ -o ~/Documents/omop_test/cehr-bert -tc condition_occurrence procedure_occurrence drug_exposure -d 1985-01-01 --is_new_patient_representation -iv
```
### 3. Generate hf readmission prediction task
If you don't have your own OMOP instance, we have provided a sample of patient sequence data generated using Synthea
at `sample/hf_readmissioon` in the repo

```console
PYTHONPATH=./:$PYTHONPATH spark-submit spark_apps/prediction_cohorts/hf_readmission.py -c hf_readmission -i ~/Documents/omop_test/ -o ~/Documents/omop_test/cehr-bert -dl 1985-01-01 -du 2020-12-31 -l 18 -u 100 -ow 360 -ps 0 -pw 30 --is_new_patient_representation
```

## Contact us
If you have any questions, feel free to contact us at CEHR-BERT@lists.cumc.columbia.edu

## Citation
Please acknowledge the following work in papers

Chao Pang, Xinzhuo Jiang, Krishna S. Kalluri, Matthew Spotnitz, RuiJun Chen, Adler
Perotte, and Karthik Natarajan. "Cehr-bert: Incorporating temporal information from
structured ehr data to improve prediction tasks." In Proceedings of Machine Learning for
Health, volume 158 of Proceedings of Machine Learning Research, pages 239–260. PMLR,
04 Dec 2021.
