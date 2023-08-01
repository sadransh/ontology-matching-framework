#!/bin/bash

. ../.venv/bin/activate

mkdir -p ../data/source_data/

cd ../data/source_data/

echo '\ndownloading source data UMLS ...\n'

wget -O UMLS.zip https://zenodo.org/record/6946466/files/UMLS.zip?download=1

unzip -uq UMLS.zip

echo '\ndownloading source data NCIT ...\n'

wget --timeout=3 --tries=15 --retry-connrefused -O Thesaurus21.FLAT.zip https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/2021/21.02d_Release/Thesaurus.FLAT.zip

unzip -uq Thesaurus21.FLAT.zip

mv ./Thesaurus.txt NCITThesaurus_v_21_02.txt

cd ../../scripts/

echo 'running data ontology preparation script ...'

python prepare_ont_data.py
python snomed_extension.py

echo 'running data pre-training data preparation script ...'

python create_hierarchical_ID_pretraining.py

echo 'running fine-tune data preparation script...'

python finetune_data.py
