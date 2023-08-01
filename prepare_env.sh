
#!/bin/bash


mkdir -p ./.venv

python -m venv ./.venv

. ./.venv/bin/activate

pip install -r ./requirements.txt

git clone  https://github.com/KRR-Oxford/DeepOnto

cd ./DeepOnto

git reset --hard a48227c 


pip install .

