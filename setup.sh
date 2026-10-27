# !/bin/bash
# HOW TO USE:
#   $ ./setup.sh

WORKDIR=$(pwd)
EDIA_DATA_DIR="$WORKDIR/data"

if [ ! -d $EDIA_DATA_DIR ]; 
then 
    echo "* Creating 'data/' directory ..."
    mkdir $EDIA_DATA_DIR
fi

echo "* Downloading files inside 'data/' directory ..."
gdown --id "1uI6HsBw1XWVvTEIs9goSpUVfeVJe-zEP" -O "$EDIA_DATA_DIR/mini_vocab_v6.zip"
gdown --id "1T_pLFkUucP-NtPRCsO7RkOuhMqGi41pe" -O "$EDIA_DATA_DIR/full_vocab_v6.zip"
gdown --id "1EN0pp1RKyRwi072QhVWJaDO8KlcFZo46" -O "$EDIA_DATA_DIR/100k_en_embedding.vec"
gdown --id "1YwjyiDN0w54P55-y3SKogk7Zcd-WQ-eQ" -O "$EDIA_DATA_DIR/100k_es_embedding.vec"

# wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
# tar -xvzf git-lfs-linux-amd64-v3.5.1.tar.gz
# ./git-lfs-3.5.1/git-lfs clone https://huggingface.co/datasets/vialibre/splittedspanish3bwc