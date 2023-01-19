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
ID="1uI6HsBw1XWVvTEIs9goSpUVfeVJe-zEP"
wget -q --show-progress "https://drive.google.com/uc?export=download&id=$ID&export=download&confirm=yes" -O "$EDIA_DATA_DIR/mini_vocab_v6.zip"
ID="1T_pLFkUucP-NtPRCsO7RkOuhMqGi41pe"
wget -q --show-progress "https://drive.google.com/uc?export=download&id=$ID&export=download&confirm=yes" -O "$EDIA_DATA_DIR/full_vocab_v6.zip"
ID="1EN0pp1RKyRwi072QhVWJaDO8KlcFZo46"
wget -q --show-progress "https://drive.google.com/uc?export=download&id=$ID&export=download&confirm=yes" -O "$EDIA_DATA_DIR/100k_en_embedding.vec"
ID="1YwjyiDN0w54P55-y3SKogk7Zcd-WQ-eQ"
wget -q --show-progress "https://drive.google.com/uc?export=download&id=$ID&export=download&confirm=yes" -O "$EDIA_DATA_DIR/100k_es_embedding.vec"