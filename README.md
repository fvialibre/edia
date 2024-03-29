# EDIA: Stereotypes and Discrimination in Artificial Intelligence
[[Paper]](https://arxiv.org/abs/2207.06591) [[HuggingFace🤗 Demo]](https://huggingface.co/spaces/vialibre/edia) 

Language models and word representations obtained with machine learning contain discriminatory stereotypes. Here we present the EDIA project (Stereotypes and Discrimination in Artificial Intelligence). This project aimed to design and evaluate a methodology that allows social scientists and domain experts in Latin America to explore biases and discriminatory stereotypes present in word embeddings (WE) and language models (LM). It also allowed them to define the type of bias to explore and do an intersectional analysis using two binary dimensions (for example, female-male intersected with fat-skinny).

EDIA contains several functions that serve to detect and inspect biases in natural language processing systems based on language models or word embeddings. We have models in Spanish and English to work with and explore biases in different languages ​​at the user's request. Each of the following spaces contains different functions that bring us closer to a particular aspect of the problem of bias and they allow us to understand different but complementary parts of it.

You can test and explore this functions with our live demo hosted on HuggingFace🤗 by clicking [here](https://huggingface.co/spaces/vialibre/edia).

## Installation

Setup the code in a virtualenv

```sh
# Clone repo
$ git clone https://github.com/fvialibre/edia.git && cd edia
# Create and activate virtualenv
$ python3 -m venv venv  && source venv/bin/activate
# Install requirements
$ python3 -m pip install -r requirements.txt
```
## Setup data

In order to start using this tool, you need to create the requiered structure for it to retrieve the data. To do this, we provide you a script for doing it automatically, but also explainations on how to do it manually for more personal customization.

### Automatic setup
In the cloned repository you have the `setup.sh` script that you can run in Linux OS:

```sh
$ ./setup.sh
```

This will create a `data/` folder inside the repository and download from *Google Drive* two 100k embeddings files (for English and Spanish), and two vocabulary files (`Min` and `Full`, see [Manual setup](#Manual-setup)).

### Manual setup
To setup the structure manually just create a `data/` folder inside the `edia` repository just cloned:

```sh
$ mkdir data
```

And then download inside this newly created folder the files you will need:

* [Min vocabulary:](https://drive.google.com/file/d/1uI6HsBw1XWVvTEIs9goSpUVfeVJe-zEP/view?usp=sharing) Composed of only 56 words, for tests purpose only.
* [Full vocabulary:](https://drive.google.com/file/d/1T_pLFkUucP-NtPRCsO7RkOuhMqGi41pe/view?usp=sharing) Composed of 1.2M words.
* [Spanish word embeddings: ](https://drive.google.com/file/d/1YwjyiDN0w54P55-y3SKogk7Zcd-WQ-eQ/view?usp=sharing) 100K spanish word embeddings of 300 dimensions (from [Jorge Pérez's website](http://dcc.uchile.cl/~jperez))
* [English word embeddings: ](https://drive.google.com/file/d/1EN0pp1RKyRwi072QhVWJaDO8KlcFZo46/view?usp=sharing) 100K english word embeddings of 300 dimensions (from [Eyal Gruss's github](https://github.com/eyaler/word2vec-slim))

> **Note**: You will need one of the two vocabulary files (`Min` or `Full`) if you don't want to be bothered to create the complex structure needed. The embeddings file, on the other side, can be one of your own, we just give this two as functional options.

## Usage
```sh
# If you are not already in the venv
$ source venv/bin/activate
$ python3 app.py
```

## Tool Configuration

The file `tool.cfg` contains configuration parameters for the tool:

| **Name** | **Options** | **Description** |
|---|---|---|
| language | `es`, `en` | Changes the interface language |
| embeddings_path | `data/100k_es_embedding.vec`, `data/100k_en_embedding.vec` | Path to word embeddings to use. You can use your own embedding file as long as it is in `.vec` format. If it's a `.bin` extended file, only gensims c binary format are valid. The options correspond to pretrained english and spanish embeddings. |
| nn_method | `sklearn`, `ann` | Method used to fetch nearest neighbors. Sklearn uses [sklearn nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html) exact calculation so your embedding must fit in your computer's memory, it's a slower approach for large embeddings. [Ann](https://pypi.org/project/annoy/1.0.3/) is a approximate nearest neighbors search suitable for large embeddings that don't fit in memory |
| max_neighbors | (int) `20` | Select amount of neighbors to fit sklearn nearest neighbors method. |
| context_dataset | `vialibre/splittedspanish3bwc` | Path to splitted 3bwc dataset optimised for word context search. |
| vocabulary_subset | `mini`, `full` | Vocabulary necessary for context search tool |
| available_wordcloud | `True`, `False` | Show wordcloud in "Data" interface |
| language_model | `bert-base-uncased`, `dccuchile/bert-base-spanish-wwm-uncased` | `bert-base-uncased` is an english language model, `bert-base-spanish-wwm-uncased` is an spanish model. You can inspect any bert-base language model uploaded to the [HuggingfaceHub](https://huggingface.co/models). | 
| available_logs | `True`, `False` | Activate logging of user's input. Saved logs will be stores in `logs/` folder. |                                               

## Resources
### Videotutorials and user's manual
* Word explorer: [[video]]() [manual: [es](https://shorturl.at/cgwxJ) | en]
* Word bias explorer: [[video]]() [manual: [es](https://shorturl.at/htuEI) | en]
* Phrase bias explorer: [[video]]() [manual: [es](https://shorturl.at/fkBL3) | en]
* Data explorer: [[video]]() [manual: [es](https://shorturl.at/CIVY6) | en]
* Crows-Pairs: [[video]]() [manual: [es](https://shorturl.at/gJLTU) | en]
### Interactive nooteboks
* How to use (*road map*): [[es](notebook/EDIA_Road_Map.ipynb) | en]
* Classes and methods docs: [[es](notebook/EDIA_Docs.ipynb) | en]

## Citation Information
```c
@misc{https://doi.org/10.48550/arxiv.2207.06591,
    doi = {10.48550/ARXIV.2207.06591},
    url = {https://arxiv.org/abs/2207.06591},
    author = {Alemany, Laura Alonso and Benotti, Luciana and González, Lucía and Maina, Hernán and Busaniche, Beatriz and Halvorsen, Alexia and Bordone, Matías and Sánchez, Jorge},
    keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), 
    FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {A tool to overcome technical barriers for bias assessment in human language technologies},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```

## License Information 
This project is under a [MIT license](LICENSE).

