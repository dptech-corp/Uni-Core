## A simple BERT example

1. download data `wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip & unzip wikitext-2-v1.zip`
2. run `python preprocess.py ./wikitext-2/wiki.train.tokens ./train.lmdb`
3. run `python preprocess.py ./wikitext-2/wiki.valid.tokens ./valid.lmdb`