# seq2seq-model

This is an implementation of sequence to sequence model.

To train the model, use

```sh
python train_model.py --path path_train
```

where path_train is the path of the training data. It should has a .txt file and each line has the following structure

source_sentence  target_sentence

> Note: they are tab separated.

The parameters of the learned model are saved at "saved_model/seq2seq_weights" which can be loaded for testing. Please also see the notebook file for an exmaple.
