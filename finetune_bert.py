import json
import joblib
import os
os.environ['TF_KERAS'] = '1'
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags, logging
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from optimizers import create_optimizer
from models import get_model
from utils import get_single_sentence_data

FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string("training_file", None, "Path to the csv training file.")
flags.DEFINE_string("testing_file", None, "Path to the the csv testing file.")
flags.DEFINE_string(
    "model_config_file", None,
    "The config json file specifying the model architecture.")
flags.DEFINE_string(
    "vocab_file", None,
    "The vocab txt file specifying the vocab.")
flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written. If "
    "`init_checkpoint' is not provided when exporting, the latest checkpoint "
    "from this directory will be exported.")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint, usually from a pre-trained BERT model. In the case of "
    "exporting, one can optionally provide path to a particular checkpoint to "
    "be exported here.")
flags.DEFINE_integer(
    "max_seq_length", 32,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter than "
    "this will be padded.")
flags.DEFINE_string("other_sentence", 'data/other_sentences.txt', 'other sentences')
flags.DEFINE_string("pooling_strategy", "cls_out", "Pooling strategy of BERT.")
flags.DEFINE_string("saved_model_name", "bert_model.h5", "Saved model name.")
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")
flags.DEFINE_integer("num_train_epochs", 3, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for.")  


def main(argv):
    logging.set_verbosity(logging.INFO)

    # load data
    logging.info(f'read {FLAGS.training_file} & {FLAGS.testing_file}')
    df_trn = pd.read_csv(FLAGS.training_file)
    df_ts = pd.read_csv(FLAGS.testing_file)

    # other sentence
    with open(FLAGS.other_sentence, 'r') as f:
            lines = f.readlines()
    add_data_size = min(df_trn.shape[0]*10, 3000)
    lines = lines[:add_data_size]
    lines = [line.rstrip('\n') for line in lines]
    logging.info(f'add {len(lines)} other sentences')
    df_trn = pd.concat([df_trn, pd.DataFrame({'sentence': lines, 'intent':'other'})], 
                       axis=0, sort=True)

    logging.info(f'Tokenizing data... ')
    tokenizer = Tokenizer(FLAGS.vocab_file, do_lower_case=True)
    x_trn, _ = get_single_sentence_data(df_trn.sentence.tolist(), FLAGS.max_seq_length, tokenizer)
    x_ts, _ = get_single_sentence_data(df_ts.sentence.tolist(), FLAGS.max_seq_length, tokenizer)

    le = LabelEncoder()
    y_trn = le.fit_transform(df_trn.intent)
    y_ts = le.transform(df_ts.intent)


    # init model
    logging.info(f'Initializing checkpoint from {FLAGS.init_checkpoint}... ')
    model = get_model(
        num_labels=len(le.classes_),
        config_path=FLAGS.model_config_file,
        checkpoint_path=FLAGS.init_checkpoint,
        pooling=FLAGS.pooling_strategy,
        )
    
    # optimizer parameters
    data_size = x_trn.shape[0]
    num_train_steps = data_size // FLAGS.train_batch_size * FLAGS.num_train_epochs
    warm_up_steps = int(num_train_steps * FLAGS.warmup_proportion)
    optimizer = create_optimizer(FLAGS.learning_rate, num_train_steps, warm_up_steps)
    model.compile(
        optimizer,
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    logging.info('Start training... ')
    model.fit([x_trn, np.zeros_like(x_trn)], y_trn, FLAGS.train_batch_size, FLAGS.num_train_epochs)
    _, accuracy = model.evaluate([x_ts, np.zeros_like(x_ts)], y_ts)
    logging.info(f'Testing accuracy: {accuracy}')
    
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir, exist_ok=True)

    with open(os.path.join(FLAGS.output_dir, 'output.txt'), 'w') as f:
        f.writelines(f'Testing accuracy: {accuracy}')
    
    logging.info(f'Saving checkpoint at {FLAGS.output_dir}')
    
    model.save(os.path.join(FLAGS.output_dir, FLAGS.saved_model_name), include_optimizer=False)
    joblib.dump(le, os.path.join(FLAGS.output_dir, 'LabelEncoder.pkl'))
    
if __name__ == '__main__':
    flags.mark_flag_as_required("training_file")
    flags.mark_flag_as_required("testing_file")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("output_dir")
    app.run(main)
