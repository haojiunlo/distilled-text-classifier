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
from utils import get_single_sentence_data

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("data_path", None, "Path to the data csv.")
flags.DEFINE_string("model_path", None, "Path to the model h5.")
flags.DEFINE_string(
    "vocab_file", None,
    "The vocab txt file specifying the vocab.")
flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the logits will be written")

# Other parameters
flags.DEFINE_integer(
    "max_seq_length", 32,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter than "
    "this will be padded.")
flags.DEFINE_integer("batch_size", 256, "Batch size for logits creation")

def main(argv):
    logging.set_verbosity(logging.INFO)

    # load data
    logging.info(f'reading {FLAGS.data_path}')
    df_unlabeled = pd.read_csv(FLAGS.data_path)
    logging.info(f'data size: {df_unlabeled.shape[0]}')

    logging.info('Tokenizing data... ')
    tokenizer = Tokenizer(FLAGS.vocab_file, do_lower_case=True)
    x_unlabeled, _ = get_single_sentence_data(df_unlabeled.sentence.tolist(), FLAGS.max_seq_length, tokenizer)

    # load trained model
    logging.info('Loading model... ')
    model = keras.models.load_model(FLAGS.model_path)
    
    # predict logit
    logging.info('Predicting logit... ')
    logit = model.predict([x_unlabeled, np.zeros_like(x_unlabeled)], batch_size=FLAGS.batch_size)

    # save logit
    logging.info(f'Saving logit at {FLAGS.output_dir}')
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    np.save(os.path.join(FLAGS.output_dir, 'logits.npy'), logit)
    
if __name__ == '__main__':
    flags.mark_flag_as_required("data_path")
    flags.mark_flag_as_required("model_path")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("output_dir")
    app.run(main)