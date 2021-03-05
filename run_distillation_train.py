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
from utils import get_single_sentence_data
from models import get_model

FLAGS = flags.FLAGS

logging.set_verbosity(logging.INFO)

## Required parameters

flags.DEFINE_string("training_file", None, "Path to the csv training file.")
flags.DEFINE_string("testing_file", None, "Path to the the csv testing file.")
flags.DEFINE_string("distillation_data_path", None, "path where the data for distillation are.")
flags.DEFINE_string("distillation_logits_path", None, "path where the logits for distillation are.")
flags.DEFINE_string("label_encoder_path", None, "path where the label encoder of teacher model are.")
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
flags.DEFINE_string("saved_model_name", "distilbert_model.h5", "Saved model name.")
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_float("learning_rate", 2e-4, "The initial learning rate for Adam.")
flags.DEFINE_integer("num_train_epochs", 3, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for.")
flags.DEFINE_float(
    "softmax_temp", -1,
    "The temperature of the Softmax when making prediction on unlabeled examples." 
    "-1 means to use normal Softmax")
flags.DEFINE_float(
    "confidence_thresh", -1,
    "The threshold on predicted probability on unsupervised data. If set,"
    "UDA loss will only be calculated on unlabeled examples whose largest"
    "probability is larger than the threshold")
flags.DEFINE_float(
    "coeff", 1,
    help="Coefficient on the unlabeled loss.")


def main(argv):
    # define train step

    @tf.function
    def train_step(inputs, labels, logit_t, global_step):
        with tf.GradientTape() as tape: # 開始記錄正向傳播計
            logits = model([inputs, tf.zeros_like(inputs)], training=True)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            
            # labeled_loss
            labeled_log_probs = log_probs[:labeled_batch_size]
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            tgt_label_prob = one_hot_labels
            
            per_example_loss = -tf.reduce_sum(tgt_label_prob * labeled_log_probs, axis=-1)
            loss_mask = tf.ones_like(per_example_loss, dtype=per_example_loss.dtype)

            loss_mask = tf.stop_gradient(loss_mask)
            per_example_loss = per_example_loss * loss_mask
            labeled_loss = (tf.reduce_sum(per_example_loss) /
                        tf.maximum(tf.reduce_sum(loss_mask), 1))
                        
            # unlabeled_loss
            ori_log_probs = tf.nn.log_softmax(logit_t, axis=-1)
            aug_log_probs = log_probs[labeled_batch_size : ]
            unlabeled_loss_mask = 1
            
            if FLAGS.softmax_temp != -1:
                tgt_ori_log_probs = tf.nn.log_softmax(logit_t / FLAGS.softmax_temp, 
                                                    axis=-1)
                tgt_ori_log_probs = tf.stop_gradient(tgt_ori_log_probs)
            else:
                tgt_ori_log_probs = tf.stop_gradient(ori_log_probs)
                
            if FLAGS.confidence_thresh != -1:
                largest_prob = tf.reduce_max(tf.exp(ori_log_probs), axis=-1)
                unlabeled_loss_mask = tf.cast(tf.greater(largest_prob, FLAGS.confidence_thresh),
                                        tf.float32)
                unlabeled_loss_mask = tf.stop_gradient(unlabeled_loss_mask)
            per_example_ce_loss = -tf.reduce_sum(
                tf.exp(tgt_ori_log_probs) * aug_log_probs, axis=-1) * unlabeled_loss_mask

            unlabeled_loss = tf.reduce_mean(per_example_ce_loss)
                        
            # total_loss
            total_loss = labeled_loss + FLAGS.coeff * unlabeled_loss

        # 計算 gradient 反向傳播
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, labeled_loss, unlabeled_loss, logits[:labeled_batch_size]
    
    # load data
    logging.info(f'Reading {FLAGS.training_file} & {FLAGS.testing_file}')
    df_trn = pd.read_csv(FLAGS.training_file)
    df_ts = pd.read_csv(FLAGS.testing_file)
    
    logging.info(f'Reading {FLAGS.distillation_data_path}')
    df_unlabeled = pd.read_csv(FLAGS.distillation_data_path)
    logits_t = np.load(FLAGS.distillation_logits_path)

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
    
    # transform label
    le = joblib.load(FLAGS.label_encoder_path)
    y_trn = le.transform(df_trn.intent)
    y_ts = le.transform(df_ts.intent)
    
    # pack data into batches
    x_unlabeled, _ = get_single_sentence_data(df_unlabeled.sentence.tolist(), FLAGS.max_seq_length, tokenizer)

    data_unlabeled = tf.data.Dataset.from_tensor_slices((x_unlabeled, logits_t))

    batched_dataset_unlabeled = data_unlabeled.batch(
        FLAGS.train_batch_size, drop_remainder=True
        ).shuffle(x_unlabeled.shape[0])

    data_trn = tf.data.Dataset.from_tensor_slices((x_trn, y_trn))
    
    batched_dataset_labeled = data_trn.shuffle(x_trn.shape[0])
    batched_dataset_labeled = batched_dataset_labeled.repeat()
    batched_dataset_labeled = batched_dataset_labeled.batch(
        FLAGS.train_batch_size // 2, 
        drop_remainder=True
    ).take(
        df_unlabeled.shape[0] // FLAGS.train_batch_size
    )
    
    batch_datasets = tf.data.Dataset.zip(
        (batched_dataset_labeled, batched_dataset_unlabeled)
        )
                    
    steps_per_epochs = df_unlabeled.shape[0] // FLAGS.train_batch_size
    num_labels = len(le.classes_)
    labeled_batch_size = FLAGS.train_batch_size // 2

    # init student model
    logging.info(f'Initializing checkpoint from {FLAGS.init_checkpoint}')
    model = get_model(
        num_labels=num_labels,
        config_path=FLAGS.model_config_file,
        checkpoint_path=FLAGS.init_checkpoint,
        pooling=FLAGS.pooling_strategy,
        )

    # optimizer
    num_train_steps = steps_per_epochs * FLAGS.num_train_epochs
    warm_up_steps = int(num_train_steps * FLAGS.warmup_proportion)
    optimizer = create_optimizer(FLAGS.learning_rate, num_train_steps, warm_up_steps)
    logging.info(f'num train steps: {num_train_steps}')
    
    # Start training
    for epoch in range(FLAGS.num_train_epochs):
        epoch_total_loss_avg = tf.keras.metrics.Mean()
        epoch_labeled_loss_avg = tf.keras.metrics.Mean()
        epoch_unlabeled_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg = tf.keras.metrics.SparseCategoricalAccuracy()
        
        for batch in batch_datasets:
            global_step = optimizer.iterations
            inputs = tf.concat([batch[0][0], batch[1][0]], axis=0)
            labels = batch[0][1]
            logits_t = batch[1][1]
            total_loss, labeled_loss, unlabeled_loss, logits = train_step(inputs, labels, logits_t, global_step)
            epoch_total_loss_avg.update_state(total_loss)
            epoch_labeled_loss_avg.update_state(labeled_loss)
            epoch_unlabeled_loss_avg.update_state(unlabeled_loss)
            epoch_accuracy_avg.update_state(labels, logits)
            
            print(f'\rStep {global_step.numpy()}/{num_train_steps}, '
                f'Total loss: {total_loss:.4f}, '
                f'labeled loss: {labeled_loss:.4f}, '
                f'Unlabeled loss: {unlabeled_loss:.4f}', end='\r')

        print(f'Epoch {epoch}, Total loss={epoch_total_loss_avg.result():.4f}, '
            f'labeled loss={epoch_labeled_loss_avg.result():.4f}, '
            f'Unlabeled loss={epoch_unlabeled_loss_avg.result():.4f}, '
            f'Train acc={epoch_accuracy_avg.result():.4f}, ')
        
    # testing
    pred = model([x_ts, np.zeros_like(x_ts)], training=False)
    accuracy = (pred.numpy().argmax(-1) == y_ts).mean()
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
    flags.mark_flag_as_required("distillation_data_path")
    flags.mark_flag_as_required("distillation_logits_path")
    flags.mark_flag_as_required("label_encoder_path")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("output_dir")
    app.run(main)