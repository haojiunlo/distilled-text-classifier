TRAIN_DIR=data/train_domain.csv
TEST_DIR=data/test_domain.csv
OUTPUT_DIR=output

TEACHER_BERT_BASE_DIR=../Projects/Pretrained_model/RoBERTa_large_clue
TEACHER_OUTPUT_DIR=RoBERTa_large_clue_teacher
TEACHER_NUM_EPOCHS=3
TEACHER_BATCH_SIZE=32
TEACHER_LEARNING_RATE=0.00002

UNLABELED_DATA_DIR=data/dailylog_201910_sample_100k.csv
UNLABELED_DATA_LOGITS_DIR=logits/rbt_l_clue

STUDENT_BERT_BASE_DIR=../Projects/Pretrained_model/RoBERTa-tiny3L768-clue
STUDENT_OUTPUT_DIR=RoBERTa-tiny3L768_student
STUDENT_NUM_EPOCHS=15
STUDENT_BATCH_SIZE=256
STUDENT_LEARNING_RATE=0.0002

# finetune teacher model
python finetune_bert.py \
  --training_file=${TRAIN_DIR} \
  --testing_file=${TEST_DIR} \
  --model_config_file=${TEACHER_BERT_BASE_DIR}/bert_config.json \
  --vocab_file=${TEACHER_BERT_BASE_DIR}/vocab.txt \
  --output_dir=${OUTPUT_DIR}/${TEACHER_OUTPUT_DIR} \
  --init_checkpoint=${TEACHER_BERT_BASE_DIR}/bert_model.ckpt \
  --train_batch_size=${TEACHER_BATCH_SIZE} \
  --num_train_epochs=${TEACHER_NUM_EPOCHS} \
  --learning_rate=${TEACHER_LEARNING_RATE}

# get distillation logits
python get_distillation_logits.py \
  --data_path=${UNLABELED_DATA_DIR} \
  --model_path={OUTPUT_DIR}/${TEACHER_OUTPUT_DIR}/bert_model.h5 \
  --vocab_file=${TEACHER_BERT_BASE_DIR}/vocab.txt \
  --output_dir=${UNLABELED_DATA_LOGITS_DIR}

# distillation
python run_distillation_train.py \
  --training_file={TRAIN_DIR} \
  --testing_file={TEST_DIR} \
  --distillation_data_path=${UNLABELED_DATA_DIR} \
  --distillation_logits_path=${UNLABELED_DATA_LOGITS_DIR}/logits.npy \
  --label_encoder_path=${OUTPUT_DIR}/${TEACHER_OUTPUT_DIR}/LabelEncoder.pkl  \
  --model_config_file=${STUDENT_BERT_BASE_DIR}/bert_config.json \
  --init_checkpoint=${STUDENT_BERT_BASE_DIR}/bert_model.ckpt \
  --vocab_file=${STUDENT_BERT_BASE_DIR}/vocab.txt \
  --output_dir=${OUTPUT_DIR}/${STUDENT_OUTPUT_DIR} \
  --train_batch_size=256 \
  --num_train_epochs=15 \
  --learning_rate=${STUDENT_LEARNING_RATE}
