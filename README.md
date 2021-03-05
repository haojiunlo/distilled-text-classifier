# Distilled Sentence Classifier
---
## Pretrained Model
https://github.com/CLUEbenchmark/CLUEPretrainedModels
https://github.com/ymcui/Chinese-BERT-wwm
## Usage
### 1 Finetune BERT using large model as teacher
```bash
export TRAIN_DIR=data/train_domain.csv
export TEST_DIR=data/test_domain.csv
export OUTPUT_DIR=output

export TEACHER_BERT_BASE_DIR=../Projects/Pretrained_model/RoBERTa_large_clue
export TEACHER_OUTPUT_DIR=RoBERTa_large_clue_teacher
export TEACHER_NUM_EPOCHS=3
export TEACHER_BATCH_SIZE=32
export TEACHER_LEARNING_RATE=0.00002

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
```

### 2 Create Logits for Distillation from the Finetuned teacher BERT

```bash
export UNLABELED_DATA_DIR=data/dailylog_201910_sample_100k.csv
export UNLABELED_DATA_LOGITS_DIR=logits/rbt_l_clue

python get_distillation_logits.py \
  --data_path=${UNLABELED_DATA_DIR} \
  --model_path={OUTPUT_DIR}/${TEACHER_OUTPUT_DIR}/bert_model.h5 \
  --vocab_file=${TEACHER_BERT_BASE_DIR}/vocab.txt \
  --output_dir=${UNLABELED_DATA_LOGITS_DIR}
```

### 3 Distill student model using the Finetuned BERT Logits and target datasets
```bash
export STUDENT_BERT_BASE_DIR=../Projects/Pretrained_model/RoBERTa-tiny3L768-clue
export STUDENT_OUTPUT_DIR=RoBERTa-tiny3L768_student
export STUDENT_NUM_EPOCHS=15
export STUDENT_BATCH_SIZE=256
export STUDENT_LEARNING_RATE=0.0002

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
```

or modified run_experiment.sh and run
```bash
sh run_experiment.sh
```