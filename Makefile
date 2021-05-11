SAVE_DIR=./output/
DATA_DIR=./datasets/

MAX_LENGTH=128
BERT_MODEL=bert-base-cased
BATCH_SIZE=32
NUM_EPOCHS=35
SAVE_STEPS=-1
SEED=1
LR=5e-5
WARMUP=0
ENTITY=NCBI-disease
cuda=1
SM=100
CA=1
WA=1
LV=0.03
run_name=check

run_bioner_eval:
	CUDA_VISIBLE_DEVICES=$(cuda) python run_eval.py \
		--data_dir $(DATA_DIR)/NER/$(ENTITY) \
		--labels $(DATA_DIR)/NER/$(ENTITY)/labels.txt \
		--model_name_or_path $(SAVE_DIR)/biobert-$(ENTITY)-$(NUM_EPOCHS)-$(LR)-SEED$(SEED) \
		--output_dir $(SAVE_DIR)/biobert-$(ENTITY)-$(NUM_EPOCHS)-$(LR)-$(eval_data_name)-SEED$(SEED) \
		--max_seq_length  $(MAX_LENGTH) \
		--num_train_epochs $(NUM_EPOCHS) \
		--per_device_eval_batch_size $(BATCH_SIZE) \
		--eval_data_name $(eval_data_name) \
		--save_steps $(SAVE_STEPS) \
		--seed $(SEED) \
		--warmup_steps $(WARMUP) \
		--learning_rate $(LR) \
		--do_eval \
		--do_predict \
		--wandb_name $(run_name) \
		--overwrite_output_dir \

run_ner_eval:
	CUDA_VISIBLE_DEVICES=$(cuda) python run_eval.py \
		--data_dir $(DATA_DIR)/NER/$(ENTITY) \
		--labels $(DATA_DIR)/NER/$(ENTITY)/labels.txt \
		--model_name_or_path $(SAVE_DIR)/bert-$(ENTITY)-$(NUM_EPOCHS)-$(LR)-SEED$(SEED) \
		--output_dir $(SAVE_DIR)/bert-$(ENTITY)-$(NUM_EPOCHS)-$(LR)-$(eval_data_name)-SEED$(SEED) \
		--max_seq_length  $(MAX_LENGTH) \
		--num_train_epochs $(NUM_EPOCHS) \
		--per_device_eval_batch_size $(BATCH_SIZE) \
		--eval_data_name $(eval_data_name) \
		--save_steps $(SAVE_STEPS) \
		--seed $(SEED) \
		--warmup_steps $(WARMUP) \
		--learning_rate $(LR) \
		--do_eval \
		--do_predict \
		--wandb_name $(run_name) \
		--overwrite_output_dir \

run_biobert-base-cased-bioner:
	CUDA_VISIBLE_DEVICES=$(cuda) python run_ner.py \
		--data_dir $(DATA_DIR)/NER/$(ENTITY) \
		--labels $(DATA_DIR)/NER/$(ENTITY)/labels.txt \
		--model_name_or_path dmis-lab/biobert-v1.1 \
		--output_dir $(SAVE_DIR)/biobert-$(ENTITY)-$(NUM_EPOCHS)-$(LR)-SEED$(SEED) \
		--max_seq_length  $(MAX_LENGTH) \
		--num_train_epochs $(NUM_EPOCHS) \
		--per_device_train_batch_size $(BATCH_SIZE) \
		--per_device_eval_batch_size $(BATCH_SIZE) \
		--save_steps $(SAVE_STEPS) \
		--seed $(SEED) \
		--warmup_steps $(WARMUP) \
		--learning_rate $(LR) \
		--smooth $(SM) \
		--class_alpha $(CA) \
		--word_alpha $(WA) \
		--lambda_val $(LV) \
		--is_pmi \
		--is_subword \
		--do_train \
		--do_eval \
		--do_predict \
		--wandb_name $(run_name) \
		--overwrite_output_dir \

run_bert-base-cased-ner:
	CUDA_VISIBLE_DEVICES=$(cuda) python run_ner.py \
		--data_dir $(DATA_DIR)/NER/$(ENTITY) \
		--labels $(DATA_DIR)/NER/$(ENTITY)/labels.txt \
		--model_name_or_path bert-base-cased \
		--output_dir $(SAVE_DIR)/bert-$(ENTITY)-$(NUM_EPOCHS)-$(LR)-SEED$(SEED) \
		--max_seq_length  $(MAX_LENGTH) \
		--num_train_epochs $(NUM_EPOCHS) \
		--per_device_train_batch_size $(BATCH_SIZE) \
		--per_device_eval_batch_size $(BATCH_SIZE) \
		--save_steps $(SAVE_STEPS) \
		--seed $(SEED) \
		--warmup_steps $(WARMUP) \
		--learning_rate $(LR) \
		--smooth $(SM) \
		--class_alpha $(CA) \
		--word_alpha $(WA) \
		--lambda_val $(LV) \
		--is_pmi \
		--is_subword \
		--do_train \
		--do_eval \
		--do_predict \
		--wandb_name $(run_name) \
		--overwrite_output_dir \

