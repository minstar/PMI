# Regularizing Models via Pointwise Mutual Information for Named Entity Recognition
This repository suggests the code for "Regularizing Models via Pointwise Mutual Information for Named Entity Recognition" (https://arxiv.org/abs/2104.07249)

## Environments
- cuda 10.2 was used
- conda environment was suggested with environment.yaml (`conda env create --file environment.yaml`)

## Additional Requirements
- seqeval: Used for NER evaluation (`pip install seqeval`)

## Traininig
```bash
export SAVE_DIR=./output/
export DATA_DIR=./datasets/NER/

export MAX_LENGTH=128
export BATCH_SIZE=32
export NUM_EPOCHS=35
export SAVE_STEPS=-1
export ENTITY=NCBI-disease
export WARMUP=5000
export run_name=check
export SEED=1
export SM=100
export LV=0.03

python run_ner.py \
    --data_dir ${DATA_DIR}/NER/${ENTITY} \
    --labels ${DATA_DIR}/NER/${ENTITY}/labels.txt \
    --model_name_or_path dmis-lab/biobert-v1.1 \
    --output_dir ${SAVE_DIR}/biobert-${ENTITY}-${NUM_EPOCHS}-${LR}-SEED${SEED} \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --warmup_steps ${WARMUP} \
    --learning_rate ${LR} \
    --smooth ${SM} \
    --lambda_val ${LV} \
    --is_pmi \
    --is_subword \
    --do_train \
    --do_eval \
    --do_predict \
    --wandb_name ${run_name} \
    --overwrite_output_dir \
```

## Citation
```bash
  @article{jeong2021regularizing,
  title={Regularizing Models via Pointwise Mutual Information for Named Entity Recognition},
  author={Jeong, Minbyul and Kang, Jaewoo},
  journal={arXiv preprint arXiv:2104.07249},
  year={2021}
  }
```

## Contact
For help or issues using our code, please create an issue or send an email to `minbyuljeong@korea.ac.kr`
