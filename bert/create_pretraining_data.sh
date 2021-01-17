! python create_pretraining_data.py \
  --input_file=/home/leonard/leonard/nlp/ReINTEL/data/tokenized_news_data.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=/home/leonard/leonard/nlp/ReINTEL/pretrained_phobert-base/vocab.txt \
  --do_lower_case=False \
  --max_seq_length=184 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
