#!/bin/bash
ROOT=$HOME"/python2_nsm/"
DATA_DIR=$ROOT"data/wikitable_self_preprocessed/"
start=0
end=90
inc=8
for ((i=$start;i<=$end;i+=$inc))
do
	let j=$i+$inc	
	j=$(($j>90?90:$j))
	python ../random_explore.py \
	       --train_file_tmpl=$DATA_DIR"processed_input/wtq_preprocess/data_split_1/train_split_shard_90-{}.jsonl" \
	       --table_file=$DATA_DIR"processed_input/wtq_preprocess/tables.jsonl" \
	       --use_trigger_word_filter \
	       --trigger_word_file=$DATA_DIR"raw_input/trigger_word_all.json" \
	       --output_dir=$DATA_DIR"output" \
	       --experiment_name="random_explore" \
	       --n_explore_samples=50 \
	       --save_every_n=5 \
	       --n_epoch=200 \
	       --id_start=$i \
	       --id_end=$j \
	       --alsologtostderr
done
