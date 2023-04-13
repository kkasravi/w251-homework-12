
tar:
	curl https://hw12.s3.us-west-1.amazonaws.com/libri.tar.gz --output libri.tar.gz

tokenizer:
	python process_asr_text_tokenizer.py --manifest=../librispeech/train_clean_100.json --data_root=../librispeech/LibriSpeech --tokenizer=spe --spe_type=unigram --vocab_size=1024		

train:
	python speech_to_text_rnnt_bpe.py \
    		--config-path=. --config-name=conformer_transducer_bpe \
	      	model.train_ds.manifest_filepath=../librispeech/train_clean_100.json \
    		model.validation_ds.manifest_filepath=../librispeech/test_clean.json \
    		model.tokenizer.dir=../librispeech/LibriSpeech/tokenizer_spe_unigram_v1024 \
    		model.tokenizer.type=bpe \
    		trainer.devices=-1 \
    		trainer.accelerator="gpu" \
    		trainer.strategy="ddp" \
    		trainer.max_epochs=100 \
    		model.optim.name="adamw" \
    		model.optim.lr=0.001 \
    		model.optim.betas=[0.9,0.999] \
    		model.optim.weight_decay=0.0001 \
    		model.optim.sched.warmup_steps=2000 \
    		exp_manager.create_wandb_logger=True \
    		exp_manager.wandb_logger_kwargs.name=speech_to_text_asr \
    		exp_manager.wandb_logger_kwargs.project=homework-12
