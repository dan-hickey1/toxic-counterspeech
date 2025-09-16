# Assessing the Toxicity and Effects of Counterspeech on Reddit

To train the counterspeech detection models, run the command `python train_counterspeech_model.py --train_csv PATH_TO_TRAINING FILE`

The script `train_counterspeech_model.py` assumes your training/testing file is a CSV with two text columns: `parent` and `reply`. The `label` column describes the category of the reply: 1 for hate speech, 2 for counterspeech, and 0 for other speech. 
