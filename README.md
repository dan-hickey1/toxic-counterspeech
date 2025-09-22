# Assessing the Toxicity and Effects of Counterspeech on Reddit

This repository contains code and data for the submission "Assessing the Toxicity and Effects of Counterspeech on Reddit."

### Data Preprocessing
Due to the size of the Reddit data used for this study, we do not share all of it, though it can be downloaded from the Pushshift archives. However, the `preprocessing.py` script contains code that takes all data from a given subreddit and organizes it into a CSV file with all relevant information for matching and regression analyses. Assumes you have a directory with subdirectories of JSONL files containing data for each subreddit.

The code must be run with the following (required) arguments `python preprocessing.py --subreddit SUBREDDIT --path_to_reddit_files PATH_TO_DIRECTORY`

### Model Training

To train the RoBERTa counterspeech detection models, run the command `python train_counterspeech_model.py --train_csv PATH_TO_TRAINING FILE`

The script `train_counterspeech_model.py` assumes your training/testing file is a CSV with two text columns: `parent` and `reply`. The `label` column describes the category of the reply: 1 for hate speech, 2 for counterspeech, and 0 for other speech.

To fine-tune LLaMa 3, we used the [Axolotl](https://docs.axolotl.ai/docs/getting-started.html) package. Our YAML file with training parameters is found in `fine_tune_llama3.yaml`. After installing Axolotl, you can fine-tune the model with `axolotl train fine_tune_llama3.yaml`.

`use_llama.py` makes inferences with LLaMa 3. This requires a path to an input file to use for inference, an output file path to save results, and you can optionally set the temperature of the model (default 1.2).

### User Matching
There are two scripts for matching users based on their features: `match_non_hate_users.py` matches pairs of users from non-hate subreddits, while `match_user_triplets.py` matches triplets of users from hate subreddits based on the type of reply they receive. For both scripts, `--subreddit` is a required argument. Optionally, the type of matching (`--matching_type`, logistic or direct), or the number of principal components to reduce matching features to (`--n_components`, integer) can be set for either script. The `--counterspeech_threshold` argument, which specifies which hate speech/counterspeech threshold to use for classifying users, can also be set for the triplet matching script.
