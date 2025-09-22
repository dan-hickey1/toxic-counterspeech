import pandas as pd
import numpy as np
from scipy import spatial
from sentence_transformers import SentenceTransformer
import sys
from functools import partial
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from numpy.linalg import norm
import click

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

def calculate_smd(treatment, control, type='continuous'):
    if type == 'continuous':
        num = np.mean(treatment) - np.mean(control)
        denom = np.sqrt((np.var(treatment) + np.var(control)) / 2)
        return np.abs(num / denom)

    elif type == 'binary':
        p_treatment = sum(treatment) / len(treatment)
        p_control = sum(control) / len(control)

        num = p_treatment - p_control
        denom = np.sqrt((p_treatment * (1 - p_treatment) + p_control * (1 - p_control)) / 2)

        return np.abs(num / denom)

# ---------------------------
# Matching methods
# ---------------------------

def optimal_matching(dist_matrix):
    """Hungarian algorithm for optimal matching."""
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return list(zip(row_ind, col_ind))

def pairwise_mahalanobis_distances(group1, group2):
    VI, group1, group2 = calculate_inverse_covariance(group1, group2)
    mahalanobis_metric = partial(spatial.distance.mahalanobis, VI=VI)

    D = pairwise_distances(group1, group2, metric=mahalanobis_metric)
    return D

def calculate_inverse_covariance(group1, group2):
    pooled = np.vstack([group1, group2])
    n1 = group1.shape[0]
    
    count_of_distinct_values_per_feature = defaultdict(list)
    for i in range(pooled.shape[1]):
        size = len(np.unique(pooled[:, i]))
        count_of_distinct_values_per_feature[size].append(i)
    
    features_to_remove = count_of_distinct_values_per_feature[1].copy()
    
    # Get Features with perfect correlation
    keep_indices = [i for i in range(pooled.shape[1]) if i not in features_to_remove]
    pooled_filtered = pooled[:, keep_indices]
    
    corr_matrix = np.corrcoef(pooled_filtered, rowvar=False)
    
    # Identify columns with perfect correlation (excluding self-correlation)
    mask = (corr_matrix == 1).sum(axis=0) > 1
    perfect_corr_indices = [keep_indices[i] for i, m in enumerate(mask) if m]
    features_to_remove += perfect_corr_indices

    # Drop all identified features
    final_indices = [i for i in range(pooled.shape[1]) if i not in features_to_remove]
    pooled_final = pooled[:, final_indices]

    # Compute inverse covariance matrix
    V = np.cov(pooled_final, rowvar=False)
    VI = np.linalg.inv(V)
    
    group1_cleaned = pooled_final[:n1, :]
    group2_cleaned = pooled_final[n1:, :]
        
    return VI, group1_cleaned, group2_cleaned

@click.command()
@click.option("--subreddit", required=True,
              help="Subreddit to perform matching for")
@click.option("--matching_type", default="direct",
              help="Method to use for matching. Options are 'direct' for matching directly on reduced features or 'logistic' for a logistic regression propensity score model")
@click.option("--n_components", default=5,
              help="If using direct matching, the number of principal components to reduce dimensions to. Doesn't matter for logistic matching")
def main(subreddit, matching_type, n_components):    
    subreddit_df = pd.read_csv(f'../../data/subreddit_data/{subreddit}_full_data.csv', lineterminator='\n').drop_duplicates(subset='id')
        
    subreddit_df = subreddit_df[(subreddit_df['reply_text'].isnull()) | (subreddit_df['toxicity'].isnull() == False)]
    subreddit_df = subreddit_df[~subreddit_df['text'].isnull()]

    #find the matched hate subreddit to get the cutoff timestamp
    matching_df = pd.read_csv('../../data/matched_treatment_and_control_subreddits.csv')
    treatment_sub = matching_df.loc[matching_df['control'] == subreddit, 'treatment'].values[0]
    sub_features = pd.read_csv(f'../../data/hate_subreddit_features/{treatment_sub}_features.csv')
    
    print("Subreddit: ", subreddit)
    
    return_time = sub_features.loc[len(sub_features) -1, 'return_time'] #cutoff time to drop users who posted too late to get a chance to follow up
        
    timestamp_df = pd.read_csv('./subreddits_max_timestamps.csv')
    max_timestamp = timestamp_df.loc[timestamp_df['subreddit'] == treatment_sub, 'max_timestamp'].values[0]
    
    print("Dropping ineligible users...")
    submission_len = len(subreddit_df[(subreddit_df['link_id'].apply(lambda x: x[3:]) == subreddit_df['id'])])
    print(f"Dropping {submission_len} users out of {len(subreddit_df)} who posted submissions first")
    subreddit_df = subreddit_df[(subreddit_df['link_id'].apply(lambda x: x[3:]) != subreddit_df['id'])]
    print(f"Dropping {len(subreddit_df[subreddit_df['created_utc'] > (max_timestamp - return_time)])} out of {len(subreddit_df)} users due to late posts")
    subreddit_df = subreddit_df[subreddit_df['created_utc'] <= (max_timestamp - return_time)]
    
    nest_level_df = pd.read_csv(f'../../data/nest_levels/{subreddit}_missing_nest_levels.csv')
    subreddit_df = subreddit_df.merge(nest_level_df, on='id', how='left')
    
    #subsample to reduce runtime of matching
    if len(subreddit_df) > 150_000:
        subreddit_df = subreddit_df.sample(150_000, replace=False, random_state=42069)
    
    subreddit_df = subreddit_df.reset_index(drop=True)
    
    subreddit_df['sub_lifespan'] = (subreddit_df['created_utc']- subreddit_df['created_utc'].min()) / (subreddit_df['created_utc'].max() - subreddit_df['created_utc'].min())
    
    vars = ['nest_level', 'submission_total_comments', 'submission_unique_commenters', 'submission_direct_replies', 'direct_replies_unique', 'newcomer_sentiment', 'sub_lifespan', 'score']
    
    ids_to_ignore = {}
    cols_to_remove = []
    for col in vars:
        if subreddit_df[col].isnull().sum() == len(subreddit_df):
            cols_to_remove.append(col)
    
        else:
            ids_to_ignore[col] = subreddit_df[subreddit_df[col].isnull()]['id'].values
            subreddit_df[col] = subreddit_df[col].fillna(subreddit_df[col].median())
    
    for col in cols_to_remove:
        vars.remove(col)
    
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(subreddit_df['text'].to_list(), show_progress_bar=True)
    
    subreddit_df['embedding'] = embeddings.tolist()
    
    no_reply = subreddit_df[subreddit_df['reply_text'].isnull()]
    reply = subreddit_df[~subreddit_df['reply_text'].isnull()]
    
    smd_data = defaultdict(list)
    cols = ['covariate', 'smd']
    for col in vars:
        treatment = reply[reply['id'].isin(ids_to_ignore[col]) == False][col]
        control = no_reply[no_reply['id'].isin(ids_to_ignore[col]) == False][col]
    
        smd_data[col].append(calculate_smd(treatment,  control))
    
    cosine_similarity = []
    for i in tqdm(range(10)):
        smaller_sample = no_reply.sample(len(reply), replace=False, random_state=i)
        sample_similarities = []
        for embed_a, embed_b in zip(smaller_sample['embedding'], reply['embedding']):
            sample_similarities.append(np.dot(embed_a, embed_b) / (norm(embed_a) * norm(embed_b)))
    
        
        cosine_similarity.append(np.mean(sample_similarities))
    
    smd_data['mean_cosine_sim'].append(np.mean(cosine_similarity))
        
    
    smd_df = pd.DataFrame(smd_data).T.reset_index()
    smd_df.columns = cols
    
    smd_df.to_csv(f'./covariate_balance/pre_matching_control/{subreddit}_pre_matching_smd.csv', index=False)
    
    subreddit_df['class'] = 0
    subreddit_df.loc[(subreddit_df['reply_text'].isnull() == False), 'class'] = 1
    
    Y = subreddit_df['class'].values
    X = subreddit_df[vars].values
    
    scaler = StandardScaler()
    X_structured_scaled = scaler.fit_transform(X)
    
    X_combined = np.hstack([embeddings, X_structured_scaled])
    
    reply_copy = reply.reset_index(drop=True)
    nr_copy = no_reply.reset_index(drop=True)
    
    if matching_type == 'logistic':
        pscores = cross_val_predict(
            LogisticRegression(solver='newton-cholesky'),
            X_combined, Y,
            method='predict_proba',
            cv=5
        )
        
        subreddit_df['prob'] = pscores[:,1]
    
        reply_probs = subreddit_df[subreddit_df['class'] == 1]['prob'].values.reshape(-1, 1)
        nr_probs = subreddit_df[subreddit_df['class'] == 0]['prob'].values.reshape(-1, 1)
    
        metric = 'euclidean'
    
    elif matching_type == 'direct':
        X_combined = normalize(X_combined)
    
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_combined)
    
        explained_variance_ratio = pca.explained_variance_ratio_
        total_variance_preserved = explained_variance_ratio.sum()
    
        print(f"PCA with {n_components} components preserves {total_variance_preserved:.2%} of the total variance.")
    
        with open(f'../../data/pca_preserved_variance_{n_components}_control.csv', 'a+') as f:
            f.write(f'{subreddit},{total_variance_preserved:.2%}\n')
    
    
        reply_probs = X_reduced[subreddit_df.index[subreddit_df['class'] == 1]]
        nr_probs = X_reduced[subreddit_df.index[subreddit_df['class'] == 0]]
        metric = 'mahalanobis'
    
    
    all_dfs = [reply_copy, nr_copy]
    names = ['counterspeech', 'no reply', 'other']
    
    method = 'optimal'
    
    if metric == 'euclidean':
        dist = pairwise_distances(reply_probs, nr_probs, metric)
    
    elif metric == 'mahalanobis':
        dist = pairwise_mahalanobis_distances(reply_probs, nr_probs)
    
    
    matches = optimal_matching(dist) #match pairs
    
        
    reply_matched = reply_copy.loc[[m[0] for m in matches]]
    nr_matched = nr_copy.loc[[m[1] for m in matches]]
    
    smd_matched = defaultdict(list)
    cols = ['covariate', 'smd']
    for col in vars:
        treatment = reply_matched[reply_matched['id'].isin(ids_to_ignore[col]) == False][col]
        control = nr_matched[nr_matched['id'].isin(ids_to_ignore[col]) == False][col]
    
        smd_matched[col].append(calculate_smd(treatment, control))
    
    smd_matched['mean_cosine_sim'].append(np.mean([np.dot(embed_a, embed_b) / (norm(embed_a) * norm(embed_b)) for embed_a, embed_b in zip(reply_matched['embedding'], nr_matched['embedding'])]))
    
    smd_matched_df = pd.DataFrame(smd_matched).T.reset_index()
    smd_matched_df.columns = cols
    
    matched_pairs = pd.DataFrame([nr_matched['id'].to_list(), reply_matched['id'].to_list()]).T
    matched_pairs.columns = ['control', 'treatment']
    
    
    smd_matched_df.to_csv(f'./covariate_balance/post_matching_control/{subreddit}_post_matching_{matching_type}_nc_{n_components}.csv', index=False)
    matched_pairs.to_csv(f'./matched_control/{subreddit}_matched_pairs_{matching_type}_nc_{n_components}.csv', index=False)

if __name__ == '__main__':
    main()