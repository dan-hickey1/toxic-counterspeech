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
from sklearn.metrics import pairwise_distances

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

def total_three_way_distance(i, j, k, dist):
    return dist['12'][i, j] + dist['13'][i, k] + dist['23'][j, k]

# ---------------------------
# Matching methods
# ---------------------------

def optimal_matching(dist_matrix):
    """Hungarian algorithm for optimal matching."""
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return list(zip(row_ind, col_ind))

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


def pairwise_mahalanobis_distances(group1, group2):
    VI, group1, group2 = calculate_inverse_covariance(group1, group2)
    mahalanobis_metric = partial(spatial.distance.mahalanobis, VI=VI)

    D = pairwise_distances(group1, group2, metric=mahalanobis_metric)
    return D


# ---------------------------
# Triplet initialization
# ---------------------------

def initialize_triplets(group1, group2, group3, metric='euclidean'):
    if metric == 'mahalanobis':
        dist12 = pairwise_mahalanobis_distances(group1, group2)
        dist13 = pairwise_mahalanobis_distances(group1, group3)
        dist23 = pairwise_mahalanobis_distances(group2, group3)

    else:
        dist12 = pairwise_distances(group1, group2, metric)
        dist13 = pairwise_distances(group1, group3, metric)
        dist23 = pairwise_distances(group2, group3, metric)
        
    dist = {'12': dist12, '13': dist13, '23': dist23}

    matches_12 = optimal_matching(dist12)

    used_3 = set()
    triplets = []
    for i, j in matches_12:
        best_k = None
        best_d = float('inf')
        for k in range(len(group3)):
            if k in used_3:
                continue
            d = total_three_way_distance(i, j, k, dist)
            if caliper is not None and d > caliper:
                continue
            if d < best_d:
                best_d = d
                best_k = k
        if best_k is not None:
            triplets.append((i, j, best_k))
            used_3.add(best_k)

    return triplets, dist

# ---------------------------
# Iterative refinement
# ---------------------------

def iterate_matching(triplets, group1, group2, group3, dist):
    prev_triplets = triplets
    prev_distance = sum(total_three_way_distance(i, j, k, dist) for i, j, k in triplets)

    while True:
        improved = False

        # fix (j, k), reassign i
        pairs_23 = [(j, k) for _, j, k in prev_triplets]
        best_triplets_1 = []
        used_1 = set()
        for j, k in pairs_23:
            best_i = None
            best_d = float('inf')
            for i in range(len(group1)):
                if i in used_1:
                    continue
                d = total_three_way_distance(i, j, k, dist)
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i is not None:
                best_triplets_1.append((best_i, j, k))
                used_1.add(best_i)
        dist1 = sum(total_three_way_distance(i, j, k, dist) for i, j, k in best_triplets_1)

        #fix (i, k), reassign j
        pairs_13 = [(i, k) for i, _, k in prev_triplets]
        best_triplets_2 = []
        used_2 = set()
        for i, k in pairs_13:
            best_j = None
            best_d = float('inf')
            for j in range(len(group2)):
                if j in used_2:
                    continue
                d = total_three_way_distance(i, j, k, dist)
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j is not None:
                best_triplets_2.append((i, best_j, k))
                used_2.add(best_j)
        dist2 = sum(total_three_way_distance(i, j, k, dist) for i, j, k in best_triplets_2)

        #select best configuration
        if dist1 < prev_distance and dist1 <= dist2:
            prev_triplets = best_triplets_1
            prev_distance = dist1
            improved = True
        elif dist2 < prev_distance:
            prev_triplets = best_triplets_2
            prev_distance = dist2
            improved = True

        if not improved:
            break

    return prev_triplets, prev_distance


@click.command()
@click.option("--subreddit", required=True,
              help="Subreddit to perform matching for")
@click.option("--matching_type", default="direct",
              help="Method to use for matching. Options are 'direct' for matching directly on reduced features or 'logistic' for a logistic regression propensity score model")
@click.option("--n_components", default=5,
              help="If using direct matching, the number of principal components to reduce dimensions to. Doesn't matter for logistic matching")
@click.option("--counterspeech_threshold", default=1,
              help="Threshold to use for counterspeech/hate speech classification. Should be a value in [0.2, 0.4, 0.6, 0.8, 1], representing the fraction of LLaMa 3 prompts that predict counterspeech/hate speech for a given comment")
def main(subreddit, matching_type, n_components, counterspeech_threshold):    
    subreddit_df = pd.read_csv(f'../../data/subreddit_data/{subreddit}_full_data.csv', lineterminator='\n')
    
    sub_features = pd.read_csv(f'../../data/hate_subreddit_features/{subreddit}_features.csv') #describes features used to match subreddits, includes the cutoff time for the subreddit
    
    print("Subreddit: ", subreddit)
    
    return_time = sub_features.loc[len(sub_features) -1, 'return_time'] #cutoff time to drop users who posted too late to get a chance to follow up
    
    timestamp_df = pd.read_csv('../../data/subreddits_max_timestamps.csv')
    max_timestamp = timestamp_df.loc[timestamp_df['subreddit'] == subreddit, 'max_timestamp'].values[0]
    
    print("Dropping ineligible users...")
    submission_len = len(subreddit_df[subreddit_df['nest_level'] == 0])
    print(f"Dropping {submission_len} users out of {len(subreddit_df)} who posted submissions first")
    subreddit_df = subreddit_df[subreddit_df['nest_level'] != 0]
    
    print(f"Dropping {len(subreddit_df[subreddit_df['created_utc'] > (max_timestamp - return_time)])} out of {len(subreddit_df)} users due to late posts")
    subreddit_df = subreddit_df[subreddit_df['created_utc'] <= (max_timestamp - return_time)]
    
    print(f"Dropping {len(subreddit_df[subreddit_df['newcomer_counterspeech'] >= counterspeech_threshold])} out of {len(subreddit_df)} newcomers who used counterspeech")
    subreddit_df = subreddit_df[subreddit_df['newcomer_counterspeech'] < counterspeech_threshold]
    
    print(f"Dropping {len(subreddit_df[subreddit_df['newcomer_hatespeech'] < counterspeech_threshold])} out of {len(subreddit_df)} newcomers who didn't use hate speech")
    subreddit_df = subreddit_df[subreddit_df['newcomer_hatespeech'] >= counterspeech_threshold]
      
    print(f"Dropping {len(subreddit_df[(subreddit_df['reply_counterspeech'] > 0) & (subreddit_df['reply_counterspeech'] < counterspeech_threshold)])} out of {len(subreddit_df)} replies that are likely also counterspeech")
    subreddit_df = subreddit_df[(subreddit_df['reply_counterspeech'] == 0) | (subreddit_df['reply_counterspeech'] >= counterspeech_threshold)]
    
    
    nest_level_df = pd.read_csv(f'../../data/nest_levels/{subreddit}_missing_nest_levels.csv')
    subreddit_df = subreddit_df.merge(nest_level_df, on='id', how='left')
    subreddit_df['nest_level'] = subreddit_df['nest_level_x'].fillna(0) + subreddit_df['nest_level_y'].fillna(0)
    subreddit_df = subreddit_df.drop(columns=['nest_level_x', 'nest_level_y'])
    
    subreddit_df = subreddit_df.reset_index(drop=True)
    
    subreddit_df['sub_lifespan'] = (subreddit_df['created_utc']- subreddit_df['created_utc'].min()) / (subreddit_df['created_utc'].max() - subreddit_df['created_utc'].min())
    
    #define features to use for matching
    vars = ['nest_level', 'submission_total_comments', 'submission_unique_commenters', 'submission_direct_replies', 'direct_replies_unique', 'newcomer_sentiment',
            'sub_lifespan', 'score']
    
    #This strategy will ignore the users that have missing values for any of the features above when calculating the SMD for that feature.
    #In practice, missing data isn't really an issue for any of these features, however it was helpful for us when experimenting on other features.
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
    
    #generate embeddings for newcomer text
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(subreddit_df['text'].to_list(), show_progress_bar=True)
    
    subreddit_df['embedding'] = embeddings.tolist()
    
    #make treatment groups
    reply_counterspeech = subreddit_df[subreddit_df['reply_counterspeech'] >= COUNTERSPEECH_THRESHOLD]
    no_reply = subreddit_df[subreddit_df['reply_text'] == '<MISSING>']
    reply_other = subreddit_df[(subreddit_df['reply_text'] != '<MISSING>') & (subreddit_df['reply_counterspeech'] < COUNTERSPEECH_THRESHOLD)]
    
    smd_data = defaultdict(list)
    cols = ['covariate']
    #for each combo of treatment groups
    for combo in combinations([('counterspeech', reply_counterspeech), ('no_reply', no_reply), ('other', reply_other)], 2):
        cols.append(f'{combo[0][0]} vs. {combo[1][0]}')
        for col in vars:
            treatment = combo[0][1][combo[0][1]['id'].isin(ids_to_ignore[col]) == False][col]
            control = combo[1][1][combo[1][1]['id'].isin(ids_to_ignore[col]) == False][col]
    
            smd_data[col].append(calculate_smd(treatment,  control))
    
        if len(combo[0][1]) <= len(combo[1][1]):
            smaller_df = combo[0][1]
            larger_df = combo[1][1]
        else:
            smaller_df = combo[1][1]
            larger_df = combo[0][1]
        
        
        cosine_similarity = []
        for i in tqdm(range(10)):
            smaller_sample = larger_df.sample(len(smaller_df), replace=False, random_state=i)
            sample_similarities = []
            for embed_a, embed_b in zip(smaller_sample['embedding'], smaller_df['embedding']):
                sample_similarities.append(np.dot(embed_a, embed_b) / (norm(embed_a) * norm(embed_b)))
    
            
            cosine_similarity.append(np.mean(sample_similarities))
    
        smd_data['mean_cosine_sim'].append(np.mean(cosine_similarity))
        
    
    smd_df = pd.DataFrame(smd_data).T.reset_index()
    smd_df.columns = cols
    
    smd_df['avg'] = smd_df[cols[1:]].apply(np.mean, axis=1)
    smd_df['max'] = smd_df[cols[1:]].apply(max, axis=1)
    
    #save smd data
    smd_df.to_csv(f'../../data/covariate_balance/pre_matching/{subreddit}_pre_matching_hs_{counterspeech_threshold}_smd.csv', index=False)
    
    subreddit_df['class'] = 0
    subreddit_df.loc[subreddit_df['reply_counterspeech'] >= counterspeech_threshold, 'class'] = 2
    subreddit_df.loc[(subreddit_df['reply_counterspeech'] < counterspeech_threshold) & (subreddit_df['reply_text'] != '<MISSING>'), 'class'] = 1
    
    #define matching features/output
    Y = subreddit_df['class'].values
    X = subreddit_df[vars].values
    print(len(Y))
    
    scaler = StandardScaler()
    X_structured_scaled = scaler.fit_transform(X)
    
    X_combined = np.hstack([embeddings, X_structured_scaled])
    
    cs_copy = reply_counterspeech.reset_index(drop=True)
    nr_copy = no_reply.reset_index(drop=True)
    or_copy = reply_other.reset_index(drop=True)
    
    if MATCHING_TYPE == 'logistic':
        #train propensity score model
        pscores = cross_val_predict(
            LogisticRegression(solver='newton-cholesky'),
            X_combined, Y,
            method='predict_proba',
            cv=5
        )
        
        subreddit_df[[0, 1, 2]] = pscores
    
        cs_probs = subreddit_df[subreddit_df['class'] == 2][[0, 1, 2]].values
        nr_probs = subreddit_df[subreddit_df['class'] == 0][[0, 1, 2]].values
        or_probs = subreddit_df[subreddit_df['class'] == 1][[0, 1, 2]].values
        metric = 'euclidean'
    
    elif MATCHING_TYPE == 'direct':
        #match on reduced features
        X_combined = normalize(X_combined)
        
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_combined)
    
        explained_variance_ratio = pca.explained_variance_ratio_
        total_variance_preserved = explained_variance_ratio.sum()
    
        print(f"PCA with {n_components} components preserves {total_variance_preserved:.2%} of the total variance.")
    
        with open(f'../../data/pca_preserved_variance_{n_components}_threshold_{int(counterspeech_threshold * 100)}.csv', 'a+') as f:
            f.write(f'{subreddit},{total_variance_preserved:.2%}\n')
    
        cs_probs = X_reduced[subreddit_df.index[subreddit_df['class'] == 2]]
        nr_probs = X_reduced[subreddit_df.index[subreddit_df['class'] == 0]]
        or_probs = X_reduced[subreddit_df.index[subreddit_df['class'] == 1]]
        metric = 'mahalanobis'
    
    all_groups = [cs_probs, nr_probs, or_probs]
    all_dfs = [cs_copy, nr_copy, or_copy]
    names = ['counterspeech', 'no reply', 'other']
    
    current_matching = pd.DataFrame()
    current_balance = pd.DataFrame()
    
    #go through all possible pairs of treatment groups
    for idx1, idx2 in tqdm(combinations([0, 1, 2], 2)):
        idx3 = [i for i in [0, 1, 2] if i not in (idx1, idx2)][0]
        g1, g2, g3 = all_groups[idx1], all_groups[idx2], all_groups[idx3]
    
        triplets, dist = initialize_triplets(g1, g2, g3, method=matching_algorithm, metric=metric) #match triplets
        refined_triplets, final_dist = iterate_matching_nn(triplets, g1, g2, g3, dist)
    
        print(f"{names[idx1]} and {names[idx2]} first to match")
            
        g1_matched = all_dfs[idx1].loc[[t[0] for t in refined_triplets]]
        g2_matched = all_dfs[idx2].loc[[t[1] for t in refined_triplets]]
        g3_matched = all_dfs[idx3].loc[[t[2] for t in refined_triplets]]
        
        smd_matched = defaultdict(list)
        cols = ['covariate']
        for combo in combinations([(names[idx1], g1_matched), (names[idx2], g2_matched), (names[idx3], g3_matched)], 2):
            cols.append(f'{combo[0][0]} vs. {combo[1][0]}')
            for col in vars:
                treatment = combo[0][1][combo[0][1]['id'].isin(ids_to_ignore[col]) == False][col]
                control = combo[1][1][combo[1][1]['id'].isin(ids_to_ignore[col]) == False][col]
        
                smd_matched[col].append(calculate_smd(treatment,  control))
            
            smd_matched['mean_cosine_sim'].append(np.mean([np.dot(embed_a, embed_b) / (norm(embed_a) * norm(embed_b)) for embed_a, embed_b in zip(combo[0][1]['embedding'], combo[1][1]['embedding'])]))
    
        smd_matched_df = pd.DataFrame(smd_matched).T.reset_index()
        smd_matched_df.columns = cols
    
        smd_matched_df['avg'] = smd_matched_df[cols[1:]].apply(np.mean, axis=1)
        smd_matched_df['max'] = smd_matched_df[cols[1:]].apply(max, axis=1)
    
        #choose best balance
        if current_balance.empty:
            current_balance = smd_matched_df
            current_matching = pd.DataFrame([g1_matched['id'].to_list(), g2_matched['id'].to_list(), g3_matched['id'].to_list()]).T
            current_matching.columns = [names[idx1], names[idx2], names[idx3]]
    
        elif (current_balance.loc[current_balance['covariate'].isin(vars), 'avg'] < 0.1).sum() < (smd_matched_df.loc[smd_matched_df['covariate'].isin(vars), 'avg'] < 0.1).sum():
            current_balance = smd_matched_df
            current_matching = pd.DataFrame([g1_matched['id'].to_list(), g2_matched['id'].to_list(), g3_matched['id'].to_list()]).T
            current_matching.columns = [names[idx1], names[idx2], names[idx3]]
    
        elif (current_balance.loc[current_balance['covariate'].isin(vars), 'avg'] < 0.1).sum() == (smd_matched_df.loc[smd_matched_df['covariate'].isin(vars), 'avg'] < 0.1).sum():
            if current_balance.loc[current_balance['covariate'].isin(vars), 'avg'].mean() < smd_matched_df.loc[smd_matched_df['covariate'].isin(vars), 'avg'].mean():
                current_balance = smd_matched_df
                current_matching = pd.DataFrame([g1_matched['id'].to_list(), g2_matched['id'].to_list(), g3_matched['id'].to_list()]).T
                current_matching.columns = [names[idx1], names[idx2], names[idx3]]
    
    
    current_balance.to_csv(f'../../data/covariate_balance/post_matching/{subreddit}_post_matching_{matching_type}_threshold_{int(counterspeech_threshold * 100)}_nc_{n_components}.csv', index=False)
    current_matching.to_csv(f'../../data/matched_triplets/{subreddit}_matched_triplets_{matching_type}_threshold_{int(counterspeech_threshold * 100)}_nc_{n_components}.csv', index=False)

if __name__ == '__main__':
    main()