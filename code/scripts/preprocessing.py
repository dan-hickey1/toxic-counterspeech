import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from w3lib.html import replace_entities
import re
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googleapiclient import discovery
import click
tqdm.pandas()

#initialize client for toxicity calculation
client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey="YOUR_API_KEY",
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

def calculate_toxicity(row):
    if 'reply_text' in row:
        unaltered_comment = str(row['reply_text'])
        time.sleep(1)
        analyze_request = {
            'comment': { 'text': unaltered_comment },
            'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY': {}, 'IDENTITY_ATTACK': {}, 'ATTACK_ON_COMMENTER': {}}
        }
        try:
            response = client.comments().analyze(body=analyze_request).execute()
            row['severe_toxicity'] = (response['attributeScores']['SEVERE_TOXICITY']['spanScores'][0]['score']['value'])
            row['toxicity'] = (response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value'])
            row['identity_attack'] = (response['attributeScores']['IDENTITY_ATTACK']['spanScores'][0]['score']['value'])
            row['attack_on_commenter'] = (response['attributeScores']['ATTACK_ON_COMMENTER']['spanScores'][0]['score']['value'])

        except:
            row['severe_toxicity'] = np.nan
            row['toxicity'] = np.nan
            row['identity_attack'] = np.nan
            row['attack_on_commenter'] = np.nan

    else:
        row['severe_toxicity'] = np.nan
        row['toxicity'] = np.nan
        row['identity_attack'] = np.nan
        row['attack_on_commenter'] = np.nan

    return row

def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    
    return sentiment_dict['compound']

def clean_text(text):
    text = str(text)
    hyperlink_regex = '\[(.*?)\]\(.*?\)'
    url_regex = r'http\S+'

    text = replace_entities(text)
    text = re.sub(hyperlink_regex, r'\1', text)
    text = re.sub(url_regex, '', text)
    text = re.sub('\(\s+\)', '', text)
    text = text.replace('[deleted]', '').replace('[removed]', '').strip()
    return text.strip()

def make_text_col(row):
    if 'body' not in row or str(row['body']) == 'nan':
        if str(row['selftext']) == 'nan' or row['selftext'] == '[removed]' or row['selftext'] == '[deleted]':
            return clean_text(row['title'])

        else:
            return clean_text(row['title'] + ' ' + row['selftext'])

    else:
        return clean_text(row['body'])


@click.command()
@click.option("--subreddit", required=True,
              help="Subreddit to perform matching for")
@click.option("--path_to_reddit_files", type=click.Path(exists=True, dir_okay=True), required=True,
              help="Path to directory containing JSONL files of subreddit data")
def main(subreddit, path_to_reddit_files):
    comment_files = glob(f'{path_to_reddit_files}/comments/{subreddit}_Comments/*')
    submission_files = glob(f'{path_to_reddit_files}/submissions/{subreddit}_Submissions/*')

    #remove likely bots by filtering out users with these substrings
    bot_substrings = ['bot','auto','transcriber','[deleted]','changetip','gif','bitcoin','tweet','messenger','mention','tube','link', 'b0t']

    #get pushshift comment + submission data
    if len(comment_files) > 0:
        comment_dfs = []
        comment_cols = ['id', 'created_utc', 'subreddit', 'author', 'body', 'score', 'parent_id', 'link_id', 'author_created_utc']
        submission_cols = ['id', 'created_utc', 'subreddit', 'author', 'title', 'selftext', 'author_created_utc', 'link_id']
        for file in comment_files:
            date = file.split('comments_')[-1].split('.')[0].split('_')
            df = pd.read_json(file, lines=True)
            kept_cols = []
            for col in comment_cols:
                if col in df:
                    kept_cols.append(col)
    
            comment_dfs.append(df[kept_cols])
    
        submission_dfs = []
        for file in submission_files:
            date = file.split('submissions_')[-1].split('.')[0].split('_')
            df = pd.read_json(file, lines=True)
            df['link_id'] = 't3_' + df['id']
    
            kept_cols = []
            for col in submission_cols:
                if col in df:
                    kept_cols.append(col)
    
            submission_dfs.append(df[kept_cols])
    
        subreddit_df = pd.concat(comment_dfs + submission_dfs)
        
        subreddit_df['text'] = subreddit_df.apply(make_text_col, axis=1)
        subreddit_df = subreddit_df.drop(columns=['body'])
        
        if len(submission_files) > 0:
            subreddit_df = subreddit_df.drop(columns=['title', 'selftext'])
            
        subreddit_df['stripped_parent_id'] = subreddit_df['parent_id'].apply(lambda x: x[3:] if str(x) != 'nan' else x) #create parent_id to rejoin
        subreddit_df = subreddit_df[~subreddit_df['author'].str.lower().apply(lambda x: any(sub in str(x) for sub in bot_substrings))] #remove likely bots

        #find first replies to newcomers
        reply_df = subreddit_df.sort_values(by='created_utc').drop_duplicates(subset=['stripped_parent_id'], keep='first')[['stripped_parent_id', 'text']].rename(columns={'stripped_parent_id':'id', 'text':'reply_clean'})
        min_timestamps = subreddit_df.groupby('author')['created_utc'].min().reset_index().rename(columns={'created_utc':'min_timestamp'})
        subreddit_df = subreddit_df.merge(min_timestamps, on='author', how='left')
        newcomer_df = subreddit_df[subreddit_df['min_timestamp'] == subreddit_df['created_utc']]
        newcomer_df = newcomer_df.merge(reply_df, how='left', on='id')

        #find last timestamps for authors
        whole_sub = subreddit_df.groupby(['author', 'link_id'])['created_utc'].min().reset_index()
        whole_sub = whole_sub.merge(min_timestamps, on='author', how='left')
        whole_sub['diff'] = whole_sub['created_utc'] - whole_sub['min_timestamp']
        author_max = whole_sub.groupby(['author'])['created_utc'].max().reset_index().rename(columns={'created_utc':'max_timestamp'})
        author_max = author_max.merge(min_timestamps, on='author')

        #measure 4 week activity rates
        time_threshold = 4 * 7 * 86400
        whole_sub[f'activity_4wks'] = (whole_sub['diff'] < time_threshold)
        activity = whole_sub.groupby('author')[f'activity_4wks'].sum().reset_index()
        author_max = author_max.merge(activity, on='author')
    
        newcomer_df = newcomer_df.merge(author_max.drop(columns='min_timestamp'), on='author', how='left')

        #find features of submissions (e.g., thread activity level)
        submissions = dict(tuple(subreddit_df.groupby('link_id')))
        
        results = []
        
        for i, row in newcomer_df.iterrows():
            user = row['author']
            submission = row['link_id']
            time = row['min_timestamp']
            
            # Get all comments in that submission before this comment
            sub_df = submissions[submission]
            sub_before = sub_df[sub_df['created_utc'] < time]
            
            # Calculate features
            row['submission_total_comments'] = len(sub_before)
            row['submission_unique_commenters'] = sub_before['author'].nunique()
            
            top_level_before = sub_before[sub_before['link_id'] == sub_before['parent_id']]
            row['submission_direct_replies'] = len(top_level_before)
            row['direct_replies_unique'] = top_level_before['author'].nunique()
            
            results.append(row)
        
        all_results = pd.DataFrame(results)

        reply_df = all_results[all_results['reply_text'].isnull() == False]

        #get sentiment/toxicity scores
        reply_df = reply_df.progress_apply(calculate_toxicity, axis=1)
        reply_df['reply_sentiment'] = reply_df['reply_text'].progress_apply(sentiment_scores)

        all_results = all_results.merge(reply_df[['id', 'toxicity', 'severe_toxicity', 'identity_attack', 'attack_on_commenter', 'reply_sentiment']], on='id', how='left')

        #save dataframe
        all_results.to_csv(f'../../data/subreddit_data/{subreddit}_full_data.csv', index=False)


if __name__ == "__main__":
    main()