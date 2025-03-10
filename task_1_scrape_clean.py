import praw
import json
import os
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import demoji
from datetime import datetime

tqdm.pandas()

def load_secrets(filepath):
    try:
        with open(filepath, 'r') as f:
            secrets = json.load(f)
        return secrets
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return None
    
def connect_to_reddit(filepath):
    secrets = load_secrets(filepath)
    if secrets:
        id = secrets.get('client_id')
        secret = secrets.get('client_secret')
        user = secrets.get('user_name')
        
        reddit = praw.Reddit(
            client_id=id,      
            client_secret=secret,
            user_agent=f"u/{user}"
        )
        return reddit

def get_subreddit_titles(filepath, subreddit_name, post_limit=10):
    reddit = connect_to_reddit(filepath)  
    subreddit = reddit.subreddit(subreddit_name)  # Choose the subreddit

    recent_posts = subreddit.new(limit=post_limit)

    post_titles = [post.title for post in recent_posts]
    return post_titles

def get_posts_by_query(filepath, query, post_limit=10, subreddit_name="all"):
    data = {"post_id": [], "title": [], "likes": [], "comments": [], "content": [], "date": []}
    reddit = connect_to_reddit(filepath)

    results_generator = reddit.subreddit(subreddit_name).search(query, limit=post_limit)
    results_list = list(results_generator)
    data["post_id"] = [post.id for post in results_list]
    data["title"] = [post.title for post in results_list]
    data["likes"] = [post.score for post in results_list]
    data["comments"] = [post.num_comments for post in results_list]
    data["shares"] = [post.ups for post in results_list]
    data["content"] = [post.selftext for post in results_list]
    data["date"] = [post.created_utc for post in results_list]
    data["keywords"] = [query for _ in results_list]

    return pd.DataFrame(data)

# nltk.download('stopwords')
# nltk.download('punkt_tab')

def remove_special_characters(text):
	return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_stopwords(text):
	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(text)
	filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
	return filtered_sentence

def remove_emojis(text):
    return demoji.replace(text, '')

def full_preprocessing_pipeline(text):
    if pd.isnull(text):
        text = ""
    text = str(text)

    text = remove_special_characters(text)
    text = remove_emojis(text)
    text = remove_stopwords(text)
    return text


# Note: r/addiction doesn't allow for internal search through API

if __name__ == "__main__":
    secrets = "confidential.json"
    keyword_list = [
        "drug addiction",
        "Suicide Watch",
        "suicide recovery",
        "mental distress",
        "depression",
        "substance abuse",
        "mental health recovery",
        "drug and mental health",
        "suicidal help",
        "mental therapy",
        "mental therapy area",
        "suicidal thought",
        "suicide attempt",
        "do drug",
        ""
    ]

    subreddit_list = [
        "all",
        "SuicideWatch",
        "MentalHealthSupport",
        "mentalhealth",
        "addiction",
        "DrugAddiction",
        "depression",
        "depression_help",
        "therapy"
    ]

    post_limit = 4000
    df = pd.DataFrame()

    for keyword in tqdm(keyword_list, desc="Searching subreddit - keyword combinations"):
        for subreddit in subreddit_list:
            try:
                data = get_posts_by_query(
                    filepath=secrets,
                    query=keyword,
                    subreddit_name=subreddit,
                    post_limit=post_limit
                )
                df = pd.concat([df, data], ignore_index=True)
            except Exception as e:
                # If any error occurs (like an empty column issue or PRAW error),
                # we skip this subreddit and continue with the next.
                print(f"Skipped subreddit '{subreddit}' does not support internal retrieval.")
                continue

    df = df.drop_duplicates(subset=["post_id"])
    print("Final DataFrame shape:", df.shape)
    df["title"] = df["title"].progress_apply(full_preprocessing_pipeline)
    df["content"] = df["content"].progress_apply(full_preprocessing_pipeline)
    df["date_readable"] = df["date"].apply(
        lambda t: datetime.utcfromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
        )
    # df.to_csv(rf"D:\humanai_crisis_analysis\data\reddit_posts.csv", index=False)
