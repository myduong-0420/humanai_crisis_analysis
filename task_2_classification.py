import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import gensim.downloader as api
from gensim.models import Word2Vec

tqdm.pandas()

def vader_sentiment(text):
	analyzer = SentimentIntensityAnalyzer()
	sentiment = analyzer.polarity_scores(text)
	return sentiment["neg"], sentiment["neu"], sentiment["pos"] 	

def textblob_sentiment(text):
	text_doc = TextBlob(text)
	sentiment = text_doc.sentiment
	return sentiment[0], sentiment[1]

def df_sentiment(df):
	for i, row in tqdm(df.iterrows(), total=len(df)):
		if pd.isnull(row["content"]):
			text = row["title"]
		text = str(row["title"]) + " " + str(row["content"])
		neg, neu, pos = vader_sentiment(text)
		df.at[i, "v_neg"] = neg
		df.at[i, "v_neu"] = neu
		df.at[i, "v_pos"] = pos
		df.at[i, "t_polarity"] = textblob_sentiment(text)[0]
		df.at[i, "t_subjectivity"] = textblob_sentiment(text)[1]
	return df

if __name__ == "__main__":
    df = pd.read_csv(rf"D:\humanai_crisis_analysis\data\reddit_posts.csv")
    df_with_sa = df_sentiment(df)
    df_with_sa.to_csv(rf"D:\humanai_crisis_analysis\data\reddit_data_with_sa.csv", index=False)
    print("Done!")