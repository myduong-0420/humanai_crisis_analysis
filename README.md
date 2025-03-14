# AI-Powered Behavioral Analysis for Suicide Prevention, Substance Use, and Mental Health Crisis Detection with Longitudinal Geospatial Crisis Trend

### Task 1. Social Media Data Extraction & Preprocessing (API Handling & Text Cleaning)
📌 **Deliverable:**
- `task_1_scrape_clean.py` that retrieves and stores **filtered social media posts**.
- `reddit_posts_cleaned.csv` ready for NLP analysis.

The main submission with tokenized text has the name `reddit_posts_cleaned.csv`, but additional datasets were curated to work with task 2 and task 3 more conveniently. `reddit_data_with_sa.csv` contains sentiment scores generated by VADER and Textblob, `sample_test_with_risk_words.csv` contains risk words and severity of the content, and `reddit_posts_geotags.csv` contains ngrams isolated for geotagging, and geographical cues.

I choose to focus on Reddit data, since I want to try tackling insufficient geotagging with NLP methods. Besides, in sensitive areas like mental health and drug use, it's understandable that original posters (OPs) often choose not to disclose their locations, so geotagging will only work when OPs willingly share about their whereabouts in the posts. While the lack of geographical information feature presents a big challenge about data representation, it is also an ethical standard that I want to uphold. One area that I would like to explore further is to visit regional subreddits (E.g, r/tampa) and make queries with selected keywords, but due to time and compute constraint (explained in the next task), I was not able to do this.


### Task 2. Sentiment & Crisis Risk Classification (NLP & Text Processing)
📌 **Deliverable:**
- `task_2_classification.py` that classifies posts based on **sentiment and risk level**.
- `sentiment_risk_distribution.png` showing the **distribution of posts by sentiment and risk category**.

The Python script is meant for the whole dataset, but due to the size of the original dataset (21K entries), it would take me an estimated time of **17 hours** to extrapolate all risk words and classify which risk category each post falls in. I decided to sample 500 random posts from the dataset and generate `sample_test_with_risk_words.csv`, which has all the sentiment scores, risk words and risk category. This process took around 2 hours to run.

The distribution of posts by sentiment is not surprising - most posts have negative sentiment, followed by positive and neutral. In searching for risk words, I also pulled reference text from various sources on the internet, including counselling websites and AI-generated phrases for different risk levels. I choose the BERT model `all-MiniLM-L6-v2` due to its ability to work with context and semantic dependencies over Word2Vec, and with this, I created 5-grams for each post and compared them with reference text with a similarity score > 0.7, thus isolating the highest-ranked 5-grams as risk phrases. This is the most resource-intensive process out of the whole pipeline that I worked on.


### Task 3. Crisis Geolocation & Mapping (Basic Geospatial Analysis & Visualization)
📌 **Deliverable:**
- `task_3_geospatial.py` that geocodes posts and generates a **heatmap** of crisis discussions.
- `global_heatmap.html` - A **visualization of regional distress patterns** in the dataset.

This task is open to new ideas. Acknowledging that Reddit **does not** support geotagging, I instantly thought of NER location extraction and optimal geotagging with context. I went with 2 popular NER models, which are Spacy NER and Flair NER, but only generated location with Spacy since Flair is a highly accurate but expensive model. In isolating location tags detected by Spacy (`LOC`, `GPE`), I replaced these locations with `LOC_MASK_#` for efficiency, which not only made it easier for tracing back relevant locations, but also acted as a good constraint for efficient n-grams and relevancy to OPs' whereabouts.

It is important to recognize that sometimes, OPs also mention places that only add context to their stories instead of accurately representing their locations. Because of this, I added a filter of 4-grams around the location masks `LOC_MASK_#`, and compared these with reference text of location and assistance in geographical areas, thus generating location labels for posts if available.

I used Geopy and Folium for generating the discussion heatmap.

### Setbacks
- My approaches, especially for task 2 and 3 are still pretty naive. In task 2, I assumed a simple rule-based approach for classifying risk levels. I originally thought of assigning weights to risk words (higher score for higher risk level and higher tf-idf), but I could not find empirical methods to base this on.
- While I generated a fair amount of data, there are still several problems like NER models, (risk and sentiment) class imbalance, text cleaning and the lack of geospatial data (only 222/20962 entries contain meaningful location information) that hinder the accurate mapping of mental health and substance use. Besides, by naively approaching task 3 with strict formatting rules, I may run the risk of overcleaning location labels while misclassifying irrelevant labels.