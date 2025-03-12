import geopandas as gpd
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import spacy
from spacy import displacy
from flair.nn import Classifier
from flair.data import Sentence
from tqdm import tqdm

def ner_tagging(df, model="spacy"):
    all_geotags = []
    df["geotags"] = None 
    if model == "spacy":
        model = spacy.load("en_core_web_sm")
        for i, row in tqdm(df.iterrows(), total=len(df), desc="NER Tagging"):
            geotags = []
            doc = model(row["full_text"])
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC"):
                    geotags.append(ent.text)
            df.at[i, "geotags"] = geotags
            all_geotags.append(geotags)
    elif model == "flair":
        model = Classifier.load("ner-ontonotes")
        for i, row in tqdm(df.iterrows(), total=len(df), desc="NER Tagging"):
            geotags = []
            sentence = Sentence(row["full_text"])
            model.predict(sentence)
            for entity in sentence.get_spans("ner"):
                if entity.tag in ("GPE", "LOC"):
                    geotags.append(entity.text)
            df.at[i, "geotags"] = geotags
            all_geotags.append(geotags)

    return df

