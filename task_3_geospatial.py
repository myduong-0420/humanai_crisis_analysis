import pandas as pd
import spacy
from flair.nn import Classifier
from flair.data import Sentence
from tqdm import tqdm
import re
from task_2_classification import seed_embeddings
import folium
import time
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from collections import Counter
from sentence_transformers import util

seed_phrases = ["I live in [LOC_MASK]", 
              "I reside in [LOC_MASK]", 
              "I stay in [LOC_MASK]", 
              "based in [LOC_MASK]",
              "help in [LOC_MASK]",
              "need assistance in [LOC_MASK]",
              "I am from [LOC_MASK]",
              "located in [LOC_MASK]",
              "therapy near [LOC_MASK]",]

def extract_geotags(text, model_name="spacy"):
    """Extract all location entities (GPE and LOC) from text."""
    if model_name == "spacy":
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
    elif model_name == "flair":
        sentence = Sentence(text)
        model = Classifier.load("ner-ontonotes")
        model.predict(sentence)
        return [entity.text for entity in sentence.get_spans("ner") if entity.tag in ['GPE', 'LOC']]

def mask_location_unique(text, loc, mask_token):
    """
    Replace all occurrences of 'loc' in the text with a unique mask token.
    """
    pattern = r'\b' + re.escape(loc) + r'\b'
    masked_text = re.sub(pattern, mask_token, text, flags=re.IGNORECASE)
    return masked_text

def generate_loc_ngram(text, ngram=4):
    """
    Tokenize text (preserving tokens like [LOC_MASK_#]) and generate all n-grams of length `ngram`
    that include at least one mask token.
    """
    tokens = re.findall(r'\[LOC_MASK(?:_\d+)?\]|\w+', text)
    loc_ngrams = []
    for i in range(len(tokens) - ngram + 1):
        window = tokens[i:i+ngram]
        if any(token.startswith("[LOC_MASK") for token in window):
            loc_ngrams.append(" ".join(window))
    return loc_ngrams

def detect_top_relevant_ngrams(seed_phrases, ngrams, threshold=0.7):
    """
    Given a list of reference seed phrases and a list of n-grams,
    return the n-gram that is most similar to any seed phrase
    (if its cosine similarity is above the threshold).
    """
    ref_embeddings = seed_embeddings(seed_phrases)
    if not ngrams:
        return None

    ngram_embeddings = seed_embeddings(ngrams)
    cos_sim_matrix = util.cos_sim(ngram_embeddings, ref_embeddings)
    flagged_ngrams = []
    for i, ngram in enumerate(ngrams):
        row_similarities = cos_sim_matrix[i]
        if row_similarities.max() >= threshold:
            best_seed_idx = int(row_similarities.argmax())
            best_seed_phrase = seed_phrases[best_seed_idx]
            best_score = float(row_similarities[best_seed_idx])
            flagged_ngrams.append({
                "ngram": ngram,
                "similar_seed_phrase": best_seed_phrase,
                "similarity_score": best_score
            })
    if not flagged_ngrams:
        return None
    max_score = max(d["similarity_score"] for d in flagged_ngrams)
    for d in flagged_ngrams:
        if d["similarity_score"] == max_score:
            return d["ngram"]
    return None

def is_valid_location(loc):
    invalid_keywords = ['meth','Reddit', 'XYZ', 'AI', 'Nortriptyline', 
                        'Lyrica', 'Modafinil', 'Kaiser', 'Parent', 
                        'Niece', 'Husband', 'Spring Sem']
    return not any(keyword.lower() in loc.lower() for keyword in invalid_keywords)

def geocode_location(location):
    try:
        loc = geolocator.geocode(location)
        if loc:
            return (loc.latitude, loc.longitude)
    except Exception as e:
        print(f"Error geocoding {location}: {e}")
    return None

if __name__ == "__main__":
    data_path = rf"D:\humanai_crisis_analysis\data"
    df = pd.read_csv(rf"{data_path}\reddit_posts.csv")
    df["masked_text"] = None
    df["mapping"] = None
    df["loc_ngrams"] = None
    df["detected_location"] = None

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        text = row["full_text"]
        
        # Step 1: Extract location entities from the text
        geotags = extract_geotags(text)
        
        mapping = {}  # map unique mask tokens to the original location
        masked_text = text
        token_counter = 1
        
        # Step 2: For each unique location, create a unique token and mask it in the text
        for loc in list(set(geotags)):
            if loc == "meth" or loc == "reddit":
                continue
            mask_token = f"[LOC_MASK_{token_counter}]"
            mapping[mask_token] = loc
            masked_text = mask_location_unique(masked_text, loc, mask_token)
            token_counter += 1

        df.at[i, "masked_text"] = masked_text
        df.at[i, "mapping"] = mapping
        
        # Step 3: Generate n-grams from the masked text
        ngrams = generate_loc_ngram(masked_text, ngram=4)
        df.at[i, "loc_ngrams"] = ngrams
        
        # Step 4: Use the seed phrases to detect the most relevant n-gram.
        top_ngram = detect_top_relevant_ngrams(seed_phrases, ngrams, n=4, threshold=0.92)
        
        # Step 5: Find which unique mask token is in the top n-gram and trace back the original location
        detected_location = None
        if top_ngram is not None:
            for token in mapping.keys():
                if token in top_ngram:
                    detected_location = mapping[token]
                    break
        df.at[i, "detected_location"] = detected_location

    loc_list = []
    count = 0
    for i, row in df.iterrows():
        if row["detected_location"] is not None:
            loc_list.append(row["detected_location"])
            count += 1

    clean_locations = [loc for loc in loc_list if is_valid_location(loc)]
    geolocator = Nominatim(user_agent="geo_heatmap_example")

    # Geocode each location with a delay to avoid rate limiting
    geocoded_data = []
    for loc in clean_locations:
        coord = geocode_location(loc)
        if coord:
            geocoded_data.append(coord)
        time.sleep(1)

    freq = Counter(geocoded_data)

    df = pd.DataFrame([(lat, lon, count) for (lat, lon), count in freq.items()],
                    columns=['lat', 'lon', 'count'])

    # Folium map
    m = folium.Map(location=[20,0], zoom_start=2)

    # heat data (latitude, longitude, weight)
    heat_data = [[row['lat'], row['lon'], row['count']] for index, row in df.iterrows()]

    # HeatMap layer
    HeatMap(heat_data, radius=15).add_to(m)