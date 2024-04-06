import nltk
import json
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("data.json") as f:
    d = json.load(f)

grants_data = d


# Step 3: Compute Similarity
def find_matching_grants(stage, input_criteria, grants_data, threshold):

    filtered_data = [item for item in grants_data if stage in item["stage"]]
    # Step 1: Preprocess the Data
    grants_eligibility = [grant["eligibilityKeyword"] for grant in filtered_data]

    # Step 2: Vectorize Text
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        [" ".join(criteria) for criteria in grants_eligibility]
    )

    input_criteria_text = " ".join(input_criteria)
    input_criteria_vector = tfidf_vectorizer.transform([input_criteria_text])
    similarity_scores = cosine_similarity(input_criteria_vector, tfidf_matrix)

    # Step 4: Rank Grants and Return Top Matches
    matching_grants = []
    for idx, score in enumerate(similarity_scores.flatten()):
        if score > threshold:
            matching_grants.append((filtered_data[idx], score))

    matching_grants.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity score

    return matching_grants


def process_input(input_text):
    # Tokenize the input text
    tokens = nltk.word_tokenize(input_text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    return filtered_tokens


def search_grants(input_text, items):
    # Process the input text
    keywords = process_input(input_text)

    # Define list for words to be filtered out
    words_to_filter_out = [
        "business",
        "company",
        "owner",
    ]

    # Filter out specific keywords
    filtered_keywords = [
        keyword for keyword in keywords if keyword.lower() not in words_to_filter_out
    ]

    print(filtered_keywords)

    matched_items = []
    for item in items:
        # Check if any keyword matches in fullDescription
        if any(
            keyword.lower() in item["fullDescription"].lower()
            for keyword in filtered_keywords
        ):
            matched_items.append(item)
        else:
            # Check if any keyword matches in eligibilities
            if item["eligibility"]:
                if any(
                    keyword.lower() in elig.lower()
                    for keyword in filtered_keywords
                    for elig in item["eligibility"]
                ):
                    matched_items.append(item)

    return matched_items


class Item(BaseModel):
    text: str = None


class Profile(BaseModel):
    stage: str
    eligibilities: list[str]


@app.get("/")
def root():
    return grants_data


@app.post("/search/")
def getSearch(item: Item):
    return search_grants(item.text, grants_data)


@app.post("/recommendations/")
def getRecommendations(profile: Profile):
    matching_grants = find_matching_grants(
        profile.stage,
        profile.eligibilities,
        grants_data,
        threshold=0.5,
    )

    return matching_grants
