import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

# ─────────────────────────────────────────
# STEP 1 — Load CSVs
# ─────────────────────────────────────────
print("Loading CSVs...")
df_cups = pd.read_csv("data/WorldCups.csv")
df_matches = pd.read_csv("data/WorldCupMatches.csv")

# Remove empty rows
df_cups = df_cups.dropna(subset=["Year"])
df_matches = df_matches.dropna(subset=["Year"])

print(f"Tournaments loaded: {len(df_cups)}")
print(f"Matches loaded: {len(df_matches)}")

# ─────────────────────────────────────────
# STEP 2 — Format Documents
# ─────────────────────────────────────────
print("Formatting documents...")
documents = []

# Format WorldCups.csv rows
for _, row in df_cups.iterrows():
    text = f"""Record Type: Tournament
Year: {row['Year']}
Host Country: {row['Country']}
Winner: {row['Winner']}
Runner-up: {row['Runners-Up']}
Third Place: {row['Third']}
Fourth Place: {row['Fourth']}
Goals Scored: {row['GoalsScored']}
Qualified Teams: {row['QualifiedTeams']}
Matches Played: {row['MatchesPlayed']}
Attendance: {row['Attendance']}"""
    documents.append(Document(page_content=text))

# Format WorldCupMatches.csv rows
for _, row in df_matches.iterrows():
    text = f"""Record Type: Match
Year: {row['Year']}
Stage: {row['Stage']}
Date: {row['Datetime']}
Home Team: {row['Home Team Name']}
Away Team: {row['Away Team Name']}
Full Time Score: {row['Home Team Goals']} - {row['Away Team Goals']}
Half Time Score: {row['Half-time Home Goals']} - {row['Half-time Away Goals']}
Stadium: {row['Stadium']}
City: {row['City']}
Attendance: {row['Attendance']}
Referee: {row['Referee']}
Win Conditions: {row['Win conditions']}"""
    documents.append(Document(page_content=text))

print(f"Total documents created: {len(documents)}")

# ─────────────────────────────────────────
# STEP 3 — Create Embeddings + Save FAISS
# ─────────────────────────────────────────
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating embeddings and saving to FAISS...")
print("This may take a few minutes on first run...")
vector_store = FAISS.from_documents(documents, embeddings)

# Save to disk
vector_store.save_local("vector_store")
print("Vector store saved to disk!")
print("Ingestion complete!")