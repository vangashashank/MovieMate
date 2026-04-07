#  MovieMate: Conversational AI for Intelligent Movie Search

**MovieMate** is an intelligent conversational AI system designed to move beyond traditional keyword-based search. [cite_start]It combines **Natural Language Processing (NLP)** and **Large Language Models (LLMs)** with a structured movie dataset to create an intuitive, interactive discovery experience[cite: 4, 6].

##  Project Overview
[cite_start]The goal of this project is to assist users in navigating large entertainment datasets through natural language interactions[cite: 4]. [cite_start]Instead of relying on rigid filters, MovieMate understands flexible queries like *"Suggest sci-fi movies similar to Interstellar but less than 2 hours long"*[cite: 5, 15].

##  Technical Architecture
[cite_start]The system follows a **Retrieval-Augmented Generation (RAG)** pipeline[cite: 80, 175]:
1. [cite_start]**Natural Language Understanding (NLU):** Uses the **Google Gemini 1.5 Flash** model to interpret user intent and extract semantic constraints (e.g., Genre, Year, Rating)[cite: 68, 71, 75].
2. [cite_start]**Vector Embeddings:** Movie metadata is converted into high-dimensional vectors using the `all-MiniLM-L6-v2` model[cite: 78, 102].
3. [cite_start]**Similarity Search:** Efficient nearest-neighbor search implemented with **FAISS (Facebook AI Similarity Search)** to identify the most relevant matches[cite: 108, 109].
4. [cite_start]**Response Generation:** Relevant movies are retrieved and processed by the LLM to generate natural, human-readable responses[cite: 116, 125].
5. [cite_start]**Interface:** A lightweight, web-based UI built with **Gradio** for real-time user interaction[cite: 128, 130].

##  Data Acquisition & Construction
[cite_start]The dataset was programmatically constructed using the **TMDb (The Movie Database) API**[cite: 143, 144].
* [cite_start]**Source:** Real-world movie metadata retrieved programmatically to ensure structured access[cite: 145, 156].
* [cite_start]**Fields:** Includes Title, Rating, Year, Genre, Director, Cast, and Overview[cite: 164, 173].
* [cite_start]**Preprocessing:** Handled missing values, normalized text fields, and cleaned metadata to prepare it for embedding generation[cite: 88, 91, 93].

##  Key Features
* [cite_start]**Natural Language Search:** Query using conversational language instead of structured filters[cite: 35].
* [cite_start]**Conversational Interaction:** Supports multi-turn dialogues, allowing users to refine queries (e.g., "Only those released after 2015")[cite: 43, 47].
* [cite_start]**Intelligent Information Retrieval:** Extracts details such as Ratings, Genres, and Directors to build informative responses[cite: 58, 64].
* [cite_start]**Personalized Exploration:** Designed to potentially adapt recommendations based on user interaction patterns[cite: 50, 55].

##  Tech Stack
* [cite_start]**Language:** Python [cite: 86]
* [cite_start]**AI/ML:** Sentence-Transformers, FAISS, Google Generative AI (Gemini) [cite: 109, 115]
* [cite_start]**Data Science:** Pandas, NumPy [cite: 86]
* [cite_start]**Visualization:** Matplotlib, Seaborn [cite: 100]
* [cite_start]**UI:** Gradio [cite: 130]

##  Project Deliverables
[cite_start]The project includes a **Jupyter Notebook** covering the following components[cite: 191]:
1. [cite_start]**Dataset Exploration:** Features, summary statistics, and observations[cite: 192, 195].
2. [cite_start]**Exploratory Data Analysis (EDA):** Visualizations of ratings and genres[cite: 196, 197].
3. [cite_start]**Data Preprocessing:** Cleaning, preparing, and handling missing values[cite: 199, 201].
4. [cite_start]**Embedding and Retrieval:** Generating vector representations and implementing similarity search[cite: 202, 204].
5. [cite_start]**Conversational Movie Chatbot:** Integrating retrieval with an LLM[cite: 205, 206].
6. [cite_start]**Interactive Interface:** Simple web interface for user interaction[cite: 208, 209].
7. [cite_start]**Evaluation and Reflection:** Performance observations and limitations[cite: 210, 212].

---
*Created by Vanga Shashank Goud as part of an NLP assignment focusing on LLM-powered information systems.*
