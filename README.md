#  MovieMate: Conversational AI for Intelligent Movie Search

**MovieMate** is an intelligent conversational AI system designed to move beyond traditional keyword-based search. It combines **Natural Language Processing (NLP)** and **Large Language Models (LLMs)** with a structured movie dataset to create an intuitive, interactive discovery experience.

##  Project Overview
The goal of this project is to assist users in navigating large entertainment datasets through natural language interactions. Instead of relying on rigid filters, MovieMate understands flexible queries like *"Suggest sci-fi movies similar to Interstellar but less than 2 hours long"*.

##  Technical Architecture
The system follows a **Retrieval-Augmented Generation (RAG)** pipeline:
1. **Natural Language Understanding (NLU):** Uses the **Google Gemini 1.5 Flash** model to interpret user intent and extract semantic constraints (e.g., Genre, Year, Rating).
2. **Vector Embeddings:** Movie metadata is converted into high-dimensional vectors using the `all-MiniLM-L6-v2` model.
3. **Similarity Search:** Efficient nearest-neighbor search implemented with **FAISS (Facebook AI Similarity Search)** to identify the most relevant matches.
4. **Response Generation:** Relevant movies are retrieved and processed by the LLM to generate natural, human-readable responses.
5. **Interface:** A lightweight, web-based UI built with **Gradio** for real-time user interaction.

##  Data Acquisition & Construction
The dataset was programmatically constructed using the **TMDb (The Movie Database) API**.
* **Source:** Real-world movie metadata retrieved programmatically to ensure structured access.
* **Fields:** Includes Title, Rating, Year, Genre, Director, Cast, and Overview.
* **Preprocessing:** Handled missing values, normalized text fields, and cleaned metadata to prepare it for embedding generation.

##  Key Features
* **Natural Language Search:** Query using conversational language instead of structured filters.
* **Conversational Interaction:** Supports multi-turn dialogues, allowing users to refine queries (e.g., "Only those released after 2015").
* **Intelligent Information Retrieval:** Extracts details such as Ratings, Genres, and Directors to build informative responses.
* **Personalized Exploration:** Designed to potentially adapt recommendations based on user interaction patterns.

##  Tech Stack
* **Language:** Python 
* **AI/ML:** Sentence-Transformers, FAISS, Google Generative AI (Gemini) 
* **Data Science:** Pandas, NumPy 
* **Visualization:** Matplotlib, Seaborn 
* **UI:** Gradio

##  Project Deliverables
The project includes a **Jupyter Notebook** covering the following components:
1. **Dataset Exploration:** Features, summary statistics, and observations.
2. **Exploratory Data Analysis (EDA):** Visualizations of ratings and genres.
3. **Data Preprocessing:** Cleaning, preparing, and handling missing values.
4. **Embedding and Retrieval:** Generating vector representations and implementing similarity search.
5. **Conversational Movie Chatbot:** Integrating retrieval with an LLM.
6. **Interactive Interface:** Simple web interface for user interaction.
7. **Evaluation and Reflection:** Performance observations and limitations.

---
*Created by Vanga Shashank Goud as part of an NLP assignment focusing on LLM-powered information systems.*
