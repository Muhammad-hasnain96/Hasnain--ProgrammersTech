It looks like you want to save your movie recommendation system's state and then load it later. Also, you want to create a README file to document your work.

Here's a README file template for your project:

```markdown
# Movie Recommendation System

This project creates a movie recommendation system using movie data and collaborative filtering techniques.

## Project Structure

- **tmdb_5000_movies.csv**: Contains movie details such as genres, keywords, etc.
- **tmdb_5000_credits.csv**: Contains credits data like cast and crew information.
- **app.py**: Streamlit app to run the recommendation system.
- **movie_list.pkl**: Pickle file to store processed movie data.
- **similarity.pkl**: Pickle file to store similarity matrix.

## Installation

1. Clone the repository or download the files.
2. Install required libraries:

```bash
pip install pandas numpy scikit-learn streamlit
```

## Usage

1. Prepare the data and run the script:

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import ast

# Load the data
movies = pd.read_csv(r"C:\Users\NAC\movie recommendation\tmdb_5000_movies.csv")
credits = pd.read_csv(r"C:\Users\NAC\movie recommendation\tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Function to convert stringified lists to list of names
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

# Drop NA values
movies.dropna(inplace=True)

# Apply conversion functions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3])
movies['crew'] = movies['crew'].apply(lambda x: [i['name'].replace(" ", "") for i in ast.literal_eval(x) if i['job'] == 'Director'])

# Create tags column
movies['tags'] = movies['overview'].apply(lambda x: x.split()) + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies[['movie_id', 'title', 'tags']]
new['tags'] = new['tags'].apply(lambda x: " ".join(x))

# Convert tags to vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()

# Calculate similarity
similarity = cosine_similarity(vector)

# Function to recommend movies
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)

# Test recommendation
recommend('Gandhi')

# Save processed data
pickle.dump(new, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
```

2. Run the Streamlit app:

```bash
streamlit run "c:/Users/NAC/movie recommendation/app.py"
```

## How It Works

1. The script reads movie and credits data.
2. It processes the data to extract relevant information and create tags for each movie.
3. It calculates the cosine similarity between movie tags.
4. It provides recommendations based on the similarity scores.

## Troubleshooting

- Ensure the file paths are correct.
- Check the encoding of your CSV files if you encounter any decoding errors.
- Update any deprecated functions as needed (e.g., `beta_columns` to `columns`).

## Acknowledgements

- TMDB for providing the movie dataset.
- Streamlit for the interactive UI.

## License

This project is licensed under the MIT License.
```

Save this content as `README.md` in your project directory. This file provides an overview of the project, instructions for installation and usage, and additional information about the functionality and acknowledgements.
