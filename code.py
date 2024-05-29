# Reputation Monitoring System

# Code cell 1: Set up programming environment to use code to send prompts to OpenAI's cloud-hosted service.

import openai
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")

def llm_response(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role':'user','content':prompt}],
        temperature=0
    )
    return response.choices[0].message['content']

# Code cell 2: Create a list of reviews.

all_reviews = [
    'The mochi is excellent!',
    'Best soup dumplings I have ever eaten.',
    'Not worth the 3 month wait for a reservation.',
    'The colorful tablecloths made me smile!',
    'The pasta was cold.',
    'The service was outstanding and the ambiance was wonderful.',
    'The dessert was too sweet and overpriced.',
    'Amazing experience, will definitely come back!',
    'The steak was overcooked and tough.',
    'Loved the live music, it really set the mood.'
]

# Display the reviews
print(all_reviews)

# Code cell 3: Classify the reviews as positive or negative.

all_sentiments = []
for review in all_reviews:
    prompt = f'''
        Classify the following review 
        as having either a positive or
        negative sentiment. State your answer
        as a single word, either "positive" or
        "negative":

        {review}
    '''
    response = llm_response(prompt).strip().lower()
    all_sentiments.append(response)

# Display the sentiments
print(all_sentiments)

# Code cell 4: Count the number of positive and negative reviews and visualize the results.

# Create a DataFrame for better manipulation and visualization
df = pd.DataFrame({
    'Review': all_reviews,
    'Sentiment': all_sentiments
})

# Count the sentiments
sentiment_counts = df['Sentiment'].value_counts()

# Print the counts
num_positive = sentiment_counts.get('positive', 0)
num_negative = sentiment_counts.get('negative', 0)
print(f"There are {num_positive} positive and {num_negative} negative reviews.")

# Visualize the sentiments
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=df, palette='coolwarm')
plt.title('Sentiment Analysis of Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
