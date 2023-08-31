import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample conversation pairs (user input, bot response)
conversation_pairs = [
    ("Hello", "Hi there! How can I assist you?"),
    ("What's the weather today?", "The weather is sunny and warm."),
    ("Tell me a joke", "Sure, here's one: Why did the scarecrow win an award? Because he was outstanding in his field!"),
    ("Goodbye", "Goodbye! Have a great day!")
]

# Extract user inputs and bot responses
user_inputs = [pair[0] for pair in conversation_pairs]
bot_responses = [pair[1] for pair in conversation_pairs]

# Create a TF-IDF vectorizer and fit on user inputs
vectorizer = TfidfVectorizer()
user_input_vectors = vectorizer.fit_transform(user_inputs)

def get_bot_response(user_input):
    # Transform the user input into a TF-IDF vector
    user_input_vector = vectorizer.transform([user_input])
    
    # Calculate cosine similarities between user input and bot responses
    similarity_scores = cosine_similarity(user_input_vector, user_input_vectors)
    
    # Find the index of the most similar bot response
    most_similar_index = np.argmax(similarity_scores)
    
    # Return the corresponding bot response
    return bot_responses[most_similar_index]

# Chat loop
print("Bot: Hi there! How can I assist you?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye! Have a great day!")
        break
    bot_response = get_bot_response(user_input)
    print("Bot:", bot_response)
