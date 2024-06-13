import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import re

# Combined function to fetch, preprocess text and return filtered words
def fetch_and_preprocess_text(url):
    # Define a set of stop words
    STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the",
    "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.body.get_text(separator=' ')
    
    text = text.lower()  # convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    words = text.split()  # split into words
    
    filtered_words = [word for word in words if word not in STOP_WORDS]
    return filtered_words

# Function to plot words using matplotlib
def plot_word_cloud(unique_words, word_frequencies):
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    def random_color():
        return f'#{random.randint(0, 0xFFFFFF):06x}'
    
    max_freq = sorted(word_frequencies)[-1]
    scaling_factor = 100 / max_freq  # Adjust as needed for different text sizes
    
    for i in range(len(unique_words)):
        if word_frequencies[i] > 3:
            plt.text(
                random.uniform(0, 1), 
                random.uniform(0, 1), 
                unique_words[i], 
                fontsize=word_frequencies[i] * scaling_factor, 
                color=random_color(), 
                ha='center', 
                va='center', 
                alpha=0.7
            )
    
    plt.show()

def plot_points(points_data, pi_approximation):
    # Visualize the points and the approximated circle using matplotlib
    inside_x, inside_y = zip(*points_data["inside"])
    outside_x, outside_y = zip(*points_data["outside"])
    print(points_data["inside"])
    plt.figure(figsize=(10, 10))
    plt.scatter(inside_x, inside_y, color='blue', label='Inside Circle')
    plt.scatter(outside_x, outside_y, color='red', label='Outside Circle')
    circle = plt.Circle((0, 0), 1, color='green', fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Approximation of pi using Monte Carlo Method\npi ≈ {pi_approximation}')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_original_decoded_images(original_image, decoded_image):
    # Read the original image
    original_image = mpimg.imread(original_image)

    # Read the decoded image
    decoded_image = mpimg.imread(decoded_image)

    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original image in the first subplot
    axs[0].imshow(original_image)
    axs[0].set_title('Original')
    axs[0].axis('off')  # Hide the axes

    # Display the decoded image in the second subplot
    axs[1].imshow(decoded_image)
    axs[1].set_title('Decoded')
    axs[1].axis('off')  # Hide the axes

    # Show the plot
    plt.show()
