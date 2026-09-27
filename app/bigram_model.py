import re
import random
from collections import defaultdict, Counter

class bigram_model:
    def __init__(self, corpus):
        # Join the list of sentences into a single string for the tokenizer
        text = " ".join(corpus)
        _, self.bigram_probs = self.analyze_bigrams(text)

    def simple_tokenizer(self, text, frequency_threshold=None):
        """Simple tokenizer that splits text into words.""" #[cite: 7]
        # Convert to lowercase and extract words using regex
        tokens = re.findall(r"\b\w+\b", text.lower()) #[cite: 7]
        if not frequency_threshold: #[cite: 7]
            return tokens #[cite: 7]
        
        # Count word frequencies
        word_counts = Counter(tokens) #[cite: 7]
        filtered_tokens = [ #[cite: 7]
            token for token in tokens if word_counts[token] >= frequency_threshold #[cite: 7]
        ] #[cite: 7]
        return filtered_tokens #[cite: 7]

    def analyze_bigrams(self, text, frequency_threshold=None):
        """Analyze text to compute bigram probabilities.""" #[cite: 7]
        words = self.simple_tokenizer(text, frequency_threshold) #[cite: 7]
        bigrams = list(zip(words[:-1], words[1:]))  # Create bigrams #[cite: 7]

        # Count bigram and unigram frequencies
        bigram_counts = Counter(bigrams) #[cite: 7]
        unigram_counts = Counter(words) #[cite: 7]

        # Compute bigram probabilities
        bigram_probs = defaultdict(dict) #[cite: 7]
        for (word1, word2), count in bigram_counts.items(): #[cite: 7]
            bigram_probs[word1][word2] = count / unigram_counts[word1] #[cite: 7]

        return list(unigram_counts.keys()), bigram_probs #[cite: 7]

    def generate_text(self, start_word: str, length: int):
        """Generate text based on bigram probabilities.""" #[cite: 7]
        current_word = start_word.lower() #[cite: 7]
        generated_words = [current_word] #[cite: 7]

        for _ in range(length - 1): #[cite: 7]
            next_words = self.bigram_probs.get(current_word) #[cite: 7]
            if not next_words:  # If no bigrams for the current word, stop generating #[cite: 7]
                break #[cite: 7]

            # Choose the next word based on probabilities
            next_word = random.choices( #[cite: 7]
                list(next_words.keys()), weights=next_words.values() #[cite: 7]
            )[0] #[cite: 7]
            generated_words.append(next_word) #[cite: 7]
            current_word = next_word  # Move to the next word #[cite: 7]

        return " ".join(generated_words) #[cite: 7]