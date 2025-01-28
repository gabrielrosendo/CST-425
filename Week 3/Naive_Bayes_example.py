import math
from collections import defaultdict

# Step 1: Create a small dataset
dataset = [
	{"text": "free money now", "label": "Spam"},
	{"text": "urgent meeting today", "label": "Not Spam"},
	{"text": "win lottery money", "label": "Spam"},
	{"text": "project deadline tomorrow", "label": "Not Spam"},
	{"text": "free lottery ticket", "label": "Spam"},
]


# Step 2: Preprocess and tokenize the dataset
def tokenize(text):
	return text.lower().split()


# Step 3: Train the Naive Bayes Classifier
def train_naive_bayes(dataset):
	word_counts = {"Spam": defaultdict(int), "Not Spam": defaultdict(int)}
	class_counts = {"Spam": 0, "Not Spam": 0}
	total_words = {"Spam": 0, "Not Spam": 0}

	for data in dataset:
		label = data["label"]
		class_counts[label] += 1
		for word in tokenize(data["text"]):
			word_counts[label][word] += 1
			total_words[label] += 1

	return word_counts, class_counts, total_words


# Step 4: Calculate probabilities
def calculate_class_probabilities(word_counts, class_counts, total_words, new_text):
	probs = {}
	total_emails = sum(class_counts.values())
	for label in class_counts:
		# Prior probability
		probs[label] = math.log(class_counts[label] / total_emails)
		for word in tokenize(new_text):
			word_freq = word_counts[label][word]
			# Add Laplace smoothing
			word_prob = (word_freq + 1) / (total_words[label] + len(word_counts[label]))
			probs[label] += math.log(word_prob)
	return probs


# Step 5: Make a prediction
def predict(word_counts, class_counts, total_words, new_text):
	probs = calculate_class_probabilities(word_counts, class_counts, total_words, new_text)
	return max(probs, key=probs.get)


# Train the classifier
word_counts, class_counts, total_words = train_naive_bayes(dataset)

# Step 6: Test the classifier
new_email = "free lottery"
prediction = predict(word_counts, class_counts, total_words, new_email)
print(f"The email '{new_email}' is classified as: {prediction}")

# Additional Test Cases
test_emails = ["urgent meeting", "win free money", "project tomorrow"]
for email in test_emails:
	print(f"The email '{email}' is classified as: {predict(word_counts, class_counts, total_words, email)}")
