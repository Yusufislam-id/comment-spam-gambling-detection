## Data Flow

1. Data Collection: Fetch comments from a YouTube video and use dataset.

2. Preprocessing: Clean and prepare the text for analysis.

3. Detection/Analysis: The core ML/NLP model that classifies the comment.

4. Action/Output: Flag the comment, log it, or send an alert.

## Software to create

1. System to automaticaly deleting comment that detected as betting and gambling promotion

## Key to Improve

- user can send youtube video url that have betting comment so we can add new data from it
- user can login on youtube and give access so can automate delete comment
- create data flow to automate data collection, preprocessing

## Development Checklist

### Data Collection

- [ ] Implement functionality to fetch comments from a YouTube video.
- [ ] Allow users to submit YouTube video URLs for data collection.
- [ ] Automate data collection and storage in the dataset directory.

### Preprocessing

- [ ] Clean and preprocess text data (e.g., remove special characters, stopwords).
- [ ] Tokenize and prepare text for analysis.
- [ ] Ensure preprocessing pipeline is modular and reusable.

### Detection/Analysis

- [ ] Develop or integrate an ML/NLP model for gambling comment detection.
- [ ] Train the model using Kaggle and Huggingface datasets.
- [ ] Evaluate the model's accuracy, precision, recall, and F1 score.

### Action/Output

- [ ] Implement functionality to flag detected comments.
- [ ] Log flagged comments for review.
- [ ] Automate the deletion of flagged comments (requires YouTube API integration).

### User Features

- [ ] Allow users to log in with their YouTube account.
- [ ] Request necessary permissions to manage comments via the YouTube API.
- [ ] Provide a user-friendly interface for submitting video URLs and managing flagged comments.

### Automation

- [ ] Automate the entire data flow:
  - Data collection
  - Preprocessing
  - Detection/Analysis
  - Action/Output
- [ ] Schedule periodic updates to the dataset and model retraining.

### Documentation

- [ ] Document the setup process for the project.
- [ ] Provide usage instructions for all features.
- [ ] Include troubleshooting steps for common issues.

### Testing

- [ ] Write unit tests for all major functions.
- [ ] Test the system with real-world YouTube videos.
- [ ] Ensure the system handles edge cases (e.g., no comments, API errors).

### Deployment

- [ ] Set up the project for deployment (e.g., Docker, cloud services).
- [ ] Ensure the system is scalable and can handle multiple video URLs simultaneously.
- [ ] Monitor and log system performance in production.
