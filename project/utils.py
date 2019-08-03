import nltk
import pickle
import re
import numpy as np
import gensim

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype
    line_count = sum(1 for line in open(embeddings_path))
    with open(embeddings_path, 'r') as inp, open('data/word2vec-format.txt', 'w') as outp:
        line_count = str(line_count)    # line count of the tsv file (as string)
        for line in inp:
            words = line.strip().split()
            dimensions = str(len(words)-1)    # vector size (as string)
            break
        print(dimensions)
        outp.write(' '.join([line_count, dimensions]) + '\n')
        for line in inp:
            words = line.strip().split()
            outp.write(' '.join(words) + '\n')
    starspace_embeddings = gensim.models.KeyedVectors.load_word2vec_format('data/word2vec-format.txt',binary=True)
    return (starspace_embeddings, int(dimensions))

def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    # Hint: you have already implemented exactly this function in the 3rd assignment.

    # remove this when you're done
    vec = np.zeros(dim)
    counter = 0
    if not question:
        return np.zeros(dim)
    
    for i in question.split():
        if i in embeddings:
            vec += embeddings[i]
            counter += 1
    if not np.any(vec):
        return np.zeros(dim)
    return vec/counter
   
def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
