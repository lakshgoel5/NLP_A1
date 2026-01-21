import numpy as np
import math
from collections import Counter # for counting word frequencies

class model:
    def __init__(self, embedding_dim, model_type):
        self.embedding_dim = embedding_dim
        self.window_size = 5
        self.model_type = model_type # cbow or sg
        self.W = None
        self.U = None
        self.unigram = None

        self.total_words = 0
        self.vocab = {} # Dict: word -> index
        self.idx2word = [] # List: index -> word
        self.vocab_counts = [] # List: index -> frequency

    def build_vocab(self, file_path):
        # word_counts = Counter() #DEBUG
        with open(file_path, 'r') as f:
            for line in f:
                words_list = line.split()
                #DEBUG
                # word_counts.update(words_list) #.update updates counts of words in words_list to word_counts
                for word in words_list:
                    if word not in self.vocab:
                        self.vocab[word] = len(self.idx2word)
                        self.idx2word.append(word)
                        self.vocab_counts.append(0)

                    idx = self.vocab[word]
                    self.vocab_counts[idx] += 1

        self.total_words = sum(self.vocab_counts)

        #DEBUG
        #bring words of high frequency at starting of idx2word and vocab_counts using counter() later

    def init_weights(self):
        self.W = (np.random.randn(self.total_words, self.embedding_dim) - 0.5) / self.embedding_dim
        self.U = np.zeros(self.total_words, self.embedding_dim)

    #DEBUG
    #Implement unigram later

    def sigmoid(self, x):
        #DEBUG usually scipy.special.expit is faster.
        if x > 7: return 1
        if x < -7: return 0
        return 1 / (1 + math.exp(-x))


    def thread(self, data_chunk):
        #Read some chunk from file
            #While reading, count frequency of words
            #If a word too frequent, can dicard it

        #Sliding window
        
        for line in data_chunk:
            words_list = line.split()
            for word in words_list:
                if word not in self.vocab:
                    self.vocab[word] = len(self.idx2word)
                    self.idx2word.append(word)
                    self.vocab_counts.append(0)

                idx = self.vocab[word]
                self.vocab_counts[idx] += 1

        #If cbow
            # Sum up context into hidden layer
            # Pass hidden layer to output layer
            # Softmax
                #Some loss function
                #Backpropagation to update weights
            # or Negative sampling

        #else sg
            # Center word -> context words
            # Iterate every word in context window
            # For each word, find a center word
            # Negative sampling
                # Accumulate loss
                # Backpropagation to update weights

        #Optimization (first set a baseline)
            #You may try updating vectors without lock, as probability of collision is very low, speed may matter over noise
            #Can try using random window size in context
        pass

    def train(self, file_path):
        #prepration
            #Read vocab
            #Count word frequencies
            #Discard rare words
        self.build_vocab(file_path)
            #Create embedding matrix W
            #Create output matrix U
        self.init_weights()

        #train using multithreading
            #Do forward pass
            #Do backpropagation

        #save embeddings
            #As raw vectors i.e. W
            #As clusters after k-means
        pass


def main():
    #arguments

    #pre-processing
        #sigmoid calculation
    
    #training
        
    pass

if __name__ == "__main__":
    main()