
def thread():
    #Read some chunk from file
        #While reading, count frequency of words
        #If a word too frequent, can dicard it

    #Sliding window
        #

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

def train():
    #prepration
        #Read vocab
        #Count word frequencies
        #Discard rare words
        #Create embedding matrix W
        #Create output matrix U

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