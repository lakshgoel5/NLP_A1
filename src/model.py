import torch
import os
import torch.nn as nn
import numpy as np
import math
import re
from collections import Counter # for counting word frequencies

class Word2Vec(nn.Module):
    def __init__(self, embedding_dim, model_type = 'sg'):
        super().__init__() #My class becomes a torch module
        self.embedding_dim = embedding_dim
        self.window_size = 5
        self.model_type = model_type # cbow or sg
        self.W_in = None
        self.W_out = None
        self.unigram = None

        self.author_tokens = None

        self.lr = 0.01
        self.epochs = 10
        self.window_size = 5

        self.total_words = 0
        self.word2idx = {} # Dict: word -> index
        self.idx2word = {} # Dict: index -> word
        self.vocab_size = 0
        self.vocab_counts = [] # List: index -> frequency

        # DEBUG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def pre_process(self, author_texts: dict[str, str]):
        all_tokens = []
        author_tokens = {}

        for author_id, text in author_texts.items():
            #lowercase
            text = text.lower()
            #remove extra spaces
            text = ' '.join(text.split())
            #remove stop words -> Think
            #lemmatization -> Think

            tokens = re.findall(r'\w+|[^\w\s]', text) #tokenise with punctuations
            #\w+ matches any word character (equal to [a-zA-Z0-9_])
            #[^\w\s] matches any non-word character and non-space (equal to [^a-zA-Z0-9_])
            author_tokens[author_id] = tokens
            all_tokens.extend(tokens)

        return author_tokens, all_tokens

    def build_vocab(self, all_tokens, min_freq=10):
        #Count
        word_counts = Counter(all_tokens)

        if min_freq > 1:
            vocab = {word for word, count in word_counts.items() if count >= min_freq}
        else:
            vocab = set(all_tokens)

        vocab_list = sorted(vocab, key=word_counts.get, reverse=True)

        #Add special tokens
        vocab_list.append("<UNK>") #Unknown
        vocab_list.append("<PAD>") #Padding

        self.word2idx = {word: idx for idx, word in enumerate(vocab_list)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(vocab_list)

        self.total_words = len(all_tokens)

        print(f"Vocabulary built: {self.vocab_size} unique tokens")
        print(f"Total tokens: {self.total_words}")

        return vocab_list

    def init_weights(self):
        self.W_in = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.W_out = nn.Embedding(self.vocab_size, self.embedding_dim)

        init_range = 0.5 / math.sqrt(self.embedding_dim)
        self.W_in.weight.data.uniform_(-init_range, init_range)
        self.W_out.weight.data.uniform_(-init_range, init_range)
        
        # Move everything to GPU/CPU at once
        self.to(self.device)
        
        print(f"Embeddings initialized: {self.vocab_size} x {self.embedding_dim}")
        
        return

    #DEBUG
    #Implement unigram later

    def sigmoid(self, x):
        #DEBUG usually scipy.special.expit is faster.
        if x > 7: return 1
        if x < -7: return 0
        return 1 / (1 + np.exp(-x))


    def thread(self, data_chunk):
        #Read some chunk from file
            #While reading, count frequency of words
            #If a word too frequent, can dicard it

        #Sliding window


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

    # def forward(self, target_list, context_list, mode):
    #     loss = 0
    #     if mode == 'cbow':
    #             # Get context embedding of each word in context list
            
    #         pass

    #     else:
    #         #Get Target word embedding

    #         #Get all W_out weights

    #         #Matrix multiply to get score of each pair using torch

    #         #Find log probabilities(softmax)

    #         # Equivalent to: loss = -log(P(context_word | target_word))
    #         # Cross entropy loss
    #         pass

    #     return loss
            
    #     #Optimization (first set a baseline)
    #         #You may try updating vectors without lock, as probability of collision is very low, speed may matter over noise
    #         #Can try using random window size in context

    def forward(self, target_list, context_list, mode):
        # target_list: 
        #   CBOW: Tensor of [batch_size] (Indices of center words)
        #   SG:   Tensor of [batch_size] (Indices of center words)
        
        # context_list:
        #   CBOW: Tensor of [batch_size, window_size] (Indices of context words, Padded)
        #   SG:   Tensor of [batch_size] (Indices of single context words)

        # 1. Define Padding Index (needed for CBOW masking)
        pad_idx = self.word2idx.get("<PAD>", -1)

        if mode == 'cbow':
            # --- CBOW: Predict Center (target) from Context (context_list) ---
            
            # A. Get embeddings for all context words
            # Shape: (Batch_Size, Max_Window, Embedding_Dim)
            context_embeds = self.W_in(context_list)

            # B. Handle Padding (We must not average the <PAD> tokens)
            # Create a mask: 1 for real words, 0 for <PAD>
            # Shape: (Batch_Size, Max_Window, 1)
            mask = (context_list != pad_idx).unsqueeze(-1).float()
            
            # Zero out the embeddings of padding tokens
            masked_embeds = context_embeds * mask
            
            # Sum the embeddings along the window dimension
            sum_embeds = masked_embeds.sum(dim=1)
            
            # Count the number of real words in each window to get the mean
            counts = mask.sum(dim=1)
            counts[counts == 0] = 1 # Avoid division by zero
            
            # Calculate the average vector "h"
            # Shape: (Batch_Size, Embedding_Dim)
            h = sum_embeds / counts

            # C. Score against the entire vocabulary (Full Softmax)
            # We treat W_out as a linear layer weight matrix.
            # h @ W_out.T
            # (Batch, Emb) @ (Emb, Vocab) -> (Batch, Vocab)
            scores = torch.matmul(h, self.W_out.weight.t())

            # D. Calculate Loss
            # We want to maximize prob of the 'target_list' (the center words)
            # NLLLoss(LogSoftmax(scores), target) is standard CrossEntropy
            log_probs = torch.nn.functional.log_softmax(scores, dim=1)
            loss = torch.nn.functional.nll_loss(log_probs, target_list)

        else:
            # --- Skip-Gram: Predict Context (context_list) from Center (target_list) ---
            
            # A. Get embedding for the center word
            # Shape: (Batch_Size, Embedding_Dim)
            h = self.W_in(target_list)

            # B. Score against the entire vocabulary
            # We calculate how well 'h' predicts *every* word in the vocab.
            # Shape: (Batch_Size, Vocab_Size)
            scores = torch.matmul(h, self.W_out.weight.t())

            # C. Calculate Loss
            # In SG, we want to maximize the probability of the *true context word*.
            # 'context_list' here contains the indices of the true context words.
            log_probs = torch.nn.functional.log_softmax(scores, dim=1)
            loss = torch.nn.functional.nll_loss(log_probs, context_list)

        return loss

    def train_cbow(self):
        initial_lr = self.lr
        min_lr = 0.0001

        target_list = []
        context_lists = []

        #Create target and context list pairs
        for author_id, tokens in self.author_tokens.items():
            # Convert tokens to indices
            indices = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]

            if len(indices) < self.window_size - 1:
                continue
            
            for i in range(len(indices)):
                target_word = indices[i]
                
                # Define context window
                window_start = max(0, i - self.window_size)
                window_end = min(len(indices), i + self.window_size + 1)

                context = [indices[j] for j in range(window_start, window_end) if i != j]
                
                #(target, context list)
                if(len(context) > 0):
                    target_list.append(target_word)
                    context_lists.append(context)

        #Train
        for epoch in range(self.epochs):
            epoch_loss = 0
            batch_count = 0

            current_lr = max(min_lr, initial_lr * (1 - epoch / self.epochs))

            batch_size = 256
            num_batches = (len(target_list) + batch_size - 1) // batch_size

            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(target_list))

                #Convert to tensors
                batch_target = torch.LongTensor(target_list[start:end]).to(self.device)
                # using LongTensor gurantees that indices of embedding are intigers
                batch_contexts_raw = context_lists[start:end]
                # context lists might have different sizes, so cant be made a tensor
                max_context_size = max(len(context) for context in batch_contexts_raw)
                pad_idx = self.word2idx["<PAD>"]
                padded = [c + [pad_idx] * (max_context_size - len(c)) for c in batch_contexts_raw]
                batch_contexts = torch.LongTensor(padded).to(self.device)

                loss = self.forward(batch_target, batch_contexts, 'cbow')
                epoch_loss += loss.item()
                batch_count += 1

                if self.W_in.weight.grad is not None:
                    self.W_in.weight.grad.zero_()
                if self.W_out.weight.grad is not None:
                    self.W_out.weight.grad.zero_()

                loss.backward()
                
                with torch.no_grad():
                    # I am just updating numbers now, don't try to calculate gradients of this assignment
                    self.W_in.weight -= current_lr * self.W_in.weight.grad
                    self.W_out.weight -= current_lr * self.W_out.weight.grad
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/batch_count:.4f}")

            # Check similarity every two epochs
            if epoch % 2 == 0:
                # self.check_similarity() #DEBUG
                pass

    def train_skipgram(self):
        #Stochastic gradient descent
        #self.parameters() finds all nn.Embedding layers defined in init

        # Not allowed
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

        initial_lr = self.lr
        min_lr = 0.0001

        # Create target and context pairs
        target_list = []
        context_list = []

        for author_id, tokens in self.author_tokens.items():
            # Convert tokens to indices
            indices = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]

            if len(indices) < self.window_size - 1:
                continue

            
            for i in range(len(indices)):
                target_word = indices[i]
                
                # Define context window
                window_start = max(0, i - self.window_size)
                window_end = min(len(indices), i + self.window_size + 1)

                for j in range(window_start, window_end):
                    if i == j:
                        continue
                    
                    #pushed as pairs
                    context_word = indices[j]
                    target_list.append(target_word)
                    context_list.append(context_word)

        #Train
        for epoch in range(self.epochs):
            epoch_loss = 0
            batch_count = 0

            current_lr = max(min_lr, initial_lr * (1 - epoch / self.epochs))

            # Process in batches
            # as GPU has limited memory
            batch_size = 256
            num_batches = (len(target_list) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, len(target_list))
                
                # Convert python list to torch tensor
                # Move to GPU
                batch_targets = torch.LongTensor(target_list[start:end]).to(self.device)
                batch_contexts = torch.LongTensor(context_list[start:end]).to(self.device)

                # Loss function (forward pass)
                loss = self.forward(batch_targets, batch_contexts, 'sg')
                epoch_loss += loss.item()
                batch_count += 1

                # Gradient management (Pytorch accumulates gradients)
                if self.W_in.weight.grad is not None:
                    self.W_in.weight.grad.zero_()
                if self.W_out.weight.grad is not None:
                    self.W_out.weight.grad.zero_()

                # Backpropagation (backward pass)
                loss.backward()
                # Update weights
                with torch.no_grad():
                    self.W_in.weight -= current_lr * self.W_in.weight.grad
                    self.W_out.weight -= current_lr * self.W_out.weight.grad

            # Print Epoch Stats
            avg_loss = epoch_loss / max(1, batch_count)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.4f}")
            
            # Check similarity every two epochs
            if epoch % 2 == 0:
                # self.check_similarity() #DEBUG
                pass

    def train(self, author_texts):
        self.author_tokens, all_tokens = self.pre_process(author_texts)
        #prepration
            #Read vocab
            #Count word frequencies
            #Discard rare words
        vocab = self.build_vocab(all_tokens)
            #Create embedding matrix W
            #Create output matrix U
        self.init_weights()

        #train using multithreading
            #Do forward pass
            #Do backpropagation

        if self.model_type == 'cbow':
            self.train_cbow()
        else:
            self.train_skipgram()
        #save embeddings
            #As raw vectors i.e. W
            #As clusters after k-means

        return self.W_in.weight.data.cpu().numpy()

    def save_embeddings(self, save_dir='./models'):
        import pickle #DEBUG

        os.makedirs(save_dir, exist_ok=True)

        #Save embeddings as numpy
        embeddings = self.W_in.weight.data.cpu().numpy()
        np.save(f"{save_dir}/embeddings.npy", embeddings)

        #Save vocab
        with open(f"{save_dir}/word2idx.pkl", 'wb') as f:
            pickle.dump(self.word2idx, f)
        
        with open(f"{save_dir}/idx2word.pkl", 'wb') as f:
            pickle.dump(self.idx2word, f)

        torch.save({
            'W_in': self.W_in.state_dict(),
            'W_out': self.W_out.state_dict(),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'model_type': self.model_type,
        }, f"{save_dir}/model.pt")


def load_data(train_dir):
    author_texts = {}

    # Get all .txt files in the directory using os.listdir
    files = [f for f in os.listdir(train_dir) if f.endswith('.txt')]

    for filename in files:
        file_path = os.path.join(train_dir, filename)
        author_id = filename.split(".")[0]

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if author_id not in author_texts:
            author_texts[author_id] = text
    return author_texts

def main():

    #arguments


    train_dir = "../data/train_data"
    author_texts = load_data(train_dir)

    model = Word2Vec(100, "cbow")

    embeddings = model.train(author_texts)
    model.save_embeddings('./models')


if __name__ == "__main__":
    main()