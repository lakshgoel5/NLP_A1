import torch
import os
import torch.nn as nn
import numpy as np
import math
import re
from collections import Counter # for counting word frequencies
import time
from tqdm import tqdm

class Word2Vec(nn.Module):
    def __init__(self, embedding_dim, model_type = 'sg'):
        super().__init__() #My class becomes a torch module
        self.embedding_dim = embedding_dim
        self.window_size = 5
        self.model_type = model_type # cbow or sg
        self.W_in = None
        self.W_out = None

        self.author_tokens = None

        self.lr = 0.01
        self.epochs = 10
        self.window_size = 5

        self.total_words = 0
        self.word2idx = {} # Dict: word -> index
        self.idx2word = {} # Dict: index -> word
        self.vocab_size = 0
        self.vocab_counts = [] # List: index -> frequency

        self.model_speed = "ns"
        self.num_negatives = 5
        self.unigram_dist = None

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
        print("\n--- [Mark] Starting Vocabulary Build ---") #DEBUG
        start = time.time() #DEBUG
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

        #Store frequencies for negative sampling
        self.vocab_counts = torch.zeros(self.vocab_size)
        for word, count in word_counts.items():
            if word in self.word2idx:
                idx = self.word2idx[word]
                self.vocab_counts[idx] = count
        
        #Unigram distribution
        # Power 0.75 for smoothing the distribution
        self.unigram_dist = self.vocab_counts.pow(0.75)
        self.unigram_dist /= self.unigram_dist.sum()
        
        self.total_words = len(all_tokens)

        print(f"Vocabulary built: {self.vocab_size} unique tokens")
        print(f"Total tokens: {self.total_words}")
        print(f"Vocab build took: {time.time() - start:.2f}s\n") #DEBUG

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

    def sigmoid(self, x):
        #DEBUG usually scipy.special.expit is faster.
        if x > 7: return 1
        if x < -7: return 0
        return 1 / (1 + np.exp(-x))

    #test function written by AI
    def check_similarity(self, top_k=5):
        """
        Check similarity of some words to see if the model is learning meaningful embeddings.
        """
        if self.W_in is None:
            return

        self.eval() # Set to evaluation mode
        with torch.no_grad():
            # Pick a few indices to test (frequent words are usually at low indices)
            # We avoid the very first few (0-9) as they are often just stop words like 'the', 'a'
            # We pick some slightly deeper in the vocab
            test_indices = [15, 30, 60, 100]
            # Ensure indices are within vocab range
            test_indices = [idx for idx in test_indices if idx < self.vocab_size]
            
            if not test_indices:
                return
                
            test_tensor = torch.LongTensor(test_indices).to(self.device)
            
            # 1. Normalize all embeddings so that dot product = cosine similarity
            # (Vocab_Size, Embedding_Dim)
            all_weights = self.W_in.weight
            norm = all_weights.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings = all_weights / (norm + 1e-9) # Avoid div by zero
            
            # 2. Get embeddings for our test words
            test_embeds = normalized_embeddings[test_tensor] # (Num_Test, Embedding_Dim)
            
            # 3. Calculate similarity matrix
            # (Num_Test, Embedding_Dim) @ (Embedding_Dim, Vocab_Size) -> (Num_Test, Vocab_Size)
            similarities = torch.matmul(test_embeds, normalized_embeddings.t())
            
            print("\n--- Similarity Check ---")
            for i, idx in enumerate(test_indices):
                test_word = self.idx2word.get(idx, "UNK")
                
                # Get the top K+1 similar indices (the word itself will be #1 with similarity 1.0)
                top_values, top_indices = torch.topk(similarities[i], top_k + 1)
                
                # Filter out the word itself and get the strings
                similar_words = []
                for j in range(len(top_indices)):
                    neighbor_idx = top_indices[j].item()
                    if neighbor_idx != idx:
                        similar_words.append(self.idx2word.get(neighbor_idx, "UNK"))
                
                print(f"'{test_word}': {', '.join(similar_words[:top_k])}")
            print("------------------------\n")
            
        self.train() # Set back to training mode

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

    def forward(self, target_list, context_list):
        # context_list:
        #   CBOW: Tensor of [batch_size, window_size] (Indices of context words, Padded)
        #   SG:   Tensor of [batch_size] (Indices of single context words)

        pad_idx = self.word2idx.get("<PAD>", -1)

        if self.model_type == 'cbow':
            #Get context word embeddings
            #Dimension of context_embeds: Batch_Size x Window_Size x Embedding_Dim
            #Dimension of context_list: Batch_Size x Window_Size
            #They both are tensors
            context_embeds = self.W_in(context_list)

            # We must not average the <PAD> tokens
            # 1 for real words, 0 for <PAD>
            # Dimension of mask: Batch_Size x Window_Size x 1
            # context_list != pad_idx: Boolean Tensor
            # unsqueeze(-1) to add a dimension as we multiply with context_embeds
            mask = (context_list != pad_idx).unsqueeze(-1).float()
            
            # Zero out the embeddings of padding tokens
            # Dimension of masked_embeds: Batch_Size x Window_Size x Embedding_Dim
            masked_embeds = context_embeds * mask
            
            # Sum the embeddings along the window dimension i.e. sum over all context words of target word
            # Dimension of sum_embeds: Batch_Size x Embedding_Dim
            sum_embeds = masked_embeds.sum(dim=1)
            
            # Count the number of real words in each window to get the mean
            # Dimension of counts: Batch_Size x 1
            counts = mask.sum(dim=1)

            # BElow loop very slow for Pytorch
            # for i in range(len(counts)):
            #     if counts[i] == 0:
            #         counts[i] = 1

            # Vectorized version
            counts = counts.clamp(min=1)
            # Replace all 0s with 1 in one operation
            
            # Calculate the average vector "h"
            # Dimension of h: Batch_Size x Embedding_Dim
            h = sum_embeds / counts
            # Tensor extends dimensions by itself

            # Full Softmax
            # We treat W_out as a linear layer weight matrix.
            # h @ W_out.T
            # (Batch, Emb) @ (Emb, Vocab) -> (Batch, Vocab)
            scores = torch.matmul(h, self.W_out.weight.t())
            #Every row has probabilities of all words and word with highest probability is the predicted word

            # We want to maximize prob of the 'target_list' (the center words)
            log_probs = torch.nn.functional.log_softmax(scores, dim=1) # Log probabilities. #Dim=1 as we want to find probabilities across vocab
            #log for stability

            #NLL: Negative log likelihood
            loss = torch.nn.functional.nll_loss(log_probs, target_list)

        else:
            
            # Get embedding for the center word
            h = self.W_in(target_list)

            # Score against the entire vocabulary
            # We calculate how well 'h' predicts "every" word in the vocab.
            # Dimension of scores: (Batch_Size, Vocab_Size)
            scores = torch.matmul(h, self.W_out.weight.t())

            # Calculate Loss
            # In SG, we want to maximize the probability of the *true context word*.
            # 'context_list' here contains the indices of the true context words.
            log_probs = torch.nn.functional.log_softmax(scores, dim=1)
            loss = torch.nn.functional.nll_loss(log_probs, context_list)

        return loss

    def forward_ns(self, target_list, context_list, negative_samples):
        pad_idx = self.word2idx.get("<PAD>", -1)

        if self.model_type == 'cbow':
            context_embeds = self.W_in(context_list)
            mask = (context_list != pad_idx).unsqueeze(-1).float()
            masked_embeds = context_embeds * mask
            sum_embeds = masked_embeds.sum(dim=1)
            counts = mask.sum(dim=1)
            counts = counts.clamp(min=1)
            h = sum_embeds / counts

            # -- Positive scores --
            positive_out = self.W_out(target_list)
            # h is average of context words embeddings
            # * multiplies them element wise
            # h: (Batch, Emb), positive_out: (Batch, Emb)
            # sum along dim=1 as product followed by sum is what happens in dot product
            # Dot product along the embedding dimension
            Positive_score = torch.sum(h * positive_out, dim=1)
            # Dimension of Positive_score: (Batch,)

            # -- Negative scores --
            # negative_samples: (Batch, num_negative_samples)
            # Get embeddings for negative samples
            negative_out = self.W_out(negative_samples)
            # h: (Batch, Emb), negative_out: (Batch, num_negative_samples, Emb)
            # We need to compute dot product for each negative sample
            # h.unsqueeze(1): (Batch, 1, Emb)
            # (Batch, 1, Emb) * (Batch, num_negative_samples, Emb) -> (Batch, num_negative_samples, Emb)
            # Sum along dim=2 to get dot products
            # Negative_score = torch.sum(h.unsqueeze(1) * negative_out, dim=2)
            # Dimension of Negative_score: (Batch, num_negative_samples)

            #Efficient way
            negative_score = torch.bmm(negative_out, h.unsqueeze(-1)).squeeze(-1) # (Batch, num_negative_samples)


            # Calculate loss
            # log(sigmoid(Positive_score)) - log(sigmoid(Negative_score))
            # We want to maximize log(sigmoid(Positive_score)) + log(1 - sigmoid(Negative_score))
            loss = -torch.nn.functional.logsigmoid(Positive_score) - torch.sum(torch.nn.functional.logsigmoid(-negative_score), dim=1)
            loss = loss.mean()
            return loss
        else:
            h = self.W_in(target_list)

            positive_out = self.W_out(context_list)
            positive_score = torch.sum(h * positive_out, dim=1)

            negative_out = self.W_out(negative_samples)
            #h.unsqueeze(-1): (Batch, Emb, 1)
            #negative_out: (Batch, num_negative_samples, Emb)
            #(num_negative_samples, Emb) @ (Emb, 1) -> (num_negative_samples, 1)
            #negative_score: (Batch, num_negative_samples)
            negative_score = torch.bmm(negative_out, h.unsqueeze(-1)).squeeze(-1)


            loss = -torch.nn.functional.logsigmoid(positive_score) - torch.sum(torch.nn.functional.logsigmoid(-negative_score), dim=1)
            loss = loss.mean()
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
        print(f"Starting CBOW Training on {self.device}...") #DEBUG
        for epoch in range(self.epochs):
            start_time = time.time() #DEBUG
            epoch_loss = 0
            batch_count = 0

            current_lr = max(min_lr, initial_lr * (1 - epoch / self.epochs))

            batch_size = 256
            num_batches = (len(target_list) + batch_size - 1) // batch_size
            
            # Progress Bar
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.epochs}")

            for i in pbar:
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

                loss = 0
                if(self.model_speed == "softmax"):
                    loss = self.forward(batch_target, batch_contexts)
                else:
                    current_batch_size = end - start
                    # This picks 5 random words for every single target word in the batch. By doing this inside the batch, we ensure that the model sees different "negative" examples in every epoch, which is key for learning what a word is not.
                    batch_neg = torch.multinomial(self.unigram_dist, current_batch_size * self.num_negatives, replacement=True)
                    # .view reshapes 1D vector to 2D matrix of size (batch_size, num_negatives)
                    batch_neg = batch_neg.view(current_batch_size, self.num_negatives).to(self.device)

                    loss = self.forward_ns(batch_target, batch_contexts, batch_neg)
                
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
                    # TODO: Use adam optimiser or other optimiser if allowed
                
                # Update progress bar with loss
                pbar.set_postfix({'loss': loss.item()})
            
            elapsed = time.time() - start_time #DEBUG
            print(f"Finished Epoch {epoch+1}, Loss: {epoch_loss/batch_count:.4f}, LR: {current_lr:.4f}, Time: {elapsed:.2f}s") #DEBUG 

            # Check similarity every two epochs
            if (epoch+1) % 10 == 0:
                self.check_similarity() #DEBUG
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
        print(f"Starting Skip-Gram Training on {self.device}...") #DEBUG
        for epoch in range(self.epochs):
            start_time = time.time() #DEBUG
            epoch_loss = 0
            batch_count = 0

            current_lr = max(min_lr, initial_lr * (1 - epoch / self.epochs))

            # Process in batches
            # as GPU has limited memory
            batch_size = 256
            num_batches = (len(target_list) + batch_size - 1) // batch_size

            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.epochs}")

            for batch_idx in pbar:
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, len(target_list))
                
                # Convert python list to torch tensor
                # Move to GPU
                batch_targets = torch.LongTensor(target_list[start:end]).to(self.device)
                batch_contexts = torch.LongTensor(context_list[start:end]).to(self.device)

                # Loss function (forward pass)
                if(self.model_speed == "softmax"):
                    loss = self.forward(batch_targets, batch_contexts)
                else:
                    current_batch_size = end - start
                    batch_neg = torch.multinomial(self.unigram_dist, current_batch_size * self.num_negatives, replacement=True)
                    batch_neg = batch_neg.view(current_batch_size, self.num_negatives).to(self.device)

                    loss = self.forward_ns(batch_targets, batch_contexts, batch_neg)
                
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
                
                pbar.set_postfix({'loss': loss.item()})

            # Print Epoch Stats
            elapsed = time.time() - start_time #DEBUG
            avg_loss = epoch_loss / max(1, batch_count)
            print(f"Finished Epoch {epoch+1}, Loss: {avg_loss:.4f}, LR: {current_lr:.4f}, Time: {elapsed:.2f}s") #DEBUG
            
            # Check similarity every two epochs
            if (epoch+1) % 10 == 0:
                self.check_similarity() #DEBUG
                pass

    def train_model(self, author_texts):
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

    model = Word2Vec(100, "sg")

    embeddings = model.train_model(author_texts)
    model.save_embeddings('./models')


if __name__ == "__main__":
    main()