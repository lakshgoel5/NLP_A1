import torch
import os
import torch.nn as nn
import numpy as np
import math
import re
from collections import Counter # for counting word frequencies
import time
from multiprocessing import Pool

class Word2Vec(nn.Module):
    def __init__(self, embedding_dim = 100):
        super().__init__() #My class becomes a torch module
        self.embedding_dim = embedding_dim
        self.W_in = None
        self.W_out = None

        self.author_tokens = None

        self.lr = 0.025
        self.epochs = 40
        self.window_size = 3

        self.total_words = 0
        self.word2idx = {} # Dict: word -> index
        self.idx2word = {} # Dict: index -> word
        self.vocab_size = 0
        self.vocab_counts = [] # List: index -> frequency

        self.num_negatives = 5
        self.unigram_dist = None

        self.batch_size = 512
        
        # Subsampling threshold for frequent words
        self.subsample_threshold = 0
        self.stop_words = False

        self.max_training_time = 27*60
        
        # self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        print(f"Using device: {self.device}")

    def pre_process(self, author_texts: dict[str, str]):

        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                  'to', 'for', 'of', 'is', 'was', 'are', 'were', 'been', 
                  'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                  'would', 'should', 'could', 'may', 'might', 'must', 'can',
                  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that',
                  'these', 'those', 's', 't'}

        all_tokens = []
        author_tokens = {}

        for author_id, text in author_texts.items():
            #lowercase
            text = text.lower()
            #remove extra spaces
            # text = ' '.join(text.split())

            text = re.sub(r'\$\d+\.\d+', '', text)  # Remove prices
            text = re.sub(r'\d+-\d+\s+broadway', '', text)  # Remove addresses

            tokens = re.findall(r'\w+|[^\w\s]', text) #tokenise with punctuations
            #\w+ matches any word character (equal to [a-zA-Z0-9_])
            #[^\w\s] matches any non-word character and non-space (equal to [^a-zA-Z0-9_])

            tokens = [re.sub(r"([!?.,;:])\1+", r"\1", t) for t in tokens]
            tokens = ["<NUM>" if t.isdigit() else t for t in tokens]

            if self.stop_words == True:
                tokens = [token for token in tokens if token not in stop_words]

            author_tokens[author_id] = tokens
            all_tokens.extend(tokens)

        return author_tokens, all_tokens

    def build_vocab(self, all_tokens, min_freq=5):
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

        return

    # def subsample_prob(self, word_idx):
    #     """
    #     Calculate probability of keeping a word based on its frequency.
    #     Formula from Word2Vec paper:
    #     P(keep) = (sqrt(word_freq / (threshold * total_words)) + 1) * (threshold * total_words) / word_freq
        
    #     Frequent words are discarded with higher probability.
    #     """
    #     if self.subsample_threshold <= 0:
    #         return 1.0  # No subsampling
        
    #     word_freq = self.vocab_counts[word_idx].item()
        
    #     # Handle zero or very low frequency (<UNK>, <PAD> tokens)
    #     if word_freq == 0:
    #         return 1.0  # Always keep words with zero frequency
        
    #     freq_ratio = word_freq / (self.subsample_threshold * self.total_words)
        
    #     # If freq_ratio is very small, just keep the word
    #     if freq_ratio < 1e-10:
    #         return 1.0
        
    #     # Calculate keep probability
    #     keep_prob = (math.sqrt(freq_ratio) + 1) / freq_ratio
        
    #     return min(keep_prob, 1.0)  # Clamp to [0, 1]


    def subsample_prob(self, word_idx):
        if self.subsample_threshold <= 0:
            return 1.0

        count = self.vocab_counts[word_idx].item()
        if count <= 0:
            return 1.0

        f = count / self.total_words
        t = self.subsample_threshold

        keep_prob = math.sqrt(t / f)
        return min(1.0, keep_prob)
        
    def get_distance_weight(self, distance):
        """
        Calculate weight based on distance from target word.
        Implements Word-Space Model (Sahlgren, 2006) weighting schemes.
        
        Args:
            distance: Absolute distance from target word (1, 2, 3, ...)
        
        Returns:
            Weight value (higher for closer words)
        """
        if not self.use_distance_weighting or distance == 0:
            return 1.0
        
        if self.weighting_scheme == "aggressive":
            # Aggressive: 1/(2^(distance-1))
            # distance=1 -> 1/1, distance=2 -> 1/2, distance=3 -> 1/4, distance=4 -> 1/8
            return 1.0 / (2 ** (distance - 1))
        elif self.weighting_scheme == "glove":
            # GloVe-style: 1/distance
            # distance=1 -> 1/1, distance=2 -> 1/2, distance=3 -> 1/3, distance=4 -> 1/4
            return 1.0 / distance
        else:
            return 1.0  # Uniform weighting

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

    def forward(self, input_data):
        h = self.W_in(input_data)  # [batch, emb]

        scores = torch.matmul(h, self.W_out.weight.t())  # [batch, vocab_size]
        return scores

    def loss_function(self, target_list, context_list, negative_samples):
        h = self.W_in(target_list)
        
        positive_out = self.W_out(context_list)
        positive_score = torch.sum(h * positive_out, dim=1)

        negative_out = self.W_out(negative_samples)
        negative_score = torch.bmm(negative_out, h.unsqueeze(-1)).squeeze(-1)

        loss = -torch.nn.functional.logsigmoid(positive_score).mean()
        loss -= torch.nn.functional.logsigmoid(-negative_score).sum(dim=1).mean()
        
        return loss

    def _generate_sg_pairs(self, author_id, tokens):
        # Create target and context pairs with distance weights and document IDs
        target_list = []
        context_list = []

        # Convert tokens to indices
        indices = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        
        if len(indices) < self.window_size - 1:
            return target_list, context_list, weight_list

        
        for i in range(len(indices)):
            target_word = indices[i]
            
            # Subsampling: randomly discard frequent words
            keep_prob = self.subsample_prob(target_word)
            if np.random.random() > keep_prob:
                continue
            
            # Define context window
            dynamic_window = np.random.randint(1, self.window_size + 1)
            window_start = max(0, i - dynamic_window)
            window_end = min(len(indices), i + dynamic_window + 1)

            for j in range(window_start, window_end):
                if i == j:
                    continue
                
                #pushed as pairs with weights and doc_id
                context_word = indices[j]
                target_list.append(target_word)
                context_list.append(context_word)

        return target_list, context_list

    def train_skipgram(self):
        # Initialize Adam Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # num_workers = min(3, os.cpu_count() - 1) #DEBUG

        target_list = []
        context_list = []

        # with Pool(processes=num_workers) as pool:
        #     results = pool.starmap(self._generate_sg_pairs, [(author_id, tokens) for author_id, tokens in self.author_tokens.items()])

        # for res in results:
        #     target_list.extend(res[0])
        #     context_list.extend(res[1])


        target_list = []
        context_list = []

        for author_id, tokens in self.author_tokens.items():
            # Convert tokens to indices
            indices = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
            
            for i in range(len(indices)):
                target_word = indices[i]
                
                # Subsampling: randomly discard frequent words
                # keep_prob = self.subsample_prob(target_word)
                # if np.random.random() > keep_prob:
                    # continue

                if target_word == self.word2idx["<UNK>"]:
                    continue
                
                # Define context window
                # dynamic_window = np.random.randint(1, self.window_size + 1)
                window_start = max(0, i - self.window_size)
                window_end = min(len(indices), i + self.window_size + 1)

                for j in range(window_start, window_end):
                    if i == j:
                        continue

                    context_word = indices[j]
                    if context_word != self.word2idx["<UNK>"]:
                        target_list.append(target_word)
                        context_list.append(context_word)
        #Train
        print(f"Starting Skip-Gram Training on {self.device}...") #DEBUG
        training_start_time = time.time()
        
        for epoch in range(self.epochs):
            # Check total time before each epoch
            if time.time() - training_start_time > self.max_training_time:
                print(f"\nTime limit of {self.max_training_time/60:.1f}m reached. Stopping training.")
                break
                
            start_time = time.time() #DEBUG
            epoch_loss = 0
            batch_count = 0
            num_batches = (len(target_list) + self.batch_size - 1) // self.batch_size
            time_limit_reached = False
            for batch_idx in range(num_batches):
                # Check time inside batch loop for more precision
                if time.time() - training_start_time > self.max_training_time:
                    print(f"\nTime limit reached during epoch {epoch+1}. Stopping.")
                    time_limit_reached = True
                    break
                    
                start = batch_idx * self.batch_size
                end = min((batch_idx + 1) * self.batch_size, len(target_list))
                
                # Convert python list to torch tensor
                # Move to GPU
                batch_targets = torch.LongTensor(target_list[start:end]).to(self.device)
                batch_contexts = torch.LongTensor(context_list[start:end]).to(self.device)

                # Loss function (loss_function pass)
                current_batch_size = end - start
                batch_neg = torch.multinomial(self.unigram_dist, current_batch_size * self.num_negatives, replacement=True)
                batch_neg = batch_neg.view(current_batch_size, self.num_negatives).to(self.device)

                optimizer.zero_grad() # Clear gradients

                loss = self.loss_function(batch_targets, batch_contexts, batch_neg)
                
                epoch_loss += loss.item()
                batch_count += 1

                # Backpropagation (backward pass)
                loss.backward()
                optimizer.step() # Update weights
                
            if time_limit_reached:
                break
                
            # Print Epoch Stats
            elapsed = time.time() - start_time #DEBUG
            avg_loss = epoch_loss / max(1, batch_count)
            print(f"Finished Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s") #DEBUG
            
    def train_model(self, author_texts):
        self.author_tokens, all_tokens = self.pre_process(author_texts)
        self.build_vocab(all_tokens)
        self.init_weights()
        self.train_skipgram()
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
            'vocab_counts': self.vocab_counts
        }, f"{save_dir}/model.pt")

        print(f"Model saved to {save_dir}")

    def load(self, directory='./models'):
        import pickle
        
        checkpoint = torch.load(f"{directory}/model.pt", map_location=self.device, weights_only=False)
        self.vocab_size = checkpoint['vocab_size']
        self.embedding_dim = checkpoint['embedding_dim']
        self.vocab_counts = checkpoint['vocab_counts']
        
        self.W_in = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.W_out = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.W_in.load_state_dict(checkpoint['W_in'])
        self.W_out.load_state_dict(checkpoint['W_out'])

        #Load vocab
        with open(f"{directory}/word2idx.pkl", 'rb') as f:
            self.word2idx = pickle.load(f)
        
        with open(f"{directory}/idx2word.pkl", 'rb') as f:
            self.idx2word = pickle.load(f)
        
        self.to(self.device)

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
    #arguments train_directory
    start_time = time.time()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_directory", type=str)
    args = parser.parse_args()
    

    train_dir = args.train_directory
    author_texts = load_data(train_dir)

    model = Word2Vec(100)

    embeddings = model.train_model(author_texts)
    model.save_embeddings('./models')

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total time taken: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
