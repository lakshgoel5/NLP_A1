import torch
import os
import torch.nn as nn
import numpy as np
import math
import re
from collections import Counter # for counting word frequencies
import time
from tqdm import tqdm
from multiprocessing import Pool

class Word2Vec(nn.Module):
    def __init__(self, embedding_dim = 100, model_type = 'sg'):
        super().__init__() #My class becomes a torch module
        self.embedding_dim = embedding_dim
        self.model_type = model_type # cbow or sg
        self.W_in = None
        self.W_out = None

        self.author_tokens = None

        self.lr = 0.025
        self.epochs = 8
        self.window_size = 5

        self.total_words = 0
        self.word2idx = {} # Dict: word -> index
        self.idx2word = {} # Dict: index -> word
        self.vocab_size = 0
        self.vocab_counts = [] # List: index -> frequency

        self.model_speed = "ns"
        self.num_negatives = 5
        self.unigram_dist = None

        self.batch_size = 256
        
        # Subsampling threshold for frequent words
        self.subsample_threshold = 1e-5
        self.stop_words = False
        
        # Distance-based context weighting (Word-Space Model)
        self.use_distance_weighting = True  # Enable distance-based weighting
        self.weighting_scheme = "glove"  # "aggressive" or "glove"
        
        # Document-aware embeddings (Paragraph Vector / Doc2Vec)
        self.use_doc_embeddings = False  # Enable document embeddings
        self.num_docs = 0
        self.doc2idx = {}  # Dict: author_id -> doc_index
        self.idx2doc = {}  # Dict: doc_index -> author_id
        self.D = None  # Document embedding matrix

        # DEBUG
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
            text = ' '.join(text.split())
            #remove stop words -> Think
            #lemmatization -> Think

            tokens = re.findall(r'\w+|[^\w\s]', text) #tokenise with punctuations
            #\w+ matches any word character (equal to [a-zA-Z0-9_])
            #[^\w\s] matches any non-word character and non-space (equal to [^a-zA-Z0-9_])

            # Keep word tokens (including one-char words like "i", "a"),
            # but drop standalone punctuation tokens.
            tokens = [token for token in tokens if token.isalnum()]

            if self.stop_words == True:
                tokens = [token for token in tokens if token not in stop_words]
            author_tokens[author_id] = tokens
            all_tokens.extend(tokens)

        return author_tokens, all_tokens

    def build_vocab(self, all_tokens, min_freq=2):
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
        
        # Build document index if document embeddings are enabled
        if self.use_doc_embeddings and self.author_tokens is not None:
            doc_ids = sorted(self.author_tokens.keys())
            self.doc2idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
            self.idx2doc = {idx: doc_id for doc_id, idx in self.doc2idx.items()}
            self.num_docs = len(doc_ids)
            print(f"Document index built: {self.num_docs} documents")

        print(f"Vocabulary built: {self.vocab_size} unique tokens")
        print(f"Total tokens: {self.total_words}")
        print(f"Vocab build took: {time.time() - start:.2f}s\n") #DEBUG

        return

    def subsample_prob(self, word_idx):
        """
        Calculate probability of keeping a word based on its frequency.
        Formula from Word2Vec paper:
        P(keep) = (sqrt(word_freq / (threshold * total_words)) + 1) * (threshold * total_words) / word_freq
        
        Frequent words are discarded with higher probability.
        """
        if self.subsample_threshold <= 0:
            return 1.0  # No subsampling
        
        word_freq = self.vocab_counts[word_idx].item()
        
        # Handle zero or very low frequency (<UNK>, <PAD> tokens)
        if word_freq == 0:
            return 1.0  # Always keep words with zero frequency
        
        freq_ratio = word_freq / (self.subsample_threshold * self.total_words)
        
        # If freq_ratio is very small, just keep the word
        if freq_ratio < 1e-10:
            return 1.0
        
        # Calculate keep probability
        keep_prob = (math.sqrt(freq_ratio) + 1) / freq_ratio
        
        return min(keep_prob, 1.0)  # Clamp to [0, 1]
    
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
        
        # Initialize document embeddings if enabled
        if self.use_doc_embeddings and self.num_docs > 0:
            self.D = nn.Embedding(self.num_docs, self.embedding_dim)
            self.D.weight.data.uniform_(-init_range, init_range)
            print(f"Document embeddings initialized: {self.num_docs} x {self.embedding_dim}")
        
        # Move everything to GPU/CPU at once
        self.to(self.device)
        
        print(f"Embeddings initialized: {self.vocab_size} x {self.embedding_dim}")
        
        return

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


    def forward(self, input_data):
        if self.model_type == 'cbow':
            pad_idx = self.word2idx.get("<PAD>", -1)
            context_embeds = self.W_in(input_data)  # [batch, window, emb]
            mask = (input_data != pad_idx).unsqueeze(-1).float()  # [batch, window, 1]
            masked_embeds = context_embeds * mask
            sum_embeds = masked_embeds.sum(dim=1)  # [batch, emb]
            counts = mask.sum(dim=1).clamp(min=1)  # [batch, 1]
            h = sum_embeds / counts  # [batch, emb]
        else:  # sg
            h = self.W_in(input_data)  # [batch, emb]

        scores = torch.matmul(h, self.W_out.weight.t())  # [batch, vocab_size]
        return scores

    def loss_function(self, target_list, context_list, weights=None, doc_ids=None):
        # context_list:
        #   CBOW: Tensor of [batch_size, window_size] (Indices of context words, Padded)
        #   SG:   Tensor of [batch_size] (Indices of single context words)
        # weights: Optional tensor for distance-based weighting
        # doc_ids: Optional tensor of [batch_size] for document IDs

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
            
            # Apply distance-based weighting if provided
            if weights is not None:
                # weights: (Batch, Window_Size)
                # Expand to (Batch, Window_Size, 1) to multiply with embeddings
                weight_mask = weights.unsqueeze(-1) * mask
                # Weighted embeddings
                weighted_embeds = context_embeds * weight_mask
                # Sum weighted embeddings
                sum_embeds = weighted_embeds.sum(dim=1)
                # Sum of weights (for normalization)
                weight_sum = weight_mask.sum(dim=1).clamp(min=1e-8)
                # Weighted average
                h = sum_embeds / weight_sum
            else:
                # Original uniform averaging
                # Zero out the embeddings of padding tokens
                # Dimension of masked_embeds: Batch_Size x Window_Size x Embedding_Dim
                masked_embeds = context_embeds * mask
                
                # Sum the embeddings along the window dimension i.e. sum over all context words of target word
                # Dimension of sum_embeds: Batch_Size x Embedding_Dim
                sum_embeds = masked_embeds.sum(dim=1)
                
                # Count the number of real words in each window to get the mean
                # Dimension of counts: Batch_Size x 1
                counts = mask.sum(dim=1)

                # Vectorized version
                counts = counts.clamp(min=1e-8)
                # Replace all 0s with 1 in one operation
                
                # Calculate the average vector "h"
                # Dimension of h: Batch_Size x Embedding_Dim
                h = sum_embeds / counts
                # Tensor extends dimensions by itself
            
            # Add document embedding if enabled
            if self.use_doc_embeddings and doc_ids is not None and self.D is not None:
                doc_embeds = self.D(doc_ids)  # (Batch, Emb)
                h = h + doc_embeds  # Combine word context and document embeddings

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
            
            # Add document embedding if enabled
            if self.use_doc_embeddings and doc_ids is not None and self.D is not None:
                doc_embeds = self.D(doc_ids)  # (Batch, Emb)
                h = h + doc_embeds  # Combine word and document embeddings

            # Score against the entire vocabulary
            # We calculate how well 'h' predicts "every" word in the vocab.
            # Dimension of scores: (Batch_Size, Vocab_Size)
            scores = torch.matmul(h, self.W_out.weight.t())

            # Calculate Loss
            # In SG, we want to maximize the probability of the *true context word*.
            # 'context_list' here contains the indices of the true context words.
            log_probs = torch.nn.functional.log_softmax(scores, dim=1)
            
            # Apply distance-based weighting if provided
            if weights is not None:
                # Get log probs for the true context words
                target_log_probs = log_probs.gather(1, context_list.unsqueeze(1)).squeeze(1)
                # Weight the loss by distance
                loss = -(target_log_probs * weights).mean()
            else:
                loss = torch.nn.functional.nll_loss(log_probs, context_list)

        return loss

    def loss_function_ns(self, target_list, context_list, negative_samples, weights=None, doc_ids=None):
        pad_idx = self.word2idx.get("<PAD>", -1)

        if self.model_type == 'cbow':
            context_embeds = self.W_in(context_list)
            mask = (context_list != pad_idx).unsqueeze(-1).float()
            
            # Apply distance-based weighting if provided
            if weights is not None:
                # weights: (Batch, Window_Size)
                # Expand to (Batch, Window_Size, 1) to multiply with embeddings
                weight_mask = weights.unsqueeze(-1) * mask
                # Weighted embeddings
                weighted_embeds = context_embeds * weight_mask
                # Sum weighted embeddings
                sum_embeds = weighted_embeds.sum(dim=1)
                # Sum of weights (for normalization)
                weight_sum = weight_mask.sum(dim=1).clamp(min=1e-8)
                # Weighted average
                h = sum_embeds / weight_sum
            else:
                # Original uniform averaging
                masked_embeds = context_embeds * mask
                sum_embeds = masked_embeds.sum(dim=1)
                counts = mask.sum(dim=1)
                counts = counts.clamp(min=1)
                h = sum_embeds / counts
            
            # Add document embedding if enabled
            if self.use_doc_embeddings and doc_ids is not None and self.D is not None:
                doc_embeds = self.D(doc_ids)  # (Batch, Emb)
                h = h + doc_embeds  # Combine word context and document embeddings

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
            
            # Add document embedding if enabled
            if self.use_doc_embeddings and doc_ids is not None and self.D is not None:
                doc_embeds = self.D(doc_ids)  # (Batch, Emb)
                h = h + doc_embeds  # Combine word and document embeddings

            positive_out = self.W_out(context_list)
            positive_score = torch.sum(h * positive_out, dim=1)

            negative_out = self.W_out(negative_samples)
            #h.unsqueeze(-1): (Batch, Emb, 1)
            #negative_out: (Batch, num_negative_samples, Emb)
            #(num_negative_samples, Emb) @ (Emb, 1) -> (num_negative_samples, 1)
            #negative_score: (Batch, num_negative_samples)
            negative_score = torch.bmm(negative_out, h.unsqueeze(-1)).squeeze(-1)

            # Calculate per-sample loss
            sample_loss = -torch.nn.functional.logsigmoid(positive_score) - torch.sum(torch.nn.functional.logsigmoid(-negative_score), dim=1)
            
            # Apply distance-based weighting if provided
            if weights is not None:
                loss = (sample_loss * weights).mean()
            else:
                loss = sample_loss.mean()
            
            return loss

    def _generate_cbow_pairs(self, author_id, tokens):

        target_list = []
        context_lists = []
        weight_lists = []  # Store distance weights for each context word
        doc_id_list = []  # Store document IDs

        # Convert tokens to indices
        indices = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        
        # Get document index if document embeddings are enabled
        doc_idx = self.doc2idx.get(author_id, 0) if self.use_doc_embeddings else 0

        if len(indices) < self.window_size - 1:
            return target_list, context_lists, weight_lists, doc_id_list
        
        for i in range(len(indices)):
            target_word = indices[i]
            
            # Subsampling: randomly discard frequent words
            # As I dont want to spend too much time just training on common words
            keep_prob = self.subsample_prob(target_word)
            if np.random.random() > keep_prob:
                continue
            
            # Define context window
            dynamic_window = np.random.randint(1, self.window_size + 1)
            window_start = max(0, i - dynamic_window)
            window_end = min(len(indices), i + dynamic_window + 1)

            context = []
            weights = []
            for j in range(window_start, window_end):
                if i != j:
                    distance = abs(i - j)
                    weight = self.get_distance_weight(distance)
                    context.append(indices[j])
                    weights.append(weight)
            
            #(target, context list, weight list, doc_id)
            if(len(context) > 0):
                target_list.append(target_word)
                context_lists.append(context)
                weight_lists.append(weights)
                doc_id_list.append(doc_idx)

        return target_list, context_lists, weight_lists, doc_id_list
        
    def train_cbow(self):
        # Initialize Adam Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        target_list = []
        context_lists = []
        weight_lists = []  # Store distance weights for each context word
        doc_id_list = []  # Store document IDs

        num_workers = min(3, os.cpu_count() - 1) #DEBUG

        with Pool(processes=num_workers) as pool:
            results = pool.starmap(self._generate_cbow_pairs, [(author_id, tokens) for author_id, tokens in self.author_tokens.items()])

        for res in results:
            target_list.extend(res[0])
            context_lists.extend(res[1])
            weight_lists.extend(res[2])
            doc_id_list.extend(res[3])

        #Train
        print(f"Starting CBOW Training on {self.device}...") #DEBUG
        for epoch in range(self.epochs):
            start_time = time.time() #DEBUG
            epoch_loss = 0
            batch_count = 0

            #In train mode
            self.train()

            num_batches = (len(target_list) + self.batch_size - 1) // self.batch_size
            
            # Progress Bar
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.epochs}")

            for i in pbar:
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, len(target_list))

                #Convert to tensors
                batch_target = torch.LongTensor(target_list[start:end]).to(self.device)
                # using LongTensor gurantees that indices of embedding are intigers
                batch_contexts_raw = context_lists[start:end]
                batch_weights_raw = weight_lists[start:end]
                batch_doc_ids = torch.LongTensor(doc_id_list[start:end]).to(self.device)
                # context lists might have different sizes, so cant be made a tensor
                max_context_size = max(len(context) for context in batch_contexts_raw)
                pad_idx = self.word2idx["<PAD>"]
                padded_contexts = [c + [pad_idx] * (max_context_size - len(c)) for c in batch_contexts_raw]
                padded_weights = [w + [0.0] * (max_context_size - len(w)) for w in batch_weights_raw]
                batch_contexts = torch.LongTensor(padded_contexts).to(self.device)
                batch_weights = torch.FloatTensor(padded_weights).to(self.device)

                optimizer.zero_grad() # Clear gradients

                loss = 0
                if(self.model_speed == "softmax"):
                    loss = self.loss_function(batch_target, batch_contexts, batch_weights, batch_doc_ids)
                else:
                    current_batch_size = end - start
                    # This picks 5 random words for every single target word in the batch. By doing this inside the batch, we ensure that the model sees different "negative" examples in every epoch, which is key for learning what a word is not.

                    #It may pick target word as well, but that noise can be ignored
                    batch_neg = torch.multinomial(self.unigram_dist, current_batch_size * self.num_negatives, replacement=True)
                    # .view reshapes 1D vector to 2D matrix of size (batch_size, num_negatives)
                    batch_neg = batch_neg.view(current_batch_size, self.num_negatives).to(self.device)

                    loss = self.loss_function_ns(batch_target, batch_contexts, batch_neg, batch_weights, batch_doc_ids)
                
                epoch_loss += loss.item()
                batch_count += 1

                loss.backward()
                optimizer.step() # Update weights
                
                # Update progress bar with loss
                pbar.set_postfix({'loss': loss.item()})
            
            elapsed = time.time() - start_time #DEBUG
            print(f"Finished Epoch {epoch+1}, Loss: {epoch_loss/batch_count:.4f}, Time: {elapsed:.2f}s") #DEBUG 

    def _generate_sg_pairs(self, author_id, tokens):
        # Create target and context pairs with distance weights and document IDs
        target_list = []
        context_list = []
        weight_list = []  # Store distance weights
        doc_id_list = []  # Store document IDs

        # Convert tokens to indices
        indices = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        
        # Get document index if document embeddings are enabled
        doc_idx = self.doc2idx.get(author_id, 0) if self.use_doc_embeddings else 0

        if len(indices) < self.window_size - 1:
            return target_list, context_list, weight_list, doc_id_list

        
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
                
                # Calculate distance and weight
                distance = abs(i - j)
                weight = self.get_distance_weight(distance)
                
                #pushed as pairs with weights and doc_id
                context_word = indices[j]
                target_list.append(target_word)
                context_list.append(context_word)
                weight_list.append(weight)
                doc_id_list.append(doc_idx)

        return target_list, context_list, weight_list, doc_id_list

    def train_skipgram(self):
        # Initialize Adam Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        num_workers = min(3, os.cpu_count() - 1) #DEBUG

        target_list = []
        context_list = []
        weight_list = []
        doc_id_list = []

        with Pool(processes=num_workers) as pool:
            results = pool.starmap(self._generate_sg_pairs, [(author_id, tokens) for author_id, tokens in self.author_tokens.items()])

        for res in results:
            target_list.extend(res[0])
            context_list.extend(res[1])
            weight_list.extend(res[2])
            doc_id_list.extend(res[3])

        #Train
        print(f"Starting Skip-Gram Training on {self.device}...") #DEBUG
        for epoch in range(self.epochs):
            start_time = time.time() #DEBUG
            epoch_loss = 0
            batch_count = 0

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
                batch_weights = torch.FloatTensor(weight_list[start:end]).to(self.device)
                batch_doc_ids = torch.LongTensor(doc_id_list[start:end]).to(self.device)

                optimizer.zero_grad() # Clear gradients

                # Loss function (loss_function pass)
                if(self.model_speed == "softmax"):
                    loss = self.loss_function(batch_targets, batch_contexts, batch_weights, batch_doc_ids)
                else:
                    current_batch_size = end - start
                    batch_neg = torch.multinomial(self.unigram_dist, current_batch_size * self.num_negatives, replacement=True)
                    batch_neg = batch_neg.view(current_batch_size, self.num_negatives).to(self.device)

                    loss = self.loss_function_ns(batch_targets, batch_contexts, batch_neg, batch_weights, batch_doc_ids)
                
                epoch_loss += loss.item()
                batch_count += 1

                # Backpropagation (backward pass)
                loss.backward()
                optimizer.step() # Update weights
                
                pbar.set_postfix({'loss': loss.item()})

            # Print Epoch Stats
            elapsed = time.time() - start_time #DEBUG
            avg_loss = epoch_loss / max(1, batch_count)
            print(f"Finished Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s") #DEBUG
            
    def train_model(self, author_texts):
        self.author_tokens, all_tokens = self.pre_process(author_texts)
        #prepration
            #Read vocab
            #Count word frequencies
            #Discard rare words
        self.build_vocab(all_tokens)
            #Create embedding matrix W
            #Create output matrix U
        self.init_weights()

        #train using multithreading
            #Do loss_function pass
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

    def load(self, directory='./models'):
        import pickle
        
        checkpoint = torch.load(f"{directory}/model.pt", map_location=self.device, weights_only=False)
        self.vocab_size = checkpoint['vocab_size']
        self.embedding_dim = checkpoint['embedding_dim']
        self.model_type = checkpoint['model_type']
        
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_directory", type=str)
    args = parser.parse_args()
    

    train_dir = args.train_directory
    author_texts = load_data(train_dir)

    model = Word2Vec(300, "sg")

    embeddings = model.train_model(author_texts)
    model.save_embeddings('./models')


if __name__ == "__main__":
    main()
