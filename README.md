# Word2Vec Author Attribution Assignment

## Contents
- `data/train_data/` - Training data (author_001.txt, author_002.txt, ...)
- `sample_inputs/` - Example test inputs
- `assignment.pdf` - Full assignment description
- `run_model.sh` - Submission template
- `hungarian_eval.py` - Task 2 evaluation helper

## Quick Start
1. Read `assignment.pdf` carefully
2. Implement Word2Vec from scratch (PyTorch/NumPy)
3. Train on `data/train_data/` directory
4. Test with `sample_inputs/`
5. Submit according to guidelines

## Important Notes
- Authors are anonymized (author_001, author_002, etc.)
- Test set may contain different (unseen) authors
- Chunk sizes vary from 50-500 words
- You must implement Word2Vec yourself (no gensim/pre-trained)
