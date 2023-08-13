# Transfer Learning, Multi-Task Learning, and Few-Shot Learning Experiments

This project explores transfer learning techniques like domain adaptation, task adaptation, and few-shot learning. It also implements and analyzes various multi-task learning strategies.

## Experiments

### Transfer Learning

- Fine-tuned BERT on MultiNLI for domain adaptation from telephone/fiction genres to slate/travel 
- Implemented importance weighting algorithm to select relevant training samples 
- Evaluated zero-shot transfer and continued fine-tuning baselines
- Analyzed models using transferability metrics like LEEP score

### Multi-Task Learning 

- Trained BERT simultaneously and sequentially on WikiNER and MultiNLI datasets
- Tested static and dynamic weighting schemes during simultaneous training
- Evaluated transfer performance to machine reading comprehension with SQuAD dataset

### Few-Shot Learning

- Fine-tuned BERT backbone pretrained on MultiNLI and WikiNER for few-shot SQuAD training
- Froze backbone and trained task-specific QA head with 10% of SQuAD data

## Results

- Task adaptation outperformed domain adaptation in transfer learning experiments
- Dynamic weighting improved multi-task learning performance over static weighting
- NLI dataset transfers better than NER dataset to MRC task

## References

[1] Liu et al., End-to-End Multi-Task Learning with Attention  
[2] Project Report
