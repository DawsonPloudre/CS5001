import numpy as np
from collections import defaultdict

# Given paragraph (I just tokenized and manually POS tagged for simplicity)
tagged_corpus = [
    ('NVIDIA', 'NNP'), ('Corporation', 'NNP'), ('is', 'VBZ'), ('a', 'DT'), ('market', 'NN'), ('leader', 'NN'), 
    ('among', 'IN'), ('technology', 'NN'), ('companies', 'NNS'), ('specializing', 'VBG'), ('in', 'IN'), 
    ('graphics', 'NNS'), ('processing', 'NN'), ('units', 'NNS'), ('(', 'SYM'), ('GPUs', 'NNS'), (')', 'SYM'),
    ('and', 'CC'), ('artificial', 'JJ'), ('intelligence', 'NN'), ('(', 'SYM'), ('AI', 'NNP'), (')', 'SYM'),
    ('The', 'DT'), ('company', 'NN'), ('is', 'VBZ'), ('renowned', 'VBN'), ('for', 'IN'), ('its', 'PRP$'), 
    ('innovative', 'JJ'), ('approaches', 'NNS'), ('to', 'TO'), ('both', 'DT'), ('hardware', 'NN'), 
    ('and', 'CC'), ('software', 'NN'), ('solutions', 'NNS'), ('.', '.'), ('It', 'PRP'), ('reported', 'VBD'), 
    ('strong', 'JJ'), ('earnings', 'NNS'), ('in', 'IN'), ('the', 'DT'), ('most', 'RBS'), ('recent', 'JJ'), 
    ('quarter', 'NN'), ('.', '.'), ('It', 'PRP'), ('showcased', 'VBD'), ('robust', 'JJ'), ('revenue', 'NN'),
    ('growth', 'NN'), (',', ','), ('driven', 'VBN'), ('primarily', 'RB'), ('by', 'IN'), ('increasing', 'VBG'), 
    ('demand', 'NN'), ('for', 'IN'), ('its', 'PRP$'), ('GPUs', 'NNS'), ('across', 'IN'), ('various', 'JJ'), 
    ('sectors', 'NNS'), (',', ','), ('including', 'VBG'), ('gaming', 'NN'), ('data', 'NN'), ('centers', 'NNS'),
    ('and', 'CC'), ('professional', 'JJ'), ('visualization', 'NN'), ('.', '.'),
]

# Hopefully this will compute the transition probabilities
tag_counts = defaultdict(int)
transition_counts = defaultdict(lambda: defaultdict(int))
emission_counts = defaultdict(lambda: defaultdict(int))

for i in range(len(tagged_corpus) - 1):
    curr_word, curr_tag = tagged_corpus[i]
    next_word, next_tag = tagged_corpus[i + 1]
    tag_counts[curr_tag] += 1
    transition_counts[curr_tag][next_tag] += 1
    emission_counts[curr_tag][curr_word] += 1

tag_counts[tagged_corpus[-1][1]] += 1  # Last word's tag
emission_counts[tagged_corpus[-1][1]][tagged_corpus[-1][0]] += 1

def compute_probabilities(counts):
    probabilities = {}
    for tag, next_tags in counts.items():
        total = sum(next_tags.values())
        probabilities[tag] = {k: v / total for k, v in next_tags.items()}
    return probabilities

transition_probs = compute_probabilities(transition_counts)
emission_probs = compute_probabilities(emission_counts)

# Viterbi Algorithm, don’t touch this
def viterbi(sentence, transition_probs, emission_probs, tag_counts):
    states = list(tag_counts.keys())
    V = [{}]
    path = {}
    
    # Initialize base cases
    for state in states:
        V[0][state] = emission_probs[state].get(sentence[0], 1e-6) * (tag_counts[state] / sum(tag_counts.values()))
        path[state] = [state]
    
    # run the algorithm??
    for t in range(1, len(sentence)):
        V.append({})
        new_path = {}
        
        for curr_state in states:
            (prob, state) = max((V[t-1][prev_state] * transition_probs[prev_state].get(curr_state, 1e-6) * 
                                  emission_probs[curr_state].get(sentence[t], 1e-6), prev_state) 
                                for prev_state in states)
            V[t][curr_state] = prob
            new_path[curr_state] = path[state] + [curr_state]
        
        path = new_path
    
    # I hope this returns the best path
    (prob, state) = max((V[len(sentence) - 1][s], s) for s in states)
    return path[state]

# Finally! Just test the sentence now
test_sentence = ['NVIDIA', 'is', 'a', 'leader', 'among', 'technology', 'companies']
predicted_tags = viterbi(test_sentence, transition_probs, emission_probs, tag_counts)
print("Predicted POS Tags:", list(zip(test_sentence, predicted_tags)))
