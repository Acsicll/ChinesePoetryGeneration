import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu(reference, candidate):
    ref_tokens = [list(reference)]
    cand_tokens = [list(candidate)]
    smoothie = SmoothingFunction().method1

    return sentence_bleu(ref_tokens, cand_tokens,  smoothing_function=smoothie)

def human_evaluation(scores):
    return np.mean(scores)

