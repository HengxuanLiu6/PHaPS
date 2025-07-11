import nltk
import re
from nltk.corpus import cmudict

nltk.download('cmudict')
cmu_dict = cmudict.dict()

phoneme_weights_default = {
    'IH': 1.00, 'AH': 0.85, 'T': 0.63, 'S': 0.60,
    'EH': 0.49, 'AA': 0.48, 'IY': 0.48
}

def remove_stress(phoneme):
    return re.sub(r'\d$', '', phoneme)

def word_to_phonetics(word):
    lower_word = word.lower()
    if lower_word in cmu_dict:
        return ' '.join(cmu_dict[lower_word][0])
    else:
        return word

def phrase_to_phonetics(phrase):
    words = phrase.split()
    phonetic_words = [word_to_phonetics(word) for word in words]
    return '   '.join(phonetic_words)

def compute_score(phonemes, duration, mean_pps, weights, w_pps):
    base_score = sum(weights.get(p, 0) for p in phonemes)
    pps = len(phonemes) / duration if duration > 0 else 0
    return base_score + w_pps * (pps / mean_pps)

def select_top_phoneme_samples(dataset, top_k=600, weights=None, w_pps=0.1):
    if weights is None:
        weights = phoneme_weights_default

    train_set = []
    durations = []

    for sample in dataset:
        phonetic_seq = phrase_to_phonetics(sample['sentence'])
        phonemes = [remove_stress(p) for p in phonetic_seq.split()]
        train_set.append(phonemes)

        duration = len(sample['audio']['array']) / sample['audio']['sampling_rate']
        durations.append(duration)

    pps_list = [len(p) / d for p, d in zip(train_set, durations)]
    mean_pps = sum(pps_list) / len(pps_list)

    scores = []
    for i, (phs, dur) in enumerate(zip(train_set, durations)):
        score = compute_score(phs, dur, mean_pps, weights, w_pps)
        scores.append((i, score))

    top_samples = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {
            "index": i,
            "score": round(score, 2),
            "sentence": dataset[i]['sentence'],
            "phonemes": ' '.join(train_set[i])
        }
        for i, score in top_samples
    ]
