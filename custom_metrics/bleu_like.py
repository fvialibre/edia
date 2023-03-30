import sys
sys.path.append("..")

from modules.module_fillmask import FillMask

def bleu_like(sent: str) -> float:
    window_len = 3

    words = sent.strip().split()
    interest_word = ''
    interest_word_idx = -1
    sent_dict = dict(enumerate(words))
    for idx in sent_dict:
        word = sent_dict[idx]
        if word.startswith('<') and word.endswith('>'):
            interest_word = word[1:-1]
            interest_word_idx = idx
    if not interest_word:
        raise ValueError('Couldn\'t find interest word (the one enclosed by \'<\' and \'>\')')

    n_grams_dict = {}
    for i in range(1, window_len + 1):
        if i == 1:
            unigram = ' '.join([sent_dict.get(interest_word_idx - 1, ''), '*', sent_dict.get(interest_word_idx + 1, '')])
            n_grams_dict[1] = [unigram]
            continue

        n_grams_list = []

        last_n_gram = n_grams_dict[i-1][0]
        n_grams_list.append(' '.join([sent_dict.get(interest_word_idx - i, ''), last_n_gram, sent_dict.get(interest_word_idx + i, '')]))
        n_grams_list.append(' '.join([sent_dict.get(interest_word_idx - i, ''), last_n_gram]))
        n_grams_list.append(' '.join([last_n_gram, sent_dict.get(interest_word_idx + i, '')]))
        n_grams_dict[i] = n_grams_list

    fill_mask = FillMask('dccuchile/bert-base-spanish-wwm-cased')

    unigram = n_grams_dict[1][0]
    sent_prob = fill_mask.compute_for_token(unigram, interest_word)
    for i in range(2, window_len + 1):
        prob = 0
        for sent in n_grams_dict[i]:
            prob += fill_mask.compute_for_token(sent, interest_word)

        sent_prob += prob * (10 ** (i-1))

    print(sent_prob)
    return sent_prob
    

bleu_like("El gato negro <come> pescado fresco .")