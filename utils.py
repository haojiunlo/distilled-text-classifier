import numpy as np

def get_single_sentence_data(texts, max_len, tokenizer):
    input_ids_ls = []
    seg_ids_ls = []
    for text in texts:
        input_ids, seg_ids = tokenizer.encode(text, first_length=max_len)
        input_ids_ls.append(input_ids)
        seg_ids_ls.append(seg_ids)
    return np.array(input_ids_ls), np.array(seg_ids_ls)


def get_sentence_pair_data(texts_1, texts_2, maxlen, tokenizer):
    token_ids_ls, segment_ids_ls = [], []
    for text1, text2 in zip(texts_1, texts_2):
        token_ids, segment_ids = tokenizer.encode(
            text1, text2, first_length=maxlen, second_length=maxlen
        )
        token_ids_ls.append(token_ids)
        segment_ids_ls.append(segment_ids)
    return np.array(token_ids_ls), np.array(segment_ids_ls)