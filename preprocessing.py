import json
import os
from tqdm import tqdm
import youtokentome as yttm
from Levenshtein import distance
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.keras import preprocessing
import numpy as np


USABLE_KEYS = [i + ":" for i in "BCDFGHIKLMmNOPQRrSsTUVWwXZ"]
START_TOKEN = "<BOS>"
END_TOKEN = "<EOS>"


def read_abc(song):
    keys = []
    notes = []

    for line in song.splitlines():
        #line = line.strip()
        if line.startswith("%"):
            continue

        if any([line.startswith(key) for key in USABLE_KEYS]):
            keys.append(line)
        else:
            notes.append(line)

    keys = "\n".join(keys)
    notes = "".join(notes)
    notes = notes.replace("[", " [")
    notes = notes.replace("]", "] ")
    notes = notes.replace("(", " (")
    notes = notes.replace(")", ") ")
    notes = notes.replace("|", " | ")

    if not keys or not notes:
        return None, None

    return keys, notes


def split_bars(song):
    keys, notes = read_abc(song)
    if not keys or not notes:
        return None, None, None

    notes = notes.split("!")
    # print(notes)
    while notes[-1] == "":
        notes = notes[:-1]
    notes = [note + "!" for note in notes]

    if long_silence(notes) or frequent_repeats(notes):
        return None, None, None

    input_seq = keys.split()
    target_seq = []

    for note in notes:
        target_seq += note.split()

    return keys, input_seq, target_seq


def split_bars_test(song):
    keys, notes = read_abc(song)
    if not keys or not notes:
        return None, None
    input_seq = keys.split()
    return keys, input_seq


def long_silence(bar):
    return any([("x8" in i) for i in bar])


def bars_are_similar(bar1, bar2, difference_thresh=0.2):
    distances = []
    for n1 in bar1:
        distances.append(min([distance(n1, n2) / (len(n1) + len(n2)) for n2 in bar2]))

    return (sum(distances)/len(distances)) < difference_thresh


def frequent_repeats(bars, repeat_thresh=0.33):
    return ((len(bars) - len(set(bars)))/len(bars)) > repeat_thresh


train_data = {"input_seq": [], "target_seq": []}
test_data = {"keys": [], "input_seq": []}


def preprocess(test_size=2000):
    train_path = "dataset/trainset/abc/"
    test_path = "dataset/testset/abc/"

    input_corpus = open("dataset/input_corpus.txt", "a")
    target_corpus = open("dataset/target_corpus.txt", "a")

    # train
    for song in tqdm(os.listdir(train_path)):
        if not song.endswith(".abc"):
            continue

        keys, input_seq, target_seq = split_bars(open(train_path + song).read())
        if keys is None or input_seq is None or target_seq is None:
            continue

        input_corpus.write(keys + " ")
        input_corpus.write(" ".join(input_seq) + " ")
        target_corpus.write(" ".join(target_seq) + " ")

        # writing the data to be serialized to json
        train_data["input_seq"].append([START_TOKEN] + input_seq + [END_TOKEN])
        train_data["target_seq"].append([START_TOKEN] + target_seq + [END_TOKEN])

    train_data["input_seq"] = preprocessing.sequence.pad_sequences(train_data["input_seq"], padding="post", dtype=object, value="<PAD>").tolist()
    train_data["target_seq"] = preprocessing.sequence.pad_sequences(train_data["target_seq"], padding="post", dtype=object, value="<PAD>").tolist()

    # test
    for song in tqdm(os.listdir(test_path)[:test_size]):
        if not song.endswith(".abc"):
            continue

        keys, input_seq = split_bars_test(open(test_path + song).read())
        if keys is None or input_seq is None:
            continue

        input_corpus.write(keys + " " + " ".join(input_seq) + " ")
        test_data["keys"].append(keys + "\n")
        test_data["input_seq"].append([START_TOKEN] + input_seq + [END_TOKEN])

    test_data["input_seq"] = preprocessing.sequence.pad_sequences(test_data["input_seq"], padding="post", dtype=object, value="<PAD>").tolist()

    print(len(train_data["input_seq"]), "training sequences")
    print(len(test_data["input_seq"]), "test sequences")

    input_corpus.close()
    target_corpus.close()


def create_tokens():
    open("dataset/input_vocab.txt", "w", encoding="utf-8").write(convert_to_fast_wordpiece_format(yttm.BPE.train(data="dataset/input_corpus.txt", vocab_size=1000, model="dataset/input_tokenizer.model").vocab()))
    open("dataset/target_vocab.txt", "w", encoding="utf-8").write(convert_to_fast_wordpiece_format(yttm.BPE.train(data="dataset/target_corpus.txt", vocab_size=1000, model="dataset/target_tokenizer.model").vocab()))


def convert_to_fast_wordpiece_format(vocab):
    # in this format ▁ represents a suffix word, unlike
    # in youtokkentome where it represents a space or delimiter.
    # Therefore ▁Cool becomes Cool and er becomes ▁er

    vocab = "\n\u2581".join(vocab)
    vocab = vocab.replace("\u2581\u2581", "")
    vocab = vocab.replace("\n\n", "\n")
    vocab = vocab.replace("\u2581<UNK>", "<UNK>")
    vocab = vocab.replace("\u2581<PAD>", "<PAD>")
    vocab = vocab.replace("\u2581<BOS>", "<BOS>")
    vocab = vocab.replace("\u2581<EOS>", "<EOS>")
    return vocab


def drop_unknowns():
    input_vocab = open("dataset/input_vocab.txt", "r", encoding="utf-8").read().splitlines()
    target_vocab = open("dataset/target_vocab.txt", "r", encoding="utf-8").read().splitlines()

    input_tokenizer = tf_text.FastWordpieceTokenizer(vocab=input_vocab, suffix_indicator='\u2581', max_bytes_per_word=200, token_out_type=tf.int32,
                                                        unknown_token='<UNK>', no_pretokenization=True, support_detokenization=True, model_buffer=None)
    target_tokenizer = tf_text.FastWordpieceTokenizer(vocab=target_vocab, suffix_indicator='\u2581', max_bytes_per_word=200, token_out_type=tf.int32,
                                                        unknown_token='<UNK>', no_pretokenization=True, support_detokenization=True, model_buffer=None)

    unk_train = 0
    idx = 0
    for inp, tar in tqdm(zip(input_tokenizer.tokenize(train_data["input_seq"]), target_tokenizer.tokenize(train_data["target_seq"]))):
        inp = np.concatenate(inp.numpy(), -1)
        tar = np.concatenate(tar.numpy(), -1)
        if np.any(inp == 1) or np.any(tar == 1):
            del train_data["input_seq"][idx]
            del train_data["target_seq"][idx]
            unk_train += 1
        else:
            idx += 1

    print("unknown tokens in training set:", unk_train)


    unk_test = 0
    idx = 0
    for inp in tqdm(input_tokenizer.tokenize(test_data["input_seq"])):
        inp = np.concatenate(inp.numpy(), -1)
        if np.any(inp == 1):
            del test_data["keys"][idx]
            del test_data["input_seq"][idx]
            unk_test += 1
        else:
            idx += 1

    print("unknown tokens in test set:", unk_test)


def to_json():
    with open("dataset/training_set.json", "w") as f:
        json.dump(train_data, f)

    with open("dataset/test_set.json", "w") as f:
        json.dump(test_data, f)


def create_test_set(test_size=2000):
    test_set = {"keys": [], "input_seq": []}
    test_path = "dataset/testset/abc/"
    input_vocab = open("dataset/input_vocab.txt", "r", encoding="utf-8").read().splitlines()
    input_tokenizer = tf_text.FastWordpieceTokenizer(vocab=input_vocab, suffix_indicator='\u2581',
                                                     max_bytes_per_word=200, token_out_type=tf.int32,
                                                     unknown_token='<UNK>', no_pretokenization=True,
                                                     support_detokenization=True, model_buffer=None)

    for song in tqdm(os.listdir(test_path)[:test_size]):
        if not song.endswith(".abc"):
            continue

        keys, input_seq = split_bars_test(open(test_path + song).read())
        if keys is None or input_seq is None:
            continue

        test_input_seq = [START_TOKEN] + input_seq + [END_TOKEN]
        tokens = input_tokenizer.tokenize(test_input_seq).flat_values.numpy()
        if np.any(tokens == 1):
            continue
        test_set["keys"].append(keys + "\n")
        test_set["input_seq"].append(tokens.tolist())

    test_set["input_seq"] = preprocessing.sequence.pad_sequences(test_set["input_seq"], padding="post").tolist()
    with open("test_set.json", "w") as f:
        json.dump(test_set, f)
    print(len(test_set["input_seq"]), "test sequences")


def create_vocab():
    target_vocab = open("dataset/target_vocab.txt", "r", encoding="utf-8").read().splitlines()
    target_vocab = [("\u2581" + token).replace("\u2581\u2581", "").replace("\u2581", " ").replace(" <UNK>", "").replace(" <PAD>", "").replace(" <BOS>", "").replace(" <EOS>", "") for token in target_vocab]
    open("vocab.txt", "w", encoding="utf-8").write("\n".join(target_vocab))


if __name__ == "__main__":
    # preprocess()
    # create_tokens()
    # drop_unknowns()
    # to_json()
    # create_test_set()
    # create_vocab()
    pass

