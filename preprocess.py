import argparse
import torch
import re
import itertools
from collections import Counter
import constants
import os

parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config', help="Read options from this file")


parser.add_argument('-files', type=str, default="/home/zeng/conversation/OpenNMT-py/data/test/",
                    help="Path to the training source data")
parser.add_argument('-source_train_file', type=str, default="/home/zeng/data/OpenSubData/train.src",
                    help="Path to the training source data")
parser.add_argument('-target_train_file', type=str, default="/home/zeng/data/OpenSubData/train.tgt",
                    help="Path to the training target data")
parser.add_argument('-source_valid_file', type=str, default="/home/zeng/data/OpenSubData/valid.src",
                    help="Path to the training source data")
parser.add_argument('-target_valid_file', type=str, default="/home/zeng/data/OpenSubData/valid.tgt",
                    help="Path to the training target data")
parser.add_argument('-source_test_file', type=str, default="/home/zeng/data/OpenSubData/test.src",
                    help="Path to the training source data")
parser.add_argument('-target_test_file', type=str, default="/home/zeng/data/OpenSubData/test.tgt",
                    help="Path to the training target data")

parser.add_argument('-save_data', type=str, default="/home/zeng/data/OpenSubData/5m",
                    help="Output file for the prepared data")

parser.add_argument('-maximum_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")

parser.add_argument('-vocab',
                    help="Path to an existing vocabulary")

parser.add_argument('-seq_length', type=int, default=50,
                    help="Maximum sequence length")
parser.add_argument('-shuffle', type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed', type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=1000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def build_vocab(sequence, maximum_vocab_size=50000):
    word_count = Counter(itertools.chain(*sequence)).most_common(maximum_vocab_size)
    word2count = dict([(word[0], word[1]) for word in word_count])

    word2index = dict([(word, index + 4) for index, word in enumerate(word2count) if word != "UNknown"])
    word2index[constants.PAD_WORD], word2index[constants.BOS_WORD], word2index[constants.EOS_WORD], word2index[
        constants.UNK_WORD] = \
        constants.PAD, constants.BOS, constants.EOS, constants.UNK

    index2word = dict([(index + 4, word) for index, word in enumerate(word2count) if word != "UNknown"])
    index2word[constants.PAD], index2word[constants.BOS], index2word[constants.EOS], index2word[
        constants.UNK] = constants.PAD_WORD, \
                         constants.BOS_WORD, constants.EOS_WORD, constants.UNK_WORD

    # word2index[constants.PAD_WORD], word2index[constants.BOS_WORD], word2index[constants.EOS_WORD], word2index[constants.UNK_WORD] = \
    # constants.PAD, constants.BOS, constants.EOS, constants.UNK

    index2word[constants.PAD], index2word[constants.BOS], index2word[constants.EOS], index2word[constants.UNK] = \
        constants.PAD_WORD, constants.BOS_WORD, constants.EOS_WORD, constants.UNK_WORD
    return word2count, word2index, index2word


def makeData(sources, targets, src_word2index, tgt_word2index, shuffle=opt.shuffle):
    assert len(sources) == len(targets)
    sizes = []
    for idx in range(len(sources)):
        # Insert  `eosWord` at the end
        src_words = [src_word2index[word] if word in src_word2index else constants.UNK for word in sources[idx]] + [
            constants.EOS]
        sources[idx] = torch.LongTensor(src_words)

        sizes += [len(sources)]

        tgt_words = [constants.BOS] + [tgt_word2index[word] if word in tgt_word2index else constants.UNK for word in targets[idx]] + [
            constants.EOS]
        targets[idx] = torch.LongTensor(tgt_words)

    if shuffle == 1:
        print("... shuffling sentences")
        perm = torch.randperm(len(sources))
        sources = [sources[idx] for idx in perm]
        targets = [targets[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print("... sorting sentences")
    _, perm = torch.sort(torch.Tensor(sizes))
    sources = [sources[idx] for idx in perm]
    targets = [targets[idx] for idx in perm]

    return sources, targets


def load_source_and_target(source_file, target_file):
    """
    Source_file
    Target_file
    """

    src_lines = open(source_file, "r").readlines()
    tgt_lines = open(target_file, "r").readlines()

    sources = []
    targets = []

    for src, tgt in zip(src_lines, tgt_lines):
        src = src.strip().split()
        tgt = tgt.strip().split()

        sources.append(src)
        targets.append(tgt)

    return sources, targets


def main():

    # train
    source_train_file = os.path.join(opt.files, "train.src")
    target_train_file = os.path.join(opt.files, "train.tgt")

    # valid
    source_valid_file = os.path.join(opt.files, "valid.src")
    target_valid_file = os.path.join(opt.files, "valid.tgt")

    # test
    source_test_file = os.path.join(opt.files, "test.src")
    target_test_file = os.path.join(opt.files, "test.tgt")

    source_train, target_train = load_source_and_target(source_train_file, target_train_file)
    source_valid, target_valid = load_source_and_target(source_valid_file, target_valid_file)
    source_test, target_test = load_source_and_target(source_test_file, target_test_file)

    source_texts = source_train + source_valid + source_test
    target_texts = target_train + target_valid + target_test

    src_word2count, src_word2index, src_index2word = build_vocab(source_texts, opt.maximum_vocab_size)
    tgt_word2count, tgt_word2index, tgt_index2word = build_vocab(target_texts, opt.maximum_vocab_size)

    dicts = {}
    word2index = {}
    word2index["src"] = src_word2index
    word2index["tgt"] = tgt_word2index
    index2word = {}
    index2word["src"] = src_index2word
    index2word["tgt"] = tgt_index2word
    dicts["word2index"] = word2index
    dicts["index2word"] = index2word

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = makeData(source_train, target_train, src_word2index, tgt_word2index)

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'] = makeData(source_valid, target_valid, src_word2index, tgt_word2index)

    print('Preparing testing ...')
    valid = {}
    valid['src'], valid['tgt'] = makeData(source_test, target_test, src_word2index, tgt_word2index)

    print("saving data to \'" + opt.save_data + ".train.pt\'...")
    save_data = {
        "train": train,
        "valid": valid,
        "test": valid,
        "dicts": dicts
    }
    torch.save(save_data, opt.save_data + ".train.pt")


if __name__ == "__main__":
    main()
