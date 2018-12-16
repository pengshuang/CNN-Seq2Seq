import math
import torch
from torch.autograd import Variable


class Dataset(object):

    def __init__(self, xs, ys, batch_size, cuda, volatile=False):

        self.xs = xs
        self.ys = ys
        assert (len(self.xs) == len(self.ys))
        self.batch_size = batch_size
        self.numBatches = math.ceil(len(self.xs)/batch_size)
        self.volatile = volatile
        self.cuda = cuda

    def _batchify(self, data, align_right=False, include_lengths=False, PADDING_TOKEN=0):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(PADDING_TOKEN)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        xs, lengths = self._batchify(
            self.xs[index*self.batch_size:(index+1)*self.batch_size], align_right=False, include_lengths=True)

        ys = self._batchify(
            self.ys[index*self.batch_size:(index+1)*self.batch_size])

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(xs))
        batch = zip(indices, xs, ys)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        indices, xs, ys = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        return (wrap(xs), lengths), wrap(ys)

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.xs, self.ys))
        self.xs, self.ys = zip(*[data[i] for i in torch.randperm(len(data))])