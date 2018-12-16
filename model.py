import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
        Args:
            input: batch, seq_len
        Returns:
            attn: batch, seq_len, hidden_size
            outputs: batch, seq_len, hidden_size
    """

    def __init__(self, opt, vocab_size):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size

        self.in_channels = opt.hidden_size * 2
        self.out_channels = opt.hidden_size * 2
        self.kernel_size = opt.kernel_size
        self.stride = 1
        self.padding = (opt.kernel_size - 1) / 2
        self.layers = opt.enc_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.affine = nn.Linear(self.embedding_size, 2*self.hidden_size)
        self.softmax = nn.Softmax()

        self.conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

        self.mapping = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size * 2)

    def forward(self, *input):
        # batch, seq_len_src, dim
        inputs = self.embedding(input)
        # batch, seq_len_src, 2*hidden
        outputs = self.affine(inputs)
        # short-cut
        _outputs = outputs
        for i in range(self.layers):
            # batch, 2*hidden, seq_len_src
            outputs = outputs.permute(0, 2, 1)
            # batch, 2*hidden, seq_len_src
            outputs = self.conv(outputs)
            outputs = F.glu(outputs)
            # batch, seq_len_src, 2*hidden
            outputs = outputs.transpose(1, 2)
            # A, B: batch, seq_len_src, hidden
            A, B = outputs.split(self.hidden_size, 2)
            # A2: batch * seq_len_src, hidden
            A2 = A.contiguous().view(-1, A.size(2))
            # B2: batch * seq_len_src, hidden
            B2 = B.contiguous().view(-1, B.size(2))
            # attn: batch * seq_len_src, hidden
            attn = torch.mul(A2, self.softmax(B2))
            # attn2: batch * seq_len_src, 2 * hidden
            attn2 = self.mapping(attn)

            # outputs: batch, seq_len_src, 2 * hidden
            outputs = attn2.view(A.size(0), A.size(1), -1)

            # batch, seq_len_src, 2 * hidden_size
            out = attn2.view(A.size(0), A.size(1), -1) + _outputs
            _outputs = out

        return attn, out

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)


class Decoder(nn.Module):
    """
    Decoder
        Args:
            Input: batch, seq_len
        return:
            output: seq_len, vocab_size
    """

    def __init__(self, opt, vocab_size):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size

        self.in_channels = opt.hidden_size * 2
        self.out_channels = opt.hidden_size * 2
        self.kernel_size = opt.kernel_size
        self.kernel = opt.kernel_size
        self.stride = 1
        self.padding = (opt.kernel_size - 1) / 2
        self.layers = 1

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.affine = nn.Linear(self.embedding_size, 2 * self.hidden_size)
        self.softmax = nn.Softmax()

        self.conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        
        self.mapping = nn.Linear(self.hidden_size, 2*self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.dict_size)

        self.softmax = nn.Softmax()

    # enc_attn: src_seq_len, hidden_size
    def forward(self, target, enc_attn, source_seq_out):
        # batch, seq_len_tgt, dim
        inputs = self.embedding(target)
        # batch, seq_len_tgt, 2*hidden
        outputs = self.affine(inputs)

        for i in range(self.layers):
            # batch, 2*hidden, seq_len_tgt
            outputs = outputs.permute(0, 2, 1)
            # batch, 2*hidden, seq_len_tgt
            outputs = self.conv(outputs)

            # This is the residual connection,
            # for the output of the conv will add kernel_size/2 elements 
            # before and after the origin input
            if i > 0:
                conv_out = conv_out + outputs

            outputs = F.glu(outputs)

            # batch, seq_len_tgt, 2*hidden
            outputs = outputs.transpose(1, 2)
            # A, B: batch, seq_len_tgt, hidden
            A, B = outputs.split(self.hidden_size, 2)
            # A2: batch * seq_len_tgt, hidden
            A2 = A.contiguous().view(-1, A.size(2))
            # B2: batch * seq_len_tgt, hidden
            B2 = B.contiguous().view(-1, B.size(2))
            # attn: batch * seq_len_tgt, hidden
            dec_attn = torch.mul(A2, self.softmax(B2))

            dec_attn2 = self.mapping(dec_attn)
            dec_attn2 = dec_attn2.view(A.size(0), A.size(1), -1)

            # enc_attn1: batch, seq_len_src, hidden_size
            enc_attn = enc_attn.view(A.size(0), -1, A.size(2))
            # dec_attn1: batch, seq_len_tgt, hidden_size
            dec_attn = dec_attn.view(A.size(0), -1, A.size(2))

            # attn_matrix: batch, seq_len_tgt, seq_len_src
            _attn_matrix = torch.bmm(dec_attn, enc_attn.transpose(1, 2))
            attn_matrix = self.softmax(_attn_matrix.view(-1, _attn_matrix.size(2)))

            # normalized attn_matrix: batch, seq_len_tgt, seq_len_src
            attn_matrix = attn_matrix.view(_attn_matrix.size(0), _attn_matrix.size(1), -1)

            # attns: batch, seq_len_tgt, 2 * hidden_size
            attns = torch.bmm(attn_matrix, source_seq_out)

            # outpus: batch, seq_len_tgt, 2 * hidden_size
            outputs = dec_attn2 + attns

        outputs = F.log_softmax(self.fc(outputs))

        return outputs

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target):
        # attn: batch, seq_len, hidden
        # out: batch, seq_len, 2 * hidden_size
        attn, source_seq_out = self.encoder(source)

        # batch, seq_len_tgt, hidden_size
        out = self.decocer(target, attn, source_seq_out)

        return out

