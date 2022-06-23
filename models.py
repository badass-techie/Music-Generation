import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers
import numpy as np


def get_angles(pos, i, d_model):
    """
        Get the angles for the positional encoding

        angle = sin(pos / (10000 ** (2i / d)))

        Params:
            pos : array_like
                  Column vector containing the positions [[0], [1], ...,[N-1]]
            i : array_like
                Row vector containing the dimension span [[0, 1, 2, ..., M-1]]
            d_model : int
                Encoding size / embedding dim

        Returns:
            angles -- (pos, d_model) numpy array
        """
    angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angles


def positional_encoding(positions, d_model):
    """
        Precomputes a matrix with all the positional encodings

        PE[pos, 2i] = sin(angle),
        PE[pos, 2i+1] = cos(angle)

        Params:
            positions : int
                        Maximum number of positions to be encoded / sequence length
            d_model : int
                      Encoding size / embedding dim

        Returns:
            pos_encoding -- (1, position, d_model) A matrix with the positional encodings
        """
    angles = get_angles(np.arange(positions)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angles[:, 0::2] = np.sin(angles[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    pos_encoding = angles[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    """
        Creates a mask to ignore padding tokens

        Params:
            seq : array_like
                  Input sequence

        Returns:
            mask -- (seq, 1, 1) A mask with 1 for padding tokens and 0 for non-padding tokens
        """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
        Creates a look ahead mask to ignore padding tokens

        Params:
            size : int
                   Size of the mask

        Returns:
            mask -- (1, size, size) A mask with 1 for padding tokens and 0 for non-padding tokens
        """
    # mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0) # matrix of upper triangle of ones

    # tf.linalg.band_part() is not supported in tensorflow lite, so we have to use other functions in its place
    a = tf.range(size)
    mask = a[:, None] <= a[None, :]  # true in upper triangle
    mask = tf.transpose(mask)
    mask = tf.cast(mask, tf.float32)
    mask = 1 - mask  # the result is the same
    return mask  # (seq_len, seq_len)


def self_attention(q, k, v, mask):
    """
        Scaled Dot-Product Attention

        Attention(Q, K, V) = softmax(QK / sqrt(d_k)) * V

        Params:
            q : array_like
                Query vectors
            k : array_like
                Key vectors
            v : array_like
                Value vectors
            mask : array_like
                    Mask for padding tokens

        Returns:
            scaled_attention -- (q, 1, d_model)
        """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # apply mask if any
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # compute softmax over last axis (seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(layers.Layer):
    """
        Multi-Head Attention. Given parameters q, k, and v,
        this layer computes wq, wk, and wv,
        then computes scaled attention and its weights for each head.

        Params:
            d_model : int
                      Encoding size / embedding dim
            num_heads : int
                        Number of attention heads
            dropout : float
                      Dropout rate
        """

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
            Split the last dimension into (num_heads, depth).
            Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)

            Params:
                x : array_like
                    Input tensor
                batch_size : int
                             Batch size

            Returns:
                x : array_like
                    Output tensor
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def feed_forward(d_model, dff):
    return Sequential([
        layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(layers.Layer):
    """
        The encoder layer comprises a multi-head self-attention mechanism,
        followed by a simple, position-wise fully connected feed-forward network.
        This architecture includes a residual connection around each of the two
        sub-layers, followed by layer normalization.

        Params:
            d_model : int
                Encoding size / embedding dim
            num_heads : int
                Number of attention heads
            dff : int
                Hidden layer size
            rate : float
                Dropout rate
        """

    def __init__(self, d_model, num_heads, dff, rate=0.1, name="encoder_layer"):
        super(EncoderLayer, self).__init__(name=name)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output)  # skip connection with out1. shape: (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(layers.Layer):
    """
        The decoder layer comprises two multi-head attention blocks,
        one that takes the new input and uses self-attention, and the other
        one that combines it with the output of the encoder, followed by a
        fully connected block. This architecture includes a residual connection
        around each of the two sub-layers, followed by layer normalization.

        Params:
            d_model : int
                Encoding size / embedding dim
            num_heads : int
                Number of attention heads
            dff : int
                Hidden layer size
            rate : float
                Dropout rate
        """

    def __init__(self, d_model, num_heads, dff, rate=0.1, name="decoder_layer"):
        super(DecoderLayer, self).__init__(name=name)

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = feed_forward(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)  # skip connection

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1,
            padding_mask)  # value and key matrices from encoder and query from first mha as defined in the paper. shape: (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # skip connection. shape: (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(layers.Layer):
    """
        The entire Encoder starts by passing the input to an embedding layer
        and using positional encoding to then pass the output through a stack of
        encoder Layers

        Params:
            d_model : int
                Encoding size / embedding dim
            num_layers : int
                Number of encoder layers
            num_heads : int
                Number of attention heads
            dff : int
                Hidden layer size
            input_vocab_size : int
                Size of the input vocabulary
            maximum_positional_encoding : int
                Maximum number of positions in the sequence
            rate : float
                Dropout rate
        """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, name="encoder"):
        super(Encoder, self).__init__(name=name)

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(layers.Layer):
    """
        The entire Decoder starts by passing the input to an embedding layer
        then applying positional encoding to the embeddings to then pass the output through
        a stack of decoder layers.

        Params:
            d_model : int
                Encoding size / embedding dim
            num_layers : int
                Number of encoder layers
            num_heads : int
                Number of attention heads
            dff : int
                Hidden layer size
            target_vocab_size : int
                Size of the target vocabulary
            maximum_positional_encoding : int
                Maximum number of positions in the sequence
            rate : float
                Dropout rate
        """

    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1, name="decoder"):
        super(Decoder, self).__init__(name=name)

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(Model):
    """
        The Transformer architecture from the paper "Attention is all you need".

        Params:
            num_layers : int
                Number of encoder layers
            d_model : int
                Encoding size / embedding dim
            num_heads : int
                Number of attention heads
            dff : int
                Hidden layer size
            input_vocab_size : int
                Size of the input vocabulary
            target_vocab_size : int
                Size of the target vocabulary
            pe_input : int
                Maximum number of positions in the input sequence
            pe_target : int
                Maximum number of positions in the target sequence
            rate : float
                Dropout rate
        """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target,
                 dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, dropout_rate)
        self.final_layer = layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        inp, target = inputs
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, target)
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        dec_output, attention_weights = self.decoder(target, enc_output, training, look_ahead_mask,
                                                     dec_padding_mask)  # dec_output.shape == (batch_size, tar_seq_len, d_model)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights

    def create_masks(self, inp, target):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
        dec_target_padding_mask = create_padding_mask(target)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask


class Generator(Model):
    """
            End to end model. Does all preprocessing and tokenization,
            generates a sequence from the transformer in graph mode,
            then detokenizes the sequence and returns it.

            Params:
                input_vocab : array_like
                    The input vocabulary
                target_vocab : array_like
                    The target vocabulary
                transformer : Transformer
                    The transformer model
                max_length : int
                    Maximum length of the caption
        """

    def __init__(self, transformer, max_length):
        super(Generator, self).__init__()
        self.transformer = transformer
        self.max_length = max_length
        print(f"Output sequence length is {self.max_length}")

    @tf.function(input_signature=[tf.TensorSpec(shape=[26], dtype=tf.int32)])  # I have to hardcode the shape here because you can't pass a class instance to a decorator
                                                                                # Otherwise it would be self.inp_seq_len
                                                                                # There's probably a more elegant way to go about it
    def call(self, inputs):
        encoder_input = tf.expand_dims(inputs, 0)    # add batch dimension

        # initialize start token
        output = tf.zeros([self.max_length], dtype=tf.int32)
        target = tf.constant([[2]], dtype=tf.int32, shape=[1, 1])  # 2 - <BOS>

        def generate_next_token(tar):
            if len(tar.shape) < 2:
                tar = tf.expand_dims(tar, 0)  # somehow, the shape changes from [1,n] to [n] and one dimension is lost, so I have to work around it like this

            out, _ = self.transformer([encoder_input, tar], training=False)  # shape: [1, seq_length, vocab_size]
            out = out[0, -1, :]  # take last token of sequence. shape: [vocab_size]

            # sample the distribution
            out = tf.math.top_k(out, k=2).indices  # shape: [2]
            out = tf.where(tf.not_equal(out[0], 1), out[0], out[1])  # sample the most probable token after <UNK> (1), if <UNK> is the predicted token
                                                                     # we don't want <UNK> in our sequence

            out = tf.expand_dims(tf.expand_dims(out, 0), 0)  # shape: [] -> [1, 1] for concatenation with target
            out = tf.cast(out, tf.int32)

            tar = tf.concat([tar, out], axis=-1)  # append the new token to the sequence
            return tar

        def end_of_sequence_not_reached(tar):
            if len(tar.shape) < 2:
                tar = tf.expand_dims(tar, 0)  # somehow, the shape changes from [1,n] to [n] and one dimension is lost, so I have to work around it like this
            return tf.math.logical_and(tf.less(tf.shape(tar)[-1], self.max_length),
                                       tf.not_equal(tar[0, -1], 3))  # 3 - <EOS>

        target = tf.while_loop(cond=end_of_sequence_not_reached, body=generate_next_token, loop_vars=[target],
                               shape_invariants=[tf.TensorShape([1, None])])[0]

        # copy target into output
        if tf.shape(target)[-1] < self.max_length:
            output = tf.concat([target[0], output[tf.shape(target)[-1]:]], axis=-1)
        else:
            output = target[0]

        return output


if __name__ == "__main__":
    pass

