import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class Encoder(keras.Model):
    ''' Encoder model: takes source language and outputs context vector'''

    def __init__(self, vocab_size, embed_size, hidden_size, num_hiddens):
        super(Encoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.embed = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)

        # self.lstm = layers.LSTM(hidden_size, return_sequences=True, return_state=True)

        self.lstm = [layers.LSTM(hidden_size, return_sequences=True, return_state=True) for _ in range(num_hiddens)]

    def call(self, x):
        ''' Forward pass
            inputs:
                  x: source language (Batch, length)
            outputs:
                 output: (Batch, length, hidden_dims)
        '''

        x = self.embed(x)
        outputs = []
        for i in range(self.num_hiddens):
            x, *output = self.lstm[i](x)
            outputs.append(output)
        # _, *outputs = self.lstm(x)

        # output = self.lstm(x)
        return outputs


class Decoder(keras.Model):
    ''' Decoder model: takes context vector and outputs target langage'''

    def __init__(self, vocab_size, time_steps, embed_size, hidden_size, num_hiddens):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.time_steps = time_steps
        self.num_hiddens = num_hiddens
        self.embed = layers.Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=time_steps)
        self.lstm = [layers.LSTM(hidden_size, return_sequences=True, return_state=True) for _ in range(num_hiddens)]
        # self.lstm = layers.LSTM(hidden_size, return_sequences=True, return_state=True)
        self.dense_1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(512, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(vocab_size)

    def call(self, y, memory_carry_encoder):
        ''' Forward pass
            inputs:
                  y: target language  (Batch, length)
                  memory_carry_encoder: context vector  (2, hidden_dims)
            output:
                 output:  (Batch, length, target_vocab_size)
        '''

        y = self.embed(y)

        for i in range(self.num_hiddens):
           y, _, _ = self.lstm[i](inputs=y, initial_state=memory_carry_encoder[i])
        # y, _, _ = self.lstm(inputs=y, initial_state=memory_carry_encoder)
        out_lstm = y
        out_reshape = tf.reshape(out_lstm, [-1, self.hidden_size])

        out_reshape = self.dense_1(out_reshape)
        out_reshape = self.dense_2(out_reshape)
        out_reshape = self.dense_3(out_reshape)

        out = tf.reshape(out_reshape, [-1, self.time_steps, self.vocab_size])
        return out


class Seq2seq(keras.Model):
    '''
    sequence to sequence model
    '''
    def __init__(self, vocab_size_src, vocab_size_trg, time_steps):
        super(Seq2seq, self).__init__()
        self.time_steps = time_steps

        self.enc = Encoder(vocab_size=vocab_size_src, embed_size=256, hidden_size=512, num_hiddens=3)
        self.dec = Decoder(vocab_size=vocab_size_trg, time_steps=time_steps,
                           embed_size=256, hidden_size=512, num_hiddens=3)
        self.num_hiddens = 3
    def call(self, source, target, training=True):
        '''
        Forward pass
        inputs:
            source: source lanaguge
            target: target languge
        outputs:

        '''
        if training:
            final_memory_carry = self.enc(source)  # batch_size * hidden_size

            return self.dec(target, final_memory_carry)

        else:
            source_scentece = source
            target_token2word = target
            memory_carry_encoder = self.enc(tf.expand_dims(source_scentece, axis=0))

            # memory_encoder, _ = memory_carry_encoder

            init_token = [[2]]
            word = []
            while (init_token[0][0] != 1) and len(word) < 10:
                y = tf.constant(init_token, dtype=tf.float32)  # <bos>

                y = self.dec.embed(y)

                # memory_encoder_expand = tf.repeat(tf.expand_dims(memory_carry_encoder, axis=1), repeats=1, axis=1)
                # concats = tf.concat([y, memory_encoder_expand], axis=2)
                for i in range(self.num_hiddens):
                    y, *memory_carry_encoder[i] = self.dec.lstm[i](inputs=y, initial_state=memory_carry_encoder[i])

                # y, *_ = self.dec.lstm(inputs=y, initial_state=memory_carry_encoder)
                out = y[0]
                out_reshape = self.dec.dense_1(out)
                out_reshape = self.dec.dense_2(out_reshape)
                out_reshape = self.dec.dense_3(out_reshape)

                y_pred_index = np.argmax(tf.nn.softmax(out_reshape, axis=1)[0].numpy())

                init_token = [[y_pred_index]]

                word.append(target_token2word[y_pred_index])
                # print(target_token2word[y_pred_index])

            return ' '.join(word)
