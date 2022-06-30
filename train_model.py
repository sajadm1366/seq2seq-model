# import matplotlib.pyplot as plt
from seq2seq import Seq2seq
from dataset_gen import *
from utils import mask_sequence

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


@tf.function
def train_step(net, x, labels, weights, target, vocab_size_trg):
    dec_input_y = tf.concat([target['<bos>'] * tf.ones([labels.shape[0], 1], dtype=tf.int32), labels[:, :-1]], axis=1)
    with tf.GradientTape() as tape:
        logits = net(x[0], dec_input_y, training=True)

        labels = tf.one_hot(labels, depth=vocab_size_trg)
        loss_unweight = tf.nn.softmax_cross_entropy_with_logits(
            labels, logits, axis=-1)

        loss_weight = tf.reduce_mean(tf.reduce_sum(weights * loss_unweight, axis=1))

    grads = tape.gradient(loss_weight, net.trainable_variables)
    net.optimizer.apply_gradients(zip(grads, net.trainable_variables))

    return loss_weight




def main():

    # source, target = load_data(num_scenteces=4, max_len=max_len, min_word_count=0,  dataset_path_name="./test/test.txt")
    # source, target = load_data(num_scenteces=190000, max_len=max_len, min_word_count=2)


    data = load_data_gen(dataset_path_name="./fra-eng/fra.txt")
    get_vocab = GetVocab(data)
    vocab_source, vocab_target = get_vocab.tr_get_vocab(replace_char)

    vocab_size_src = len(vocab_source)
    vocab_size_trg = len(vocab_target)
    target_token2word = {val: key for key, val in vocab_target.items()}


    max_len = 10
    preprocess = Preprocess(data, vocab_source, vocab_target, max_len)
    data_process = preprocess.transform
    training_data = data_gen(data_process, max_len)


    training_data = training_data.shuffle(buffer_size=5000)
    training_data_batch = training_data.batch(128).prefetch(buffer_size=100)

    net = Seq2seq(vocab_size_src, vocab_size_trg, max_len)
    net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    loss_all = []
    for batch in range(500):
        losses = []
        for (x, y) in training_data_batch:
            y_len = y[1].numpy()

            labels = y[0]
            weights = tf.ones_like(labels, dtype=tf.float32)
            weights = mask_sequence(weights, y_len)

            loss_weight = train_step(net, x, labels, weights, vocab_target, vocab_size_trg)

            losses.append(loss_weight.numpy())
        loss_all.append(np.mean(losses))
        print(f"epoch: {batch}, loss_val: {np.mean(losses)}")
        # save the model
        if batch % 10 == 0:
           net.save_weights("seq2seq_weights")

    # plt.plot(loss_all)
    # plt.grid()
    # plt.show()

if __name__ == "__main__":
    main()

