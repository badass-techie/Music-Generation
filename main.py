import tensorflow as tf
from tensorflow.keras import losses, optimizers, preprocessing
import tensorflow_text as tf_text
from tensorflow.lite.python import interpreter
import numpy as np
from models import Transformer, Generator
from matplotlib import pyplot as plt
import json
import time
import os
from tqdm import tqdm


""" helpers """

#displays time as h:mm:ss
def format_time(seconds):
    return "{}:{:0>2}:{:0>2}".format(int(seconds//3600), int((seconds//60)%60), int(seconds%60))


""" processing the dataset """

input_vocab = open("dataset/input_vocab.txt", encoding="utf-8").read().splitlines()
target_vocab = open("dataset/target_vocab.txt", encoding="utf-8").read().splitlines()

input_tokenizer = tf_text.FastWordpieceTokenizer(vocab=input_vocab, suffix_indicator='\u2581', max_bytes_per_word=200, token_out_type=tf.int32,
                                                              unknown_token='<UNK>', no_pretokenization=True, support_detokenization=True, model_buffer=None)
target_tokenizer = tf_text.FastWordpieceTokenizer(vocab=target_vocab, suffix_indicator='\u2581', max_bytes_per_word=200, token_out_type=tf.int32,
                                                              unknown_token='<UNK>', no_pretokenization=True, support_detokenization=True, model_buffer=None)

inputs, targets = None, None
keys, contexts = None, None

def load_training_data():
    global inputs, targets
    input_seqs, target_seqs = [], []
    print("loading training data...")
    training_set = json.load(open("dataset/training_set.json", encoding="utf-8"))

    for inp, tar in tqdm(zip(input_tokenizer.tokenize(training_set["input_seq"]), target_tokenizer.tokenize(training_set["target_seq"]))):
        input_seqs.append(np.concatenate(inp.numpy(), -1))
        target_seqs.append(np.concatenate(tar.numpy(), -1))

    inputs = preprocessing.sequence.pad_sequences(input_seqs, padding="post")
    targets = preprocessing.sequence.pad_sequences(target_seqs, padding="post")

load_training_data()

def load_test_data():
    global keys, contexts
    test_set = json.load(open("test_set.json", encoding="utf-8"))
    keys, contexts = np.array(test_set["keys"], dtype=str), np.array(test_set["input_seq"], dtype=np.int32)

load_test_data()

def get_batch(batch_size, is_training=True):
    if is_training:
        assert batch_size < len(inputs)
        indices = np.random.choice(len(inputs), batch_size, replace=False)
        inp_batch, tar_batch = inputs[indices], targets[indices]
        return inp_batch, tar_batch
    else:
        assert batch_size < len(contexts)
        indices = np.random.choice(len(contexts), batch_size, replace=False)
        key_batch, inp_batch = keys[indices], contexts[indices]
        return key_batch, inp_batch


""" model """

embedding_dim = 128
model = Transformer(num_layers=4, d_model=embedding_dim, num_heads=8, dff=512, input_vocab_size=len(input_vocab),
                    target_vocab_size=len(target_vocab), pe_input=inputs.shape[-1]+30, pe_target=targets.shape[-1])
model.build(input_shape=[[None, inputs.shape[-1]], [None, targets.shape[-1]]])


""" training configuration """

loss_object = losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(embedding_dim, warmup_steps=4000)
optimizer = optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
    pred = tf.argmax(pred, axis=2)
    pred = tf.cast(pred, dtype=real.dtype)
    accuracies = tf.equal(real, pred)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


""" training """

@tf.function(input_signature=[
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),  # (batch_size, seq_len)
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
])
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    with tf.GradientTape() as tape:
        preds, _ = model([inp, tar_inp], training = True)
        loss = loss_function(tar_real, preds)
        accuracy = accuracy_function(tar_real, preds)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, accuracy

def train(batch_size=32, num_iterations=2000, steps=200):
    """training loop (num_iterations has to be a multiple of steps, or it will be truncated)"""
    loss_history = []
    accuracy_history = []
    prev_time = time.time()
    time_elapsed = 0

    # load saved models
    if os.path.isfile("models/weights.h5"):
        model.load_weights("models/weights.h5")

    print("Training...")

    for i in range(0, num_iterations, steps):
        for _ in tqdm(range(steps)):
            inp, tar = get_batch(batch_size)
            loss, accuracy = train_step(inp, tar)
            loss_history.append(loss.numpy().mean())
            accuracy_history.append(accuracy.numpy().mean())

            time_elapsed += time.time() - prev_time
            prev_time = time.time()

        print(f"Iteration {i + steps}/{num_iterations}. Loss: {loss_history[-1]}. Time elapsed: {format_time(time_elapsed)}\n")
        # save checkpoints
        model.save_weights("models/weights.h5")
        model.save_weights(f"models/weights{i + steps}.h5")

        # plot a graph that will show how our loss varied with time
        plt.plot(loss_history)
        plt.plot(accuracy_history)
        plt.title("Training Progress")
        plt.xlabel("Iterations")
        plt.legend(["Loss", "Accuracy"])
        plt.savefig(os.path.join("./plots/TrainingProgress"))
        # plt.show()
        plt.close()

        key, context = get_batch(1, is_training=False)
        open(f"generated/{i + steps}.abc", "w").write(generate(key[0], context[0]))


""" inference """

def generate(keys, inp, max_length = targets.shape[-1]):
    encoder_input = np.concatenate(input_tokenizer.tokenize(inp).numpy(), -1)
    encoder_input = np.expand_dims(encoder_input, 0)

    # initialize start token
    target = np.array([2]) # 2 - <BOS>

    for i in range(max_length):
        prediction, _ = model([encoder_input, np.expand_dims(target, 0)], training=False)
        prediction = prediction[:, -1, :]  # we only need the last timestep to append it to the target. shape: [batch_size, vocab_size]
        prediction = tf.random.categorical(prediction, 1).numpy().squeeze()  # shape: []
        if prediction == 3: # 3 - <EOS>
            break
        target = np.append(target, prediction)

    target = target[1:]  # remove start token
    target = target_tokenizer.detokenize(target).numpy().decode("utf-8")
    notes = target.replace(" <UNK>", "").replace(" <PAD>", "").replace(" <BOS>", "").replace("!", "!\n").replace("\u2581", "")
    return keys + notes

def generate_from_saved_weights(num_samples=5):
    for weights in os.listdir("./models"):
        model.load_weights(f"./models/{weights}")
        keys, inps = get_batch(num_samples, is_training=False)
        for idx, (key, inp) in enumerate(zip(keys, inps)):
            song = generate(key, inp)
            open(f"{weights}-{idx}.abc", "w").write(song)
            os.system(f"abc2midi {weights}-{idx}.abc -o {weights}-{idx}.mid")
            os.system(f"move {weights}-{idx}.abc ./generated/")
            os.system(f"move {weights}-{idx}.mid ./generated/")


""" deployment"""

def create_tflite_model():
    generator = Generator(model, targets.shape[-1])
    converter = tf.lite.TFLiteConverter.from_keras_model(generator)
    output = converter.convert()
    open("generator.tflite", "wb").write(output)

def run_tflite_model(inp):
    tflite_model = open("generator.tflite", "rb").read()
    interp = interpreter.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    print(input_details)
    output_details = interp.get_output_details()
    print(output_details)
    interp.set_tensor(input_details[0]['index'], inp)
    interp.invoke()
    output_data = interp.get_tensor(output_details[0]['index'])
    return output_data


if __name__ == "__main__":
    model.load_weights("models/weights.h5")

    # g = Generator(model, targets.shape[-1])
    # keys, inps = get_batch(5, is_training=False)
    # out = g(inps[0]).numpy().astype(np.int32)
    # print(out)
    # tokens = open("vocab.txt", "r", encoding="utf-8").read().splitlines()
    # out = "".join([tokens[i] for i in out]).replace("!", "!\n")
    # print(keys[0] + out)

    # train()
    # generate_from_saved_weights()

    create_tflite_model()

    keys, inps = get_batch(5, is_training=False)
    out = run_tflite_model(inps[0])
    print(out)
    tokens = open("vocab.txt", "r", encoding="utf-8").read().splitlines()
    out = "".join([tokens[i] for i in out]).replace("!", "!\n")
    print(keys[0] + out)

    pass
