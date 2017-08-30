#-*- coding:utf-8 -*-
import os
import random
import codecs
import numpy as np
import tensorflow as tf # 0.12
from tensorflow.models.rnn.translate import seq2seq_model
import math
# import data_utils
# import seq2seq_model

###数据处理部分，可以参考tensorflow.models.rnn.translate.data_utils

#获取数据
#os.system('wget https://raw.githubusercontent.com/rustch3n/dgk_lost_conv/master/dgk_shooter_min.conv.zip')
#os.system('unzip dgk_shooter_min.conv.zip')

# 数据处理过程，及训练测试过程会用到的标记，先定义好。
# 特殊标记，用来填充标记对话
PAD = "__PAD__"
GO = "__GO__"
EOS = "__EOS__"  # 对话结束
UNK = "__UNK__"  # 标记未出现在词汇表中的字符
START_VOCABULART = [PAD, GO, EOS, UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def preprocess(convers_path):
    if not os.path.exists(convers_path):
        print('数据集不存在')
        exit()
    '''
    M 表示话语，E 表示分割，遇到M就吧当前对话片段加入临时对话集，遇到E就说明遇到一个中断或者交谈双方转换了
    再把临时对话集加入convs总对话集，一次加入一个对话集。
    '''
    convers = []  # conversation set
    with codecs.open(convers_path, encoding="utf8") as f:
        one_conv = []  # a complete conversation
        for line in f:
            line = line.strip('\n').replace('/', '')
            if line == '':
                continue
            if line[0] == 'E':
                if one_conv:
                    convers.append(one_conv)
                one_conv = []
            elif line[0] == 'M':
                one_conv.append(line.split(' ')[1])
    #print(convers[:2])
    return convers

def get_question_answer(convers):
    '''
    把影视对白分成问与答, 因为场景是聊天机器人，影视剧的台词也是一人一句对答的，所以这里需要忽略2种特殊情况:
    只有一问或者只有一答，以及问和答的数量不一致。
    '''
    questions = []        # 问
    answers = []   # 答
    for conv in convers:
        if len(conv) == 1:
            continue
        if len(conv) % 2 != 0:  # 奇数对话数, 转为偶数对话
            conv = conv[:-1]
        for i in range(len(conv)):
            if i % 2 == 0:
                questions.append(conv[i])
            else:
                answers.append(conv[i])
    return questions, answers

#训练集测试集划分，以及各自对应问答文件创建
def convert_seq2seq_files(questions, answers, TESTSET_SIZE=20000):
    # 创建文件
    train_enc = codecs.open('train.enc', 'w', encoding='utf-8')  # 问
    train_dec = codecs.open('train.dec', 'w', encoding='utf-8')  # 答
    test_enc = codecs.open('test.enc', 'w', encoding='utf-8')  # 问
    test_dec = codecs.open('test.dec', 'w', encoding='utf-8')  # 答

    # 选择20000数据作为测试数据
    print len(questions)
    test_index = random.sample([i for i in range(len(questions))], TESTSET_SIZE)
    print len(test_index)
    for i in range(len(questions)):
        if i in test_index:
            test_enc.write(questions[i] + '\n')
            test_dec.write(answers[i] + '\n')
        else:
            train_enc.write(questions[i] + '\n')
            train_dec.write(answers[i] + '\n')
        if i % 1000 == 0:
            print(len(range(len(questions))), '处理进度:', i)

    # 生成的*.enc文件保存了问题
    # 生成的*.dec文件保存了回答
    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()

'''
# 生成词汇表文件
# 图像识别、语音识别,原始输入数据本身就带有很强的样本关联性,对话或者叫语料往往是不具备这种强关联性的
# 创建词汇表，然后把对话转为向量形式
'''
def gen_vocabulary_file(input_file, output_file):
    print('开始创建词汇表...')
    vocabulary = {}
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        counter = 0
        for line in f:
            counter += 1
            tokens = [word for word in line.strip()]
            for word in tokens:
                if word in vocabulary:
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1
        vocabulary_list = START_VOCABULART + sorted(vocabulary, key=vocabulary.get, reverse=True)
        # 取前10000个常用汉字
        if len(vocabulary_list) > 10000:
            vocabulary_list = vocabulary_list[:10000]
        print(input_file + " 词汇表大小:", len(vocabulary_list))
        with codecs.open(output_file, "w", encoding='utf-8') as ff:
            for word in vocabulary_list:
                ff.write(word + "\n")


## 对话向量化表示
# 把对话字符串转为向量形式
def convert_to_vector(input_file, vocabulary_file, output_file):
    print("对话转向量...")
    tmp_vocab = []
    with codecs.open(vocabulary_file, "r",encoding='utf-8') as f:
        tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    # {'硕': 3142, 'v': 577, 'Ｉ': 4789, '\ue796': 4515, '拖': 1333, '疤': 2201 ...}
    output_f = open(output_file, 'w')
    with codecs.open(input_file, 'r',encoding='utf-8') as f:
        for line in f:
            line_vec = []
            for words in line.strip():
                line_vec.append(vocab.get(words, UNK_ID))
            output_f.write(" ".join([str(num) for num in line_vec]) + "\n")
    output_f.close()

def read_data(source_path, target_path, max_size=None):
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

def train():

    # word table 6000
    vocabulary_encode_size = 6000
    vocabulary_decode_size = 6000
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    layer_size = 256  # 每层大小
    num_layers = 3  # 层数
    batch_size = 64

    model = seq2seq_model.Seq2SeqModel(source_vocab_size=vocabulary_encode_size,
                                       target_vocab_size=vocabulary_decode_size,
                                       buckets=buckets,
                                       size=layer_size,
                                       num_layers=num_layers,
                                       max_gradient_norm=5.0,
                                       batch_size=batch_size,
                                       learning_rate=0.5,
                                       learning_rate_decay_factor=0.97,
                                       forward_only=False)

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'  # 防止 out of memory

    with tf.Session(config=config) as sess:
        # 恢复前一次训练
        ckpt = tf.train.get_checkpoint_state('.')
        if ckpt != None:
            print(ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        train_set = read_data("train_encode.vec", "train_decode.vec")
        test_set = read_data("test_encode.vec", "test_decode.vec")

        train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in
                               range(len(train_bucket_sizes))]

        loss = 0.0
        total_step = 0
        previous_losses = []
        # 一直训练，每过一段时间保存一次模型
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)

            loss += step_loss / 500
            total_step += 1

            print(total_step)
            if total_step % 500 == 0:
                print(model.global_step.eval(), model.learning_rate.eval(), loss)

                # 如果模型没有得到提升，减小learning rate
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # 保存模型
                checkpoint_path = "chatbot_seq2seq.ckpt"
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                loss = 0.0
                # 使用测试数据评估模型
                for bucket_id in range(len(buckets)):
                    if len(test_set[bucket_id]) == 0:
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print(bucket_id, eval_ppx)

def test():
    train_encode_vocabulary = 'train_encode_vocabulary'
    train_decode_vocabulary = 'train_decode_vocabulary'

    def read_vocabulary(input_file):
        tmp_vocab = []
        with open(input_file, "r") as f:
            tmp_vocab.extend(f.readlines())
        tmp_vocab = [line.strip() for line in tmp_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
        return vocab, tmp_vocab

    vocab_en, _, = read_vocabulary(train_encode_vocabulary)
    _, vocab_de, = read_vocabulary(train_decode_vocabulary)

    # 词汇表大小5000
    vocabulary_encode_size = 6000
    vocabulary_decode_size = 6000

    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    layer_size = 256  # 每层大小
    num_layers = 3  # 层数
    batch_size = 1

    model = seq2seq_model.Seq2SeqModel(source_vocab_size=vocabulary_encode_size,
                                       target_vocab_size=vocabulary_decode_size,
                                       buckets=buckets,
                                       size=layer_size,
                                       num_layers=num_layers,
                                       max_gradient_norm=5.0,
                                       batch_size=batch_size,
                                       learning_rate=0.5,
                                       learning_rate_decay_factor=0.99,
                                       forward_only=True)
    model.batch_size = 1
    with tf.Session() as sess:
        # 恢复前一次训练
        ckpt = tf.train.get_checkpoint_state('.')
        if ckpt != None:
            print(ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("没找到模型")

        while True:
            input_string = raw_input('me(Human) > ')
            # 退出
            if input_string == 'quit':
                exit()

            input_string_vec = []
            for words in input_string.strip():
                input_string_vec.append(vocab_en.get(words, UNK_ID))
            bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(input_string_vec)])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(input_string_vec, [])]},
                                                                             bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            if EOS_ID in outputs:
                outputs = outputs[:outputs.index(EOS_ID)]

            response = "".join([tf.compat.as_str(vocab_de[output]) for output in outputs])
            print('Robot > ' + response)


def main():
    convers_path = 'dgk_shooter_min.conv'

    convers = preprocess(convers_path)

    questions, answers = get_question_answer(convers)

    convert_seq2seq_files(questions, answers, TESTSET_SIZE=20000)

    gen_vocabulary_file("train.enc", "train_encode_vocabulary")
    gen_vocabulary_file("train.dec", "train_decode_vocabulary")

    convert_to_vector("train.enc", "train_encode_vocabulary", 'train_encode.vec')
    convert_to_vector("train.dec", "train_decode_vocabulary", 'train_decode.vec')
    convert_to_vector("test.enc", "train_encode_vocabulary", 'test_encode.vec')
    convert_to_vector("test.dec", "train_decode_vocabulary", 'test_decode.vec')

    train()
  #  test()

if __name__ == '__main__':
    main()