"""Main module for training the translation model to learn extracting CDDDs"""
import os
import time
import sys
import argparse
import json
import tensorflow as tf
from model_helper import build_models
from evaluation import eval_reconstruct, parallel_eval_qsar
from hyperparameters import add_arguments, create_hparams
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)
FLAGS = None

def train_loop(train_model=None, eval_model=None, encoder_model=None, hparams=None):
    """Main training loop function for training and evaluating.
    Args:
        train_model: The model used for training.
        eval_model: The model used evaluating the translation accuracy.
        encoder_model: The model used for evaluating the QSAR modeling performance.
        hparams: Hyperparameters defined in file or flags.
    Returns:
        None
    """
    #qsar_process = []
    test_acc = {}
    train_acc = {}
    with train_model.graph.as_default():
        #train_model.sess.run(train_model.model.iterator.initializer)
        step = train_model.model.initilize(
            train_model.sess,
            overwrite_saves=hparams.overwrite_saves)


    start_time = time.time()
    total_acc = 0
    total_loss = 0
    with train_model.graph.as_default():

        #train_model.model.restore(train_model.sess)
        step = train_model.model.train(train_model.sess)
        print('开始运行：',step)
    while step < 160000:
        with train_model.graph.as_default():
            step = train_model.model.train(train_model.sess)
        if (step+1) %2000 == 0:
            print(step)
            with train_model.graph.as_default():
                loss,acc = train_model.model.eval(train_model.sess)
            print('用时：',time.time()-start_time,'loss:',loss,'acc',acc)
            train_acc[step] = acc

            start_time = time.time()
        if (step+1) % 4000 == 0:
            with train_model.graph.as_default():
                train_model.model.save(train_model.sess)
            with eval_model.graph.as_default():
                eval_model.model.restore(eval_model.sess)
                for i in range(100):
                    loss,acc = eval_model.model.eval(eval_model.sess)
                    total_acc = total_acc + acc
                    total_loss = total_loss+loss

            print(step,'mean_loss:',total_loss/100,'mean_acc:',total_acc/100)
            test_acc[step] = total_acc/100
            total_acc = 0
            total_loss = 0
    np.save('test_acc.npy',np.array(test_acc))
    np.save('train_acc.npy', np.array(train_acc))
    print('Done!!!!!!!')
                   
def main(unused_argv):
    """Main function that trains and evaluats the translation model"""
    hparams = create_hparams(FLAGS)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    train_model = build_models(hparams)
    test_model = build_models(hparams,'EVAL')
    encode_model = build_models(hparams,'ENCODE')
    train_loop(train_model=train_model,eval_model = test_model,encoder_model = encode_model, hparams=hparams)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    FLAGS, UNPARSED = PARSER.parse_known_args()
    main(UNPARSED)
    
