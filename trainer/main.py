import resnet
import input_data
import argparse
import tensorflow as tf
import numpy as np
import logging
import sys
import json
from tensorflow.python.lib.io import file_io
import os

def parse_model_config(json_file):
    #the 'open' function can't support goole cloud platform
    with file_io.FileIO(json_file, 'r') as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, choices=['gan', 'resn'], default='resn')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learn_rate', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--l2_reg', type=float, default=0.0001)
    parser.add_argument('--down_weight', type=float, default=1.0)
    parser.add_argument('--keep_prob', type=float, default=0.8)
    parser.add_argument('--op_alg', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--train_file_prefix', type=str)
    parser.add_argument('--unlabel_file_prefix', type=str)
    parser.add_argument('--test_file_prefix', type=str, default=None)
    parser.add_argument('--chunk_num', type=int, default=4)
    parser.add_argument('--unlabel_chunk_num', type=int)
    parser.add_argument('--gan_gen_iter', type=int, default=5)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--job_dir', type=str)
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--summary_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--mode', type=str, choices=['test', 'train'], default='train')

    args = parser.parse_args()
    if args.job_dir is not None:
        os.makedirs(args.job_dir)
        if args.summary_dir is None:
            args.summary_dir = '{}/summary'.format(args.job_dir)
            os.makedirs(args.summary_dir)
        if args.model_dir is None:
            args.model_dir = '{}/model'.format(args.job_dir)
            os.makedirs(args.model_dir)
        if args.log_file is None:
            args.log_file = '{}/run.log'.format(args.job_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log_file_stream=file_io.FileIO(args.log_file,'a')
    fh = logging.StreamHandler(log_file_stream)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    model_config = parse_model_config(args.model_config)
    logging.info('train_config: {:s}'.format(args))
    logging.info('model_config: {:s}'.format(json.dumps(model_config)))

    if args.alg == 'resn':
        with tf.Session() as sess:
            if args.mode == 'train':
                dataset = input_data.TfRecordDataset(
                        args.train_file_prefix, args.chunk_num, val_size = 1,
                        test_file_prefix = args.test_file_prefix)
                resn_ = resnet.Resnet(sess, dataset, train_config=args, model_config=model_config)
                resn_.train()
            elif args.mode == 'test':
                dataset = input_data.TfRecordDataset(test_file_prefix = args.test_file_prefix)
                resn_ = resnet.Resnet(sess, dataset, train_config=args, model_config=model_config)
                resn_.predict(args.output_dir, args.model_path)

if __name__ == '__main__':
    main()
