# coding: utf-8
import os

join = os.path.join
path = os.path

OUTPUT_PATH = 'output'
RESULT_PATH = 'result'
SUMMARY_PATH = 'summary'
CKPT_PATH = 'ckpt'


def _create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


paths = [OUTPUT_PATH, RESULT_PATH, SUMMARY_PATH]

[_create_path(p) for p in paths]
