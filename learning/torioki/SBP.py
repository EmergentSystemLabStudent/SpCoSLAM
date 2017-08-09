#!/usr/bin/python
#-*- coding:utf8 -*-
"""
DBDA FIGURE 5.1
"""

import os


def read_result(diric):
  file_dir = os.chdir(diric)
  f = open('SBP.txt')
  line = f.readline() # 1行を文字列として読み込む(改行文字も含まれる)
  place_num = int(line)
  return place_num

