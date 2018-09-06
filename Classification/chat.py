# -*- coding:utf-8 -*-
import os, sys 
from search_es import _do_query_use_file_info, connet_es
import random 

es = connet_es()
print(es)
print("--- start conversation [push Cntl-D to exit] ------")

while True:
    try:
        input_str = raw_input("U:")
        # print(input_str)
    except:
        print('Errors on the format of the input ')
        break
    # search
    candidates = _do_query_use_file_info(es, input_str) 
    if candidates is None or len(candidates) < 1:
        sys_output = (0., "something alse?")
    else:
        sys_output = random.sample(candidates, 1)[0]
    print('>> \t {} \t{} :S'.format(sys_output[1], sys_output[0]))

     
