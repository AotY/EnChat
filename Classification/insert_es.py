#-*-coding:utf8-*-
"""
Read data file, and then put them into ES.

"""
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch import exceptions
import traceback
import datetime
import sys, os

reload(sys) #
sys.setdefaultencoding('utf-8')

def _create_index(es, index_name="response", doc_type_name="rcqa"):
    create_index_body = {
	    "settings": {
		    "similarity" : {
	               #"esserverbook_dfr_similarity" : {
                       #   "type" : "DFR",
                       #   "basic_model" : "g",
                       #   "after_effect" : "l",
                       #   "normalization" : "h2",
                       #   "normalization.h2.c" : "2.0"
                      #} 
                      "esserverbook_ib_similarity" : {
                          "type" : "IB",
                          "distribution" : "ll",
                          "lambda" : "df",
                          "normalization" : "z",
                          "normalization.z.z" : "0.25"
                      }
		  }
		},
		"mappings" : {
		    "rcqa": {
                         '_all': {
                                'enabled': 'false'
                            },
                         "properties": {
	                      "query": {
			            'type': 'text',
                                    'similarity': 'esserverbook_ib_similarity'
			           },
                              "comments": {
                                    'type': 'text',
                                    'similarity': 'esserverbook_ib_similarity'
                              }
                            }
                       }
		   }
	}
    # es.indices.delete(index=index_name)
    if es.indices.exists(index=index_name) is not True:
        create_index = es.indices.create(index=index_name, body=create_index_body)

def _save_data(es, input_file):
    all_data = list()
    count = 0
    ex_c = 0
    with open(input_file) as f_r:
        for line in f_r:
            count += 1
            items = line.strip().split('#EOS#')
            try:
                q = items[0].strip().decode('utf-8')
                coms = (" #EOS# ".join(items[1:])).decode('utf-8')
            except:
                ex_c += 1
                continue
            all_data.append({
                    '_index': 'response',
                    '_type': 'rcqa',
                    '_source': {
                        'query': q,
			'comments': coms,
                      }
                })
            if len(all_data) == 50000:
                success, _ = bulk(es, all_data, index='response', raise_on_error=True)
                all_data = list()
                print('{1}: finish {0}'.format(count, input_file))
    if len(all_data) != 0:
        success, _ = bulk(es, all_data, index='response', raise_on_error=True)
        all_data = list()
        print('{1}: finish {0}'.format(count, input_file))
    print('{0}: finish all'.format(input_file))
    print(ex_c)

def _insert_data(es, dir):
    start_time = datetime.datetime.now()
    files = [ dir ]#os.listdir(dir)
    for file in files:
        _save_data(es, os.path.join(dir, file))
    cost_time = datetime.datetime.now() - start_time
    print('all cost time{0}'.format(cost_time))

def _main():
    if len(sys.argv) != 2:
        print('need file argument')
        return 
    es = Elasticsearch(hosts=["127.0.0.1:9200"], timeout=500)
    try:
        _create_index(es)
    except exceptions.RequestError:
        print(traceback.format_exc())
    _insert_data(es, sys.argv[1]);

if __name__ == '__main__':
    _main()
