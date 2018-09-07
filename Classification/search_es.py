#-*-coding:utf8-*-
"""
Read data file, and then put them into ES.

Policy:
    (1) taking the stopwords as should_term,
    (2) taking the non-stopwords as must_term,
    Making search_query based on the above policies.

"""
from __future__ import division
from __future__ import print_function


from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch import exceptions
import sys
import json
import traceback

# load stopwords_en
STOPWORDS = {}
with open('stopwords_en.txt') as f:
    for line in f:
        STOPWORDS[line.strip()] = 1
# end of 

def _release_es_query_by_raw_query(raw_query):
    '''
    raw_query, it is the input query which will be searched from the db.
    Function,
        According to the raw_query, this fun is used to make the es_query in DSL format.
    Policy:
        (1) taking the stopwords as should_term,
        (2) taking the non-stopwords as must_term,
        Making search_query based on the above policies.

    Term Weight Policy:
        (1) The simplest way, uniform, set equal weight for each term
        (2) ?
    '''
    
    # according the the raw query and the stopwords, obtain the must_term and should_term
    raw_term = raw_query.strip().split()
    new_term = []
    for term in raw_term:
        if term in STOPWORDS:
            continue
        else:
            new_term.append(term)
    # making elastic_search query
    
    ret_obj = {}
    #ret_obj['must'] = must_term
    #ret_obj['should'] = should_term
    #ret_obj['score'] = term_score

    # es_query = {'query': {'bool': bool_query}}
    #es_query = {'query': {'match':{'query': raw_query}}}
    es_query = {'query': {'match':{'query': ' '.join(new_term)}}}
    #print(es_query)
    return raw_query, es_query, json.dumps(ret_obj, ensure_ascii=False) 

def _do_query_use_file_info(es, raw_query):
    raw_query, query, all_score = _release_es_query_by_raw_query(raw_query.strip())
    res = es.search(index='response', doc_type='rcqa', body=query, size=5)

    if (len(res['hits']['hits']) == 0):
        #print('len(res["hits"]["hits"]) == 0')
        #print("{0}".format(raw_query))
        return None 
    #print("query: {0}".format(raw_query))

    candidates = []
    for item in res['hits']['hits']:
        # print("{0}".format(raw_query))
        try:
            #print("{0}\t{1}\n\t{2}".format(item['_score'], item['_source']['query'].strip(), item['_source']['answer'].strip()))
            candidates.append((item['_score'], item['_source']['comments'].strip()))
        except:
            #print(traceback.format_exc())
            #print(item['_source']['name'])
            pass
    #print('\r\n')
    outputs = []
    for score, coms in candidates:
        coms = coms.strip().split('#EOS#')
        coms = [(score, com) for com in coms]
        outputs.extend(coms)
    return outputs    

def connet_es(es_ip, es_port):
    es = Elasticsearch(hosts=[es_ip + ":" + es_port], timeout=5000)
    return es 

def _main():
    if len(sys.argv) != 2:
        print('argv error')
        return
    else:
        print('argv[1] = {0}'.format(sys.argv[1]))
    # es = Elasticsearch(hosts=["127.0.0.1:9200"], timeout=5000)
    es = connet_es('127.0.0.1', '9200')
    with open(sys.argv[1]) as f_r:
        for item in f_r:
            try:
                _do_query_use_file_info(es, item)
            except:
                print(traceback.format_exc())
    
if __name__ == '__main__':
    _main()
