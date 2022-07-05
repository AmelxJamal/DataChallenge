import numpy as np
np.random.seed(20)
import pandas as pd
from collections import defaultdict

def compute_kmers(seq, k=3):
  kmers = []
  seq = seq.strip()
  for i in range(len(seq)):
    if(len(seq[i:i+k])==k):
      kmers.append(seq[i:i+k])
  return kmers

base_indices = {'A':0, 'C':1, 'T':2, 'G':3}

def compute_perms(chars, k):
  from itertools import product
  perms = [''.join(comb) for comb in product(chars, repeat=k)]
  return perms

def compute_val(kmer, base_indices):
  val = 0
  base = len(base_indices)
  for idx,elem in enumerate(kmer):
    val+=base_indices[elem]*(base**idx)
  return val

def compute_kmer_count(seq,perms,k):
  counts = defaultdict(lambda : 0)
  kmers = compute_kmers(seq, k)
  for kmer in kmers:
    counts[kmer] +=1
  return counts

def jaccard_similarity(a, b):
    a = set(a)
    b = set(b)

    intersection = len(a.intersection(b))
    union = len(a.union(b))

    return intersection / union

def compress_kmers(dict_count,k):
  compressed_dict = defaultdict(lambda : 0)
  least_occuring = []
  most_occuring = []
  # Compute highest occuring kmers
  for key in dict_count.keys():
    if dict_count[key] < k and dict_count[key]!=0:
      least_occuring.append(key)
    elif dict_count[key]!=0:
      most_occuring.append(key)
  
  #Compute Jaccard_similarity between least and most occuring
  for key in least_occuring:
    similarities = []
    for sec_key in most_occuring:
      similarities.append((sec_key, jaccard_similarity(key,sec_key)))
    # print(most_occuring)
    most_similar = sorted(similarities, key = lambda x: x[1], reverse=True)[0][0]
    compressed_dict[most_similar] += dict_count[key]
    # print(f' element {key} , \n sorted {sorted(similarities, key = lambda x: x[1], reverse=True)}, \n most_similar {most_similar}')
    # break
  for key in most_occuring:
    compressed_dict[key] += dict_count[key]
  
  return compressed_dict

def seq_compress(row):
    # print(row.Sequence)
    perms = compute_perms('ACTG',3)
    dict_count = compute_kmer_count(row, perms, 3)
    compressed = compress_kmers(dict_count, 3)
    compressed_list = []
    for key in perms:
      if key in compressed.keys():
        compressed_list.append(compressed[key])
      else:
        compressed_list.append(0)
    # print(len(compressed_list))
    return compressed_list



def normalize(row):
  row = (row - df_min) / (df_max - df_min)
  return row

def seq_transform(row):
    perms = compute_perms('ACTG',3)
    dict_count = compute_kmer_count(row.Sequence, perms, 3)
    new_list = []
    for key in perms:
      if key in dict_count.keys():
        new_list.append(dict_count[key])
      else:
        new_list.append(0)
    return new_list

