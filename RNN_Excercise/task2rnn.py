# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 17:04:33 2018

@author: Nihar Raut
"""

import time
import numpy as np
import tensorflow as tf

# Notebook auto reloads code. (Ref: http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython)
with open('data/tinyshakespeare.txt', 'r') as f:
    text=f.read()
# length of text is the number of characters in it
print('Length of text: {} characters'.format(len(text)))
# and let's get a glance of what the text is
print(text[:500])

# The unique characters in the file
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))


# Creating a mapping from unique characters to indices
vocab_to_ind = {c: i for i, c in enumerate(vocab)}
ind_to_vocab = dict(enumerate(vocab))
text_as_int = np.array([vocab_to_ind[c] for c in text], dtype=np.int32)

# We mapped the character as indexes from 0 to len(vocab)
for char,_ in zip(vocab_to_ind, range(20)):
    print('{:6s} ---> {:4d}'.format(repr(char), vocab_to_ind[char]))
# Show how the first 10 characters from the text are mapped to integers
print ('{} --- characters mapped to int --- > {}'.format(text[:10], text_as_int[:10]))


def get_batches(array, n_seqs, n_steps):
    '''
    Partition data array into mini-batches
    input:
    array: input data
    n_seqs: number of sequences in a batch
    n_steps: length of each sequence
    output:
    x: inputs
    y: targets, which is x with one position shift
       you can check the following figure to get the sence of what a target looks like
    '''
    batch_size = n_seqs * n_steps
    n_batches = int(len(array) / batch_size)
    # we only keep the full batches and ignore the left.
    array = array[:batch_size * n_batches]
    array = array.reshape((n_seqs, -1))
    
    
    
    
batches = get_batches(text_as_int, 10, 10)
#x, y = next(batches)
print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])