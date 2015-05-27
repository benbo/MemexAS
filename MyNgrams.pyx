from collections import defaultdict
join_spaces = " ".join
def ngrams(tokens, int n=4):
    cdef Py_ssize_t i, n_tokens
    n_tokens = len(tokens)
    for i in xrange(0,n_tokens-n):
        yield join_spaces(tokens[i:i+n])


def ngramsRange(tokens, int MIN_N, int MAX_N):
    cdef Py_ssize_t i, j, n_tokens

    count = defaultdict(int)

    n_tokens = len(tokens)
    for i in xrange(n_tokens):
        for j in xrange(i+MIN_N, min(n_tokens, i+MAX_N)+1):
            count[join_spaces(tokens[i:j])] += 1

    return count
    
 
