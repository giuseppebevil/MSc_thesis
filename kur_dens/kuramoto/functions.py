import numpy as np

def modularity_louvain_und(W, gamma=1, hierarchy=False, seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges. 
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups. 

    The Louvain algorithm is a fast and accurate community detection 
    algorithm (as of writing). The algorithm may also be used to detect
    hierarchical community structure.

    Input:      W           undirected (weighted or binary) connection matrix.
                gamma,        modularity resolution parameter
                            gamma>1:    detects smaller modules
                            0<=gamma<1:    detects larger modules
                            gamma=1:    no scaling of module size (default)
                hierarchy    enables hierarchical output, false by default
                seed,        random seed. Default None, seed from /dev/urandom

    Outputs:    1. Classic
                       Ci,     community structure
                       Q,      modularity
                2. Hierarchical (if h=1)
                       Ci_h,   community structure at each hierarchy
                               (access as Ci_h{1}, Ci_h{2}, ...)
                       Q_h,    modularity at each hierarhcy
                               (access as Q_h{1}, Q_h{2}, ...)

    Note: Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    '''
    np.random.seed(seed)

    n = len(W)  # number of nodes
    s = np.sum(W)  # weight of edges
    h = 0  # hierarchy index
    ci = []
    ci.append(np.arange(n) + 1)  # hierarchical module assignments
    q = []
    q.append(-1)  # hierarchical modularity values
    n0 = n

    while True:
        if h > 300:
            raise BCTParamError('Modularity Infinite Loop Style B.  Please '
                                'contact the developer with this error.')
        k = np.sum(W, axis=0)  # node degree
        Km = k.copy()  # module degree
        Knm = W.copy()  # node-to-module degree

        m = np.arange(n) + 1  # initial module assignments

        flag = True  # flag for within-hierarchy search
        it = 0
        while flag:
            it += 1
            if it > 1000:
                raise BCTParamError('Modularity Infinite Loop Style C.  Please '
                                    'contact the developer with this error.')
            flag = False

            # loop over nodes in random order
            for i in np.random.permutation(n):
                ma = m[i] - 1
                # algorithm condition
                dQ = ((Knm[i, :] - Knm[i, ma] + W[i, i]) -
                      gamma * k[i] * (Km - Km[ma] + k[i]) / s)
                dQ[ma] = 0

                max_dq = np.max(dQ)  # find maximal modularity increase
                if max_dq > 1e-10:  # if maximal increase positive
                    j = np.argmax(dQ)  # take only one value
                    # print max_dq,j,dQ[j]

                    Knm[:, j] += W[:, i]  # change node-to-module degrees
                    Knm[:, ma] -= W[:, i]

                    Km[j] += k[i]  # change module degrees
                    Km[ma] -= k[i]

                    m[i] = j + 1  # reassign module
                    flag = True

        _, m = np.unique(m, return_inverse=True)  # new module assignments
        # print m,h
        m += 1
        h += 1
        ci.append(np.zeros((n0,)))
        for i, mi in enumerate(m):  # loop through initial module assignments
            # print i,mi,m[i],h
            # print np.where(ci[h-1]==i+1)
            ci[h][np.where(ci[h - 1] == i)] = mi  # assign new modules

        n = np.max(m)  # new number of modules
        W1 = np.zeros((n, n))  # new weighted matrix
        for i in range(n):
            for j in range(n):
                # pool weights of nodes in same module
                wp = np.sum(W[np.ix_(m == i + 1, m == j + 1)])
                W1[i, j] = wp
                W1[j, i] = wp
        W = W1

        q.append(0)
        # compute modularity
        q[h] = np.trace(W) / s - gamma * np.sum(np.dot(W / s, W / s))
        if q[h] - q[h - 1] < 1e-10:  # if modularity does not increase
            break

    ci = np.array(ci) + 1
    if hierarchy:
        ci = ci[1:-1]
        q = q[1:-1]
        return ci, q
    else:
        return ci[h - 1], q[h - 1]

def makeevenCIJ(n, k, sz_cl):
    '''
    This function generates a random, directed network with a specified 
    number of fully connected modules linked together by evenly distributed
    remaining random connections.

    Inputs:     N,      number of vertices (must be power of 2)
                K,      number of edges
                sz_cl,  size of clusters (power of 2)

    Outputs:    CIJ,    connection matrix

    Notes:  N must be a power of 2.
            A warning is generated if all modules contain more edges than K.
            Cluster size is 2^sz_cl;
    '''
    # compute number of hierarchical levels and adjust cluster size
    mx_lvl = int(np.floor(np.log2(n)))
    sz_cl -= 1

    # make a stupid little template
    t = np.ones((2, 2)) * 2

    # check n against the number of levels
    Nlvl = 2 ** mx_lvl
    if Nlvl != n:
        print("Warning: n must be a power of 2")
    n = Nlvl

    # create hierarchical template
    for lvl in range(1, mx_lvl):
        s = 2 ** (lvl + 1)
        CIJ = np.ones((s, s))
        grp1 = range(int(s / 2))
        grp2 = range(int(s / 2), s)
        ix1 = np.add.outer(np.array(grp1) * s, grp1).flatten()
        ix2 = np.add.outer(np.array(grp2) * s, grp2).flatten()
        CIJ.flat[ix1] = t  # numpy indexing is teh sucks :(
        CIJ.flat[ix2] = t
        CIJ += 1
        t = CIJ.copy()

    CIJ -= (np.ones((s, s)) + mx_lvl * np.eye(s))

    # assign connection probabilities
    CIJp = (CIJ >= (mx_lvl - sz_cl))

    # determine nr of non-cluster connections left and their possible positions
    rem_k = k - np.size(np.where(CIJp.flatten()))
    if rem_k < 0:
        print("Warning: K is too small, output matrix contains clusters only")
        return CIJp
    a, b = np.where(np.logical_not(CIJp + np.eye(n)))

    # assign remK randomly dstributed connections
    rp = np.random.permutation(len(a))
    a = a[rp[:rem_k]]
    b = b[rp[:rem_k]]
    for ai, bi in zip(a, b):
        CIJp[ai, bi] = 1

    return np.array(CIJp, dtype=int)

