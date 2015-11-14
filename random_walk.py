# Implementation of
# Xianpei Han 2011, Collective entity linking in web text: a graph-based method
# Data using the example in the paper

import numpy as np
import networkx as nx


MENTION = 'mention'
ENTITY = 'entity'

### Initial data

Bm = 'Bulls(m)'
SJm = 'Space Jam(m)'
Jm = 'Jordan(m)'

SJe = 'Space Jam'
Be = 'Bull'
CBe = 'Chicago Bulls'
MIJe = 'Michael I. Jordan'
MJe = 'Michael Jordan'
MBJe = 'Michael B. Jordan'

g = nx.DiGraph()
g.add_edges_from([(Bm, CBe, {'w': 0.13}),
                  (Bm, Be, {'w': 0.01}),
                  (CBe, MJe, {'w': 0.82}), (MJe, CBe, {'w': 0.82}),
                  (MJe, SJe, {'w': 0.66}), (SJe, MJe, {'w': 0.66}),
                  (SJm, SJe, {'w': 0.20}),
                  (Jm, MIJe, {'w': 0.03}),
                  (Jm, MJe, {'w': 0.08}),
                  (Jm, MBJe, {'w': 0.12})])

g.node[Bm]['type'] = MENTION
g.node[SJm]['type'] = MENTION
g.node[Jm]['type'] = MENTION

g.node[SJe]['type'] = ENTITY
g.node[Be]['type'] = ENTITY
g.node[CBe]['type'] = ENTITY
g.node[MIJe]['type'] = ENTITY
g.node[MJe]['type'] = ENTITY
g.node[MBJe]['type'] = ENTITY

### Get initial importance vector

importance = {
    Bm: 0.3,
    Jm: 0.25,
    SJm: 0.45
}
node2id = {n: i
           for i, n in enumerate(g.nodes())}
N = len(g.nodes())
s = np.zeros((N, 1), dtype=np.float64)

for n, v in importance.items():
    s[node2id[n]] = v
print(node2id)
print(s)

### Get Transition matrix

T = np.zeros((N, N), dtype=np.float64)
for m in g.nodes():
    neighbors = g.neighbors(m)
    denom = sum(g[m][e]['w']
                for e in neighbors)
    for e in neighbors:
        T[node2id[e], node2id[m]] = g[m][e]['w'] / denom
        print(e, m, T[node2id[e], node2id[m]])

print(T)

## Random walk

s = np.matrix(s)
T = np.matrix(T)
I = np.matrix(np.eye(N, N, dtype=np.float64))

lmbda = 0.1

# r = s
# for i in xrange(5):
#     r = (1-lmbda) * T * r  # + lmbda * s
#     print('At iter {}'.format(i))
#     for n, i in node2id.items():
#         print(n, r[i][0, 0])

r = lmbda * np.linalg.inv(I - (1 - lmbda) * T) * s

for n, i in node2id.items():
    print(n, r[i][0, 0])

# print(np.linalg.inv(I - (1 - lmbda) * T))
