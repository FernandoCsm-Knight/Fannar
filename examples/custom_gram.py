"""Exemplo: o método sem nenhum modelo de rede — suas fontes, seus kernels, suas matrizes.

    python custom_gram.py
"""

import numpy as np

import fannar as fa

rng = np.random.default_rng(0)
labels = np.repeat(np.arange(4), 30)
forma = np.eye(4)[labels] * 3 + rng.standard_normal((120, 4))       # fonte 1
cor = np.eye(4)[labels][:, ::-1] * 2 + rng.standard_normal((120, 4))  # fonte 2

# 1. componentes com kernels diferentes e a composição do artigo
pipe = fa.KernelPipeline({
    "forma": fa.Component(fa.Linear(), fa.angular() >> fa.shift(1.0)),
    "cor": fa.Component(fa.Polynomial(degree=2), fa.angular() >> fa.shift(1.0)),
})
space = pipe(sources={"forma": forma, "cor": cor}, labels=labels)
print(pipe)
print("PSD:", space.is_psd(), "| posto de participação:", space.participation_rank())
print("5 vizinhos do objeto 0:", space.neighbors(0, 5))
print("contenção das classes em V_r:", round(space.containment(), 3))

# 2. uma das componentes substituída por uma matriz que você já tem
K_cor = cor @ cor.T
space2 = pipe(sources={"forma": forma}, grams={"cor": K_cor}, labels=labels)
print("a matriz pronta dá a mesma G que o kernel:", np.allclose(space2.G, fa.KernelPipeline({
    "forma": fa.Component(fa.Linear()), "cor": fa.Component(fa.Linear())})(sources={"forma": forma, "cor": cor}).G))

# 3. uma matriz PSD qualquer, direto
space3 = fa.GramSpace(fa.center(forma @ forma.T), labels=labels)
e = space3.energies()
print("E_ratio médio:", round(float(e["E_ratio"].mean()), 3), "| CKA entre 1 e 3:", round(space.cka(space3), 3))
