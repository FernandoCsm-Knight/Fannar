"""Exemplo de texto: um transformer pequeno numa tarefa sintética de palavras-chave.

    pip install "fannar[torch]"
    python text_keywords.py

Cada "frase" é uma sequência de tokens com preenchimento (padding). A classe é dada por uma
palavra-chave que aparece em algum ponto da frase. O modelo imita a interface do Hugging Face
(`model(input_ids=..., attention_mask=...)` devolve um objeto com `.logits`), então o exemplo usa
`fannar.text.wrap_hf` exatamente como com um modelo real, sem precisar baixar nada.

Mostra `TextInterpreter` com as camadas descobertas sozinhas (os blocos do encoder) e a energia
por token de uma frase. Nos primeiros blocos ela se concentra na palavra-chave. No último, a
atenção já espalhou a informação por todos os tokens e a energia fica quase uniforme.
"""

import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from fannar.text import TextInterpreter, block_names, wrap_hf

PAD, VOCAB, KEYWORDS, LENGTH = 0, 50, [1, 2, 3], 16      # tokens 1, 2, 3 são as palavras-chave


def make_sentences(n, rng):
    """Frases de tamanho 8 a 16 com uma palavra-chave numa posição aleatória; o resto é preenchimento."""
    ids = np.full((n, LENGTH), PAD)
    y = rng.integers(0, len(KEYWORDS), n)
    where = np.zeros(n, dtype=int)
    for t in range(n):
        size = rng.integers(8, LENGTH + 1)
        ids[t, :size] = rng.integers(len(KEYWORDS) + 1, VOCAB, size)   # palavras comuns
        where[t] = rng.integers(0, size)
        ids[t, where[t]] = KEYWORDS[y[t]]
    return torch.from_numpy(ids), y, where


class TinyEncoder(nn.Module):
    """Embeddings + blocos de transformer numa ModuleList + cabeça, com a interface do Hugging Face."""

    def __init__(self, d=32, blocks=3, classes=3):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.position = nn.Parameter(torch.randn(LENGTH, d) * 0.02)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=64, dropout=0.0, batch_first=True)
            for _ in range(blocks))
        self.classifier = nn.Linear(d, classes)

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids) + self.position[: input_ids.shape[1]]
        pad = None if attention_mask is None else attention_mask == 0
        for block in self.encoder.layer:
            h = block(h, src_key_padding_mask=pad)
        m = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        pooled = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        return SimpleNamespace(logits=self.classifier(pooled))


rng = np.random.default_rng(0)
ids_tr, y_tr, _ = make_sentences(1500, rng)
ids_te, y_te, where_te = make_sentences(300, rng)

torch.manual_seed(0)
hf_like = TinyEncoder()
initial = copy.deepcopy(hf_like)
opt = torch.optim.Adam(hf_like.parameters(), 2e-3)
mask_tr, yt = (ids_tr != PAD).long(), torch.from_numpy(y_tr)
for epoch in range(20):
    for idx in torch.randperm(len(ids_tr)).split(64):
        opt.zero_grad()
        logits = hf_like(input_ids=ids_tr[idx], attention_mask=mask_tr[idx]).logits
        nn.functional.cross_entropy(logits, yt[idx]).backward()
        opt.step()
hf_like.eval()

# 1. o mesmo passo de um modelo do Hugging Face: embrulhar para que forward(ids) devolva os logits
model = wrap_hf(hf_like, pad_token_id=PAD)
reference = wrap_hf(initial.eval(), pad_token_id=PAD)
print("camadas sondadas:", block_names(model))

# 2. sem `layers=`: os blocos do encoder são encontrados sozinhos
interp = TextInterpreter(model, ids_te, y_te, reference_model=reference,
                         class_names=[f"kw{k}" for k in KEYWORDS])
report = interp.evaluate()
print(f"\nacurácia no teste: {report.meta['accuracy']:.3f}")
print(report.to_markdown())

# 3. energia por token de uma frase: os tokens viram os objetos
size = int((ids_te[0] != PAD).sum())
print(f"\nfrase 0: palavra-chave na posição {where_te[0]} de {size}; energia por token (normalizada):")
for layer in interp.layers:
    energy = interp.energy_map(ids_te[0], layer)[:size]       # ignora o preenchimento
    print(f"  {layer:22s}", np.round(energy / energy.max(), 2))

# 4. em quantas frases o token de maior energia é a palavra-chave, camada a camada?
chance = np.mean([1 / int((r != PAD).sum()) for r in ids_te[:100]])
print(f"\ntoken de maior energia = palavra-chave (100 frases; ao acaso ~{chance:.0%}):")
for layer in interp.layers:
    hits = [int(interp.energy_map(ids_te[t], layer)[: int((ids_te[t] != PAD).sum())].argmax()) == where_te[t]
            for t in range(100)]
    print(f"  {layer:22s} {np.mean(hits):.0%}")
