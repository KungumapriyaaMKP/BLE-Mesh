"""
GNN-based fake news classifier for BLE Mesh PoC.

Architecture:
  Encoder  : frozen RoBERTa backbone (jy46604790/Fake-News-Bert-Detect, already cached)
  Graph    : query node  +  N labeled anchor nodes
  Edges    : cosine-similarity between CLS embeddings (top-K per node, symmetrised)
  GNN      : 2-layer GCN (Kipf & Welling 2017)
  Head     : linear  ->  softmax over {Fake, True}

At inference the new BLE message is appended as the last node; its GCN output is read.
The GCN is trained once on the anchor graph (300 epochs, ~5 s on CPU) during warm-up.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ENCODER_NAME = "jy46604790/Fake-News-Bert-Detect"
EMBED_DIM = 768
K_NEIGHBORS = 7
GCN_HIDDEN = 256
TRAIN_EPOCHS = 500
LABEL_MAP = {0: "Fake", 1: "True"}

# Expanded anchor dataset – diverse domains, short facts, Indian context,
# news headlines, science, geography, health, tech, sports
ANCHORS = [
    # ── FAKE (label 0) ──────────────────────────────────────────────────────
    # Health misinformation
    ("Drinking bleach cures COVID-19, doctors secretly admit.", 0),
    ("Eating raw garlic every day prevents and cures all cancers.", 0),
    ("Vaccines contain microchips that track your location.", 0),
    ("Hospitals are secretly giving patients poison instead of medicine.", 0),
    ("Onion placed in a room absorbs and kills all viruses.", 0),
    # Tech / science fake
    ("5G towers are spreading a mind-control virus across cities.", 0),
    ("Scientists prove the Earth is actually flat, NASA confirms.", 0),
    ("Moon landing in 1969 was entirely staged in a Hollywood studio.", 0),
    ("Ancient aliens built the Egyptian pyramids, leaked NASA files show.", 0),
    ("NASA confirms humans successfully landed on Mars in 2025.", 0),
    ("Scientists have discovered that time travel is now possible.", 0),
    ("Government replaced all birds with surveillance drones in 2001.", 0),
    # Political / conspiracy fake
    ("Bill Gates admitted to planning a global population reduction program.", 0),
    ("The government is secretly controlling weather using HAARP machines.", 0),
    ("World leaders are secretly reptilian aliens disguised as humans.", 0),
    ("Elections are rigged by microchips hidden inside voting machines.", 0),
    # Indian fake news
    ("India banned all imported foreign goods starting next month.", 0),
    ("Indian government announced free gold for all citizens below poverty line.", 0),
    ("Taj Mahal was originally a Hindu temple called Tejo Mahalaya, government confirms.", 0),
    ("India will become the richest country in the world by 2025, IMF confirms.", 0),
    ("Cricket is officially banned in India due to match fixing.", 0),
    # General sensational fake
    ("Scientist discovers immortality pill, government suppressing the news.", 0),
    ("Chocolate and sugar officially declared more addictive than cocaine.", 0),
    ("Sunlight causes instant cancer, WHO issues global warning.", 0),
    ("Drinking hot water cures diabetes permanently within seven days.", 0),

    # ── TRUE (label 1) ──────────────────────────────────────────────────────
    # Geography and general facts
    ("India is home to the Taj Mahal, located in Agra.", 1),
    ("The Taj Mahal is a UNESCO World Heritage Site built by Mughal emperor Shah Jahan.", 1),
    ("Paris is the capital city of France.", 1),
    ("The Great Wall of China stretches thousands of kilometres across northern China.", 1),
    ("Mount Everest is the highest mountain in the world.", 1),
    ("Water boils at 100 degrees Celsius at standard atmospheric pressure.", 1),
    ("The Earth orbits the Sun once every 365 days.", 1),
    ("The human body has 206 bones in total.", 1),
    # Indian news and facts
    ("India launched Chandrayaan-3 successfully to the Moon in 2023.", 1),
    ("India's GDP growth rate was among the highest in the world in 2023.", 1),
    ("The Indian Space Research Organisation is headquartered in Bengaluru.", 1),
    ("India won the ICC Cricket World Cup in 2011 under MS Dhoni's captaincy.", 1),
    ("India has the largest population in the world as of 2023.", 1),
    ("Virat Kohli scored his 50th ODI century during an international match.", 1),
    # Tech and science news
    ("Apple released the iPhone 16 in September 2024.", 1),
    ("Microsoft acquired Activision Blizzard for approximately 69 billion dollars.", 1),
    ("Scientists detected gravitational waves from two colliding black holes.", 1),
    ("Researchers published new Alzheimer's treatment findings in the journal Nature.", 1),
    ("The James Webb Space Telescope captured its first full-colour images in 2022.", 1),
    # World and economy news
    ("The Federal Reserve raised interest rates by 0.25 percent.", 1),
    ("The United Nations held an emergency summit on climate change.", 1),
    ("WHO declared COVID-19 a global pandemic in March 2020.", 1),
    ("Global electric vehicle sales surpassed 10 million units in 2023.", 1),
    ("The Supreme Court issued a ruling on the immigration case.", 1),
    ("The unemployment rate fell to 3.8 percent last quarter.", 1),
]


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def _normalize_adj(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation: D^{-1/2} A D^{-1/2}"""
    deg = A.sum(dim=1).clamp(min=1e-6)
    d = deg ** -0.5
    return d.unsqueeze(1) * A * d.unsqueeze(0)


def _build_adj(embeddings: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity graph with top-K edges + self-loops, then normalised."""
    n = embeddings.size(0)
    norms = F.normalize(embeddings, p=2, dim=1)
    sim = (norms @ norms.T + 1.0) / 2.0          # scale to [0, 1]

    A = torch.zeros_like(sim)
    k = min(K_NEIGHBORS, n - 1)
    for i in range(n):
        row = sim[i].clone()
        row[i] = -1.0                              # exclude self from top-K
        top_idx = torch.topk(row, k).indices
        A[i, top_idx] = sim[i, top_idx]

    A = (A + A.T) / 2.0                           # symmetrise
    A = A + torch.eye(n)                           # self-loops
    return _normalize_adj(A)


# ---------------------------------------------------------------------------
# GCN model
# ---------------------------------------------------------------------------

class _GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        return F.relu(A_hat @ self.W(H))


class _FakeNewsGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = _GCNLayer(EMBED_DIM, GCN_HIDDEN)
        self.gcn2 = _GCNLayer(GCN_HIDDEN, GCN_HIDDEN // 2)
        self.out  = nn.Linear(GCN_HIDDEN // 2, 2)
        self.drop = nn.Dropout(0.3)

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.gcn1(H, A_hat))
        h = self.drop(self.gcn2(h, A_hat))
        return self.out(h)


# ---------------------------------------------------------------------------
# Main classifier class
# ---------------------------------------------------------------------------

class BLEMeshGNNClassifier:
    """Singleton – loaded once, reused for every BLE message."""

    def __init__(self):
        self._tok  = None
        self._enc  = None
        self._gcn  = None
        self._anc_emb   = None   # (N, 768) – frozen after training
        self._anc_label = None   # (N,) long

    # ------------------------------------------------------------------
    def _load_encoder(self):
        if self._tok is not None:
            return
        print("  Loading RoBERTa encoder (cached)...")
        self._tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
        self._enc = AutoModel.from_pretrained(ENCODER_NAME, ignore_mismatched_sizes=True)
        self._enc.eval()

    @torch.no_grad()
    def _embed(self, texts: list) -> torch.Tensor:
        inputs = self._tok(
            texts, return_tensors="pt",
            truncation=True, padding=True, max_length=512,
        )
        out = self._enc(**inputs)
        return out.last_hidden_state[:, 0, :]     # CLS token  (B, 768)

    # ------------------------------------------------------------------
    def _train_gcn(self):
        texts  = [a[0] for a in ANCHORS]
        labels = torch.tensor([a[1] for a in ANCHORS], dtype=torch.long)

        print(f"  Embedding {len(texts)} anchor nodes...")
        embs = self._embed(texts).detach()
        self._anc_emb   = embs
        self._anc_label = labels

        A_hat = _build_adj(embs)

        print(f"  Training GCN ({TRAIN_EPOCHS} epochs on anchor graph)...")
        self._gcn = _FakeNewsGCN()
        opt = torch.optim.Adam(self._gcn.parameters(), lr=5e-3, weight_decay=5e-4)

        self._gcn.train()
        for epoch in range(TRAIN_EPOCHS):
            opt.zero_grad()
            logits = self._gcn(embs, A_hat)
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()

        self._gcn.eval()
        with torch.no_grad():
            preds = self._gcn(embs, A_hat).argmax(dim=1)
        acc = (preds == labels).float().mean().item() * 100
        print(f"  Anchor-graph training accuracy: {acc:.1f}%")

    # ------------------------------------------------------------------
    def load(self):
        self._load_encoder()
        self._train_gcn()

    def classify(self, text: str) -> dict:
        if self._gcn is None:
            self.load()

        query_emb = self._embed([text])                         # (1, 768)
        all_embs  = torch.cat([self._anc_emb, query_emb], dim=0)   # (N+1, 768)
        A_hat     = _build_adj(all_embs)

        with torch.no_grad():
            logits = self._gcn(all_embs, A_hat)

        query_logits = logits[-1]                               # query node
        probs        = F.softmax(query_logits, dim=0)
        pred_idx     = int(probs.argmax())
        confidence   = round(float(probs[pred_idx]) * 100, 2)

        return {
            "message":    text,
            "prediction": LABEL_MAP[pred_idx],
            "confidence": confidence,
        }


# ---------------------------------------------------------------------------
# Module-level singletons – same interface as the old classifier.py
# ---------------------------------------------------------------------------

_instance = BLEMeshGNNClassifier()


def load_classifier():
    _instance.load()


def classify(text: str) -> dict:
    return _instance.classify(text)
