"""
GNN-based fake news classifier for BLE Mesh PoC.

Architecture:
  Encoder  : frozen RoBERTa backbone (jy46604790/Fake-News-Bert-Detect, cached)
  Dataset  : GonzaloA/fake_news (~72k articles, 0=Fake 1=True), HuggingFace
  Graph    : query node + 400 dataset anchors + 55 custom anchors (India/TN/CBE)
  Edges    : cosine-similarity top-K per node, symmetrised, normalised
  GNN      : 2-layer GCN (Kipf & Welling 2017)
  Head     : linear -> softmax over {Fake, True}

First run  : downloads dataset, embeds 455 anchors (~2 min), trains GCN, saves to disk.
Later runs : loads cached embeddings + weights — startup under 10 seconds.
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ENCODER_NAME       = "jy46604790/Fake-News-Bert-Detect"
EMBED_DIM          = 768
K_NEIGHBORS        = 10
GCN_HIDDEN         = 256
TRAIN_EPOCHS       = 600
N_ANCHORS_PER_CLASS = 200          # 200 Fake + 200 True = 400 anchor nodes
LABEL_MAP          = {0: "Fake", 1: "True"}

MODEL_DIR    = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH   = os.path.join(MODEL_DIR, "anchor_cache.pt")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "gcn_weights.pt")

DATASET_NAME = "GonzaloA/fake_news"   # 0 = Fake, 1 = True, ~72k samples

# ── Custom domain anchors ─────────────────────────────────────────────────────
# Added on top of dataset anchors to improve accuracy on Indian / TN / Coimbatore facts
CUSTOM_ANCHORS = [

    # ── INDIA — True ─────────────────────────────────────────────────────────
    ("Narendra Modi is the Prime Minister of India.", 1),
    ("PM of India is Modi.", 1),
    ("Droupadi Murmu is the President of India.", 1),
    ("The capital of India is New Delhi.", 1),
    ("India has 28 states and 8 union territories.", 1),
    ("India won independence from British rule on 15th August 1947.", 1),
    ("India is the world's largest democracy.", 1),
    ("The national currency of India is the Indian Rupee.", 1),
    ("The Reserve Bank of India is the central bank of India.", 1),
    ("India launched Chandrayaan-3 to the Moon in 2023.", 1),
    ("India has the largest population in the world as of 2023.", 1),
    ("India's space agency is called ISRO.", 1),
    ("The Supreme Court of India is located in New Delhi.", 1),
    ("Hindi and English are the official languages of the Indian government.", 1),

    # ── INDIA — Fake ─────────────────────────────────────────────────────────
    ("Rahul Gandhi is the current Prime Minister of India.", 0),
    ("India has 35 states as per the new 2024 constitution.", 0),
    ("India launched a crewed mission to Mars in 2024.", 0),
    ("The Indian government banned all social media platforms permanently.", 0),
    ("India's GDP overtook the USA to become number one in 2024.", 0),
    ("India declared itself a republic only in 2010.", 0),
    ("The capital of India was moved from Delhi to Mumbai in 2023.", 0),

    # ── TAMIL NADU — True ────────────────────────────────────────────────────
    ("M.K. Stalin is the Chief Minister of Tamil Nadu.", 1),
    ("The Chief Minister of Tamil Nadu is M.K. Stalin.", 1),
    ("Tamil Nadu CM is Stalin.", 1),
    ("The capital of Tamil Nadu is Chennai.", 1),
    ("Tamil Nadu has 38 districts.", 1),
    ("Tamil is the official language of Tamil Nadu.", 1),
    ("IIT Madras is located in Chennai, Tamil Nadu.", 1),
    ("Tamil Nadu was formerly known as Madras State.", 1),
    ("The Kaveri river flows through Tamil Nadu.", 1),
    ("Anna University is a technical university in Chennai, Tamil Nadu.", 1),
    ("Tamil Nadu is known as the land of temples.", 1),
    ("Chennai Super Kings is a cricket team based in Tamil Nadu.", 1),
    ("The DMK party is currently in power in Tamil Nadu.", 1),
    ("Tamil Nadu borders Kerala, Karnataka, and Andhra Pradesh.", 1),

    # ── TAMIL NADU — Fake ────────────────────────────────────────────────────
    ("Vijay is the Chief Minister of Tamil Nadu.", 0),
    ("CM of TN is Vijay.", 0),
    ("Rajinikanth became the Chief Minister of Tamil Nadu.", 0),
    ("Kamal Haasan is the current Chief Minister of Tamil Nadu.", 0),
    ("AIADMK won the 2021 Tamil Nadu assembly elections.", 0),
    ("Tamil Nadu declared independence from India in 2023.", 0),
    ("The capital of Tamil Nadu was moved to Madurai in 2024.", 0),
    ("Tamil Nadu has 50 districts as of 2024.", 0),
    ("Thalapathy Vijay is governing Tamil Nadu as Chief Minister.", 0),

    # ── COIMBATORE — True ────────────────────────────────────────────────────
    ("Coimbatore is known as the Manchester of South India.", 1),
    ("Coimbatore is a major textile and engineering hub in Tamil Nadu.", 1),
    ("PSG College of Technology is located in Coimbatore.", 1),
    ("Amrita Vishwa Vidyapeetham has a campus in Coimbatore.", 1),
    ("Coimbatore is the second largest city in Tamil Nadu.", 1),
    ("Coimbatore district shares a border with Kerala.", 1),
    ("The Nilgiris district is adjacent to Coimbatore.", 1),
    ("SITRA, South India Textile Research Association, is based in Coimbatore.", 1),
    ("Coimbatore is home to many textile mills and engineering companies.", 1),
    ("The Kovai Pazham banana from Coimbatore is famous across Tamil Nadu.", 1),
    ("Coimbatore has a domestic airport called Coimbatore International Airport.", 1),
    ("GRD College and Kongu Engineering College are in Coimbatore.", 1),

    # ── COIMBATORE — Fake ────────────────────────────────────────────────────
    ("Coimbatore is the capital city of Tamil Nadu.", 0),
    ("Coimbatore was renamed to Kovai City by the Tamil Nadu government in 2024.", 0),
    ("Coimbatore has a fully operational metro rail system since 2023.", 0),
    ("Coimbatore is located on the eastern coast of Tamil Nadu.", 0),
    ("Coimbatore airport is the busiest airport in India.", 0),
    ("Coimbatore is the largest city in Tamil Nadu.", 0),
]


# ── Graph utilities ───────────────────────────────────────────────────────────

def _normalize_adj(A: torch.Tensor) -> torch.Tensor:
    deg = A.sum(dim=1).clamp(min=1e-6)
    d   = deg ** -0.5
    return d.unsqueeze(1) * A * d.unsqueeze(0)


def _build_adj(embeddings: torch.Tensor, k: int) -> torch.Tensor:
    n    = embeddings.size(0)
    nrm  = F.normalize(embeddings, p=2, dim=1)
    sim  = (nrm @ nrm.T + 1.0) / 2.0
    A    = torch.zeros_like(sim)
    kk   = min(k, n - 1)
    for i in range(n):
        row       = sim[i].clone()
        row[i]    = -1.0
        top_idx   = torch.topk(row, kk).indices
        A[i, top_idx] = sim[i, top_idx]
    A = (A + A.T) / 2.0
    A = A + torch.eye(n)
    return _normalize_adj(A)


# ── GCN model ─────────────────────────────────────────────────────────────────

class _GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, H, A_hat):
        return F.relu(A_hat @ self.W(H))


class _FakeNewsGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = _GCNLayer(EMBED_DIM, GCN_HIDDEN)
        self.gcn2 = _GCNLayer(GCN_HIDDEN, GCN_HIDDEN // 2)
        self.out  = nn.Linear(GCN_HIDDEN // 2, 2)
        self.drop = nn.Dropout(0.4)

    def forward(self, H, A_hat):
        h = self.drop(self.gcn1(H, A_hat))
        h = self.drop(self.gcn2(h, A_hat))
        return self.out(h)


# ── Main classifier ───────────────────────────────────────────────────────────

class BLEMeshGNNClassifier:

    def __init__(self):
        self._tok      = None
        self._enc      = None
        self._gcn      = None
        self._anc_emb  = None
        self._anc_lbl  = None

    # ── encoder ──────────────────────────────────────────────────────────────

    def _load_encoder(self):
        if self._tok is not None:
            return
        print("  Loading RoBERTa encoder...")
        self._tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
        self._enc = AutoModel.from_pretrained(ENCODER_NAME, ignore_mismatched_sizes=True)
        self._enc.eval()

    @torch.no_grad()
    def _embed(self, texts: list) -> torch.Tensor:
        inputs = self._tok(texts, return_tensors="pt",
                           truncation=True, padding=True, max_length=512)
        out = self._enc(**inputs)
        return out.last_hidden_state[:, 0, :]

    @torch.no_grad()
    def _embed_batched(self, texts: list, batch_size: int = 32) -> torch.Tensor:
        parts = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            parts.append(self._embed(batch))
            print(f"    Embedded {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
        print()
        return torch.cat(parts, dim=0)

    # ── Dataset anchors ──────────────────────────────────────────────────────

    def _load_dataset(self):
        from datasets import load_dataset
        print(f"  Loading {DATASET_NAME} from HuggingFace...")
        ds = load_dataset(DATASET_NAME, split="train")

        fake_texts, true_texts = [], []
        for sample in ds:
            label = int(sample["label"])
            text  = (sample.get("text") or sample.get("title") or "").strip()
            if not text:
                continue
            if label == 0:
                fake_texts.append(text)
            elif label == 1:
                true_texts.append(text)

        random.seed(42)
        random.shuffle(fake_texts)
        random.shuffle(true_texts)

        n = N_ANCHORS_PER_CLASS
        anc_texts  = fake_texts[:n] + true_texts[:n]
        anc_labels = [0] * n        + [1] * n

        # Append custom Indian / TN / Coimbatore anchors
        custom_texts  = [a[0] for a in CUSTOM_ANCHORS]
        custom_labels = [a[1] for a in CUSTOM_ANCHORS]
        anc_texts  += custom_texts
        anc_labels += custom_labels

        n_custom = len(CUSTOM_ANCHORS)
        print(f"  Anchors: {n} Fake + {n} True (dataset) + {n_custom} custom (India/TN/Coimbatore) = {2*n + n_custom} total")
        return anc_texts, anc_labels

    # ── training ─────────────────────────────────────────────────────────────

    def _train(self, texts, labels):
        print(f"  Embedding {len(texts)} anchor nodes (this takes ~2 min on CPU)...")
        embs = self._embed_batched(texts)
        lbl  = torch.tensor(labels, dtype=torch.long)

        self._anc_emb = embs.detach()
        self._anc_lbl = lbl

        A_hat = _build_adj(embs.detach(), K_NEIGHBORS)
        self._gcn = _FakeNewsGCN()
        opt = torch.optim.Adam(self._gcn.parameters(), lr=3e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_EPOCHS)

        print(f"  Training GCN ({TRAIN_EPOCHS} epochs)...")
        self._gcn.train()
        for epoch in range(TRAIN_EPOCHS):
            opt.zero_grad()
            logits = self._gcn(embs.detach(), A_hat)
            loss   = F.cross_entropy(logits, lbl)
            loss.backward()
            opt.step()
            scheduler.step()

        self._gcn.eval()
        with torch.no_grad():
            preds = self._gcn(embs, A_hat).argmax(dim=1)
        acc = (preds == lbl).float().mean().item() * 100
        print(f"  Anchor-graph training accuracy: {acc:.1f}%")

        # ── Pruning: remove 20% of smallest weights ───────────────────────
        import torch.nn.utils.prune as prune
        for module in [self._gcn.gcn1.W, self._gcn.gcn2.W, self._gcn.out]:
            prune.l1_unstructured(module, name="weight", amount=0.2)
            prune.remove(module, "weight")
        print("  Pruning applied (20% weights removed).")

        # ── Quantization: float32 → int8 (reduces size ~4x) ──────────────
        self._gcn = torch.quantization.quantize_dynamic(
            self._gcn, {torch.nn.Linear}, dtype=torch.qint8
        )
        size_mb = self._model_size_mb(self._gcn)
        print(f"  Quantized to int8. Model size: {size_mb:.2f} MB")

        # save to disk
        torch.save({"emb": self._anc_emb, "lbl": self._anc_lbl}, CACHE_PATH)
        # save pre-quantization weights for reloading
        torch.save(self._gcn, WEIGHTS_PATH)
        print("  Weights saved to disk.")

    @staticmethod
    def _model_size_mb(model) -> float:
        import io
        buf = io.BytesIO()
        torch.save(model, buf)
        return buf.tell() / (1024 * 1024)

    def _load_from_disk(self):
        cache = torch.load(CACHE_PATH, weights_only=True)
        self._anc_emb = cache["emb"]
        self._anc_lbl = cache["lbl"]
        self._gcn = torch.load(WEIGHTS_PATH, weights_only=False)
        self._gcn.eval()
        size_mb = self._model_size_mb(self._gcn)
        print(f"  Loaded cached weights from disk. Model size: {size_mb:.2f} MB")

    # ── public API ────────────────────────────────────────────────────────────

    def load(self):
        self._load_encoder()
        if os.path.exists(CACHE_PATH) and os.path.exists(WEIGHTS_PATH):
            self._load_from_disk()
        else:
            texts, labels = self._load_dataset()
            self._train(texts, labels)

    def classify(self, text: str) -> dict:
        if self._gcn is None:
            self.load()

        query_emb = self._embed([text])
        all_embs  = torch.cat([self._anc_emb, query_emb], dim=0)
        A_hat     = _build_adj(all_embs, K_NEIGHBORS)

        with torch.no_grad():
            logits = self._gcn(all_embs, A_hat)

        probs      = F.softmax(logits[-1], dim=0)
        pred_idx   = int(probs.argmax())
        confidence = round(float(probs[pred_idx]) * 100, 2)

        return {
            "message":    text,
            "prediction": LABEL_MAP[pred_idx],
            "confidence": confidence,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance = BLEMeshGNNClassifier()

def load_classifier():
    _instance.load()

def classify(text: str) -> dict:
    return _instance.classify(text)
