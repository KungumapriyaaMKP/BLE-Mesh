"""
Evaluation script for the BLE Mesh GCN fake news classifier.
Reports Accuracy, Precision, Recall, F1 on 200 unseen test samples
from GonzaloA/fake_news (held out from the 455 anchor nodes).

Run:  python evaluate.py
"""

import random
import torch
from gnn_classifier import _instance, DATASET_NAME, N_ANCHORS_PER_CLASS


def compute_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy  = (tp + tn) / len(y_true) if y_true else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    return accuracy, precision, recall, f1


def main():
    print("=" * 60)
    print("  BLE Mesh GCN — Evaluation on GonzaloA/fake_news Test Set")
    print("=" * 60)

    # Load model (uses cached weights if available)
    print("\nLoading model...")
    _instance.load()

    # Load test samples (different from anchors)
    from datasets import load_dataset
    print(f"Loading {DATASET_NAME} test samples...")
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

    # Skip the first N_ANCHORS_PER_CLASS (used as anchors), take next 100
    n     = N_ANCHORS_PER_CLASS
    n_test = 100
    test_texts  = fake_texts[n: n + n_test] + true_texts[n: n + n_test]
    test_labels = [0] * n_test              + [1] * n_test

    print(f"Test set: {n_test} Fake + {n_test} True = {2*n_test} samples\n")

    # Classify each test sample
    y_true, y_pred = [], []
    for i, (text, label) in enumerate(zip(test_texts, test_labels)):
        result = _instance.classify(text)
        pred   = 0 if result["prediction"] == "Fake" else 1
        y_true.append(label)
        y_pred.append(pred)
        print(f"  [{i+1:3d}/{2*n_test}]  GT={'Fake' if label==0 else 'True':4s}  "
              f"Pred={result['prediction']:4s}  Conf={result['confidence']:6.2f}%  "
              f"{'OK' if label==pred else 'WRONG'}", end="\r")

    print("\n")

    accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred)
    correct = sum(t == p for t, p in zip(y_true, y_pred))

    print("=" * 60)
    print(f"  Results on {2*n_test} unseen test samples")
    print("=" * 60)
    print(f"  Correct      : {correct} / {2*n_test}")
    print(f"  Accuracy     : {accuracy*100:.2f}%")
    print(f"  Precision    : {precision*100:.2f}%")
    print(f"  Recall       : {recall*100:.2f}%")
    print(f"  F1 Score     : {f1*100:.2f}%")
    print("=" * 60)

    # Per-class breakdown
    fake_correct = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    true_correct = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    print(f"\n  Fake correctly identified : {fake_correct}/{n_test}")
    print(f"  True correctly identified : {true_correct}/{n_test}")
    print("=" * 60)


if __name__ == "__main__":
    main()
