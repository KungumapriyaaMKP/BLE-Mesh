"""
Quick test harness for the pretrained fake-news classifier.

Validates the inference pipeline (text in -> label + confidence out)
independently of BLE, so model issues and BLE issues don't get tangled.
"""

from classifier import classify

SAMPLE_MESSAGES = [
    "NASA confirms humans landed on Mars in 2025.",
    "The Federal Reserve raised interest rates by 0.25% on Wednesday.",
    "Drinking bleach cures COVID-19, doctors secretly admit.",
    "Apple released the iPhone 16 in September 2024.",
    "5G towers are spreading a mind-control virus, leaked report shows.",
]

if __name__ == "__main__":
    for msg in SAMPLE_MESSAGES:
        out = classify(msg)
        print(f"Input: {out['message']}")
        print(f"Prediction: {out['prediction']}")
        print(f"Confidence: {out['confidence']}%")
        print("-" * 60)
