"""
benchmark.py — Latency and cost reporting for the photo authenticity detector.

Measures:
  • Average and P95 inference time per image (ms)
  • On-device cost (always $0 — no cloud, no GPU)
  • Cloud-equivalent cost estimate for context

Run AFTER train.py:
    python benchmark.py
"""

import os
import glob
import time
import json
import numpy as np

from predict import predict

REAL_DIR   = r"C:\Users\hp\Downloads\real_img"
SCREEN_DIR = r"C:\Users\hp\Downloads\screen_img"
IMAGE_EXTENSIONS = ("*.jpeg", "*.jpg", "*.png", "*.bmp", "*.webp")

# How many images to sample for the benchmark (use all if fewer available)
SAMPLE_SIZE = 30


def collect_images(folder):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(set(paths))


def benchmark():
    all_images = collect_images(REAL_DIR) + collect_images(SCREEN_DIR)
    sample = all_images[:SAMPLE_SIZE]

    print(f"Benchmarking on {len(sample)} images (of {len(all_images)} total)\n")

    # ── Warm-up run (avoid cold-start bias) ──────────────────────────────────
    _ = predict(sample[0])

    # ── Timed runs ───────────────────────────────────────────────────────────
    times_ms = []
    for path in sample:
        start = time.perf_counter()
        score = predict(path)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

    times_ms = np.array(times_ms)
    avg_ms = times_ms.mean()
    p50_ms = np.percentile(times_ms, 50)
    p95_ms = np.percentile(times_ms, 95)
    min_ms = times_ms.min()
    max_ms = times_ms.max()

    import platform
    device = f"{platform.processor()} — {platform.system()} {platform.release()}"

    print("=" * 58)
    print(f"  Latency Results ({len(sample)} images)")
    print("=" * 58)
    print(f"  Average : {avg_ms:>8.2f} ms")
    print(f"  Median  : {p50_ms:>8.2f} ms")
    print(f"  P95     : {p95_ms:>8.2f} ms")
    print(f"  Min     : {min_ms:>8.2f} ms")
    print(f"  Max     : {max_ms:>8.2f} ms")
    print(f"\n  Device  : {device}")
    print(f"  GPU     : None (CPU only)")

    print("\n" + "=" * 58)
    print("  Cost Analysis")
    print("=" * 58)
    print(f"  On-device cost     : $0.00 per image")
    print(f"    -> Classical CV + tiny logistic regression.")
    print(f"       No API calls, no cloud inference, no GPU.")
    print(f"       Runs entirely in local Python process.")
    print()
    print(f"  Cloud-equivalent   : ~$0.01–$0.05 per 1,000 images")
    print(f"    -> If hosted on a small CPU cloud function")
    print(f"       (e.g. AWS Lambda / GCP Cloud Run) billing")
    print(f"       purely compute time at ~{avg_ms:.0f} ms per call.")
    print(f"       But on-device is the realistic deployment")
    print(f"       given how lightweight this model is.")
    print()
    imgs_per_sec = 1000.0 / avg_ms
    if imgs_per_sec >= 1.0:
        print(f"  Throughput (est.)  : ~{imgs_per_sec:.1f} images/second on this CPU")
    else:
        print(f"  Throughput (est.)  : ~{imgs_per_sec*60:.1f} images/minute on this CPU")
    print("=" * 58)

    # ── Save results ─────────────────────────────────────────────────────────
    results = {
        "n_images_benchmarked": len(sample),
        "avg_ms":  round(float(avg_ms), 2),
        "p50_ms":  round(float(p50_ms), 2),
        "p95_ms":  round(float(p95_ms), 2),
        "min_ms":  round(float(min_ms), 2),
        "max_ms":  round(float(max_ms), 2),
        "device":  device,
        "gpu":     "none",
        "on_device_cost_per_image_usd": 0.0,
        "cloud_cost_per_1k_images_usd": "0.01–0.05 (estimate)",
    }
    out_path = os.path.join(os.path.dirname(__file__), "model", "benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    benchmark()
