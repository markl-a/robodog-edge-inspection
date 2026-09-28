"""Train the gauge CNN, export ONNX, quantize to INT8, and compare against the classic reader.

python -m edge.train_gauge          -> results/gauge_fp32.onnx, results/gauge_int8.onnx, results/gauge_report.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch import nn

from .gauge import SPEC, dataset, read_classic

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
SIZE = 96
MIN_CONF = 0.3          # same floor the mission runner uses


class GaugeNet(nn.Module):
    """~0.2 M params. Keeps spatial layout (no global pooling) because the answer *is* a direction."""

    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 48), block(48, 64))   # 96 -> 6
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 6 * 6, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):                       # x: (N,1,96,96) float in [0,1]
        return self.head(self.features(x)).squeeze(1)


def to_tensor(imgs):
    return torch.from_numpy(imgs.astype(np.float32)[:, None] / 255.0)


def errors(pred_vals, vals):
    e = np.abs(np.asarray(pred_vals) - vals)
    return {"mae": round(float(e.mean()), 3), "p95": round(float(np.percentile(e, 95)), 3), "max": round(float(e.max()), 3),
            "within_class_pct": round(100 * float((e <= SPEC.tolerance).mean()), 1), "gross_errors_gt_1bar": int((e > 1.0).sum())}


def ort_session(path, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads          # edge boxes share cores with navigation; budget one
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_predict(sess, imgs):
    name = sess.get_inputs()[0].name
    x = imgs.astype(np.float32)[:, None] / 255.0
    return np.concatenate([sess.run(None, {name: x[i:i + 1]})[0] for i in range(len(x))]) * SPEC.span + SPEC.min_value


def latency_ms(fn, img, n=200):
    for _ in range(10):
        fn(img)
    t = time.perf_counter()
    for _ in range(n):
        fn(img)
    return round((time.perf_counter() - t) / n * 1000, 3)


class Calib(CalibrationDataReader):
    def __init__(self, imgs, name):
        self.it = iter([{name: imgs[i:i + 1].astype(np.float32)[:, None] / 255.0} for i in range(len(imgs))])

    def get_next(self):
        return next(self.it, None)


def main(epochs: int = 12):
    torch.manual_seed(0)
    OUT.mkdir(exist_ok=True)
    xtr, ytr = dataset(12000, seed=10, size=SIZE)
    xte, yte = dataset(1000, seed=30, size=SIZE)
    xhd, yhd = dataset(1000, seed=31, size=SIZE, hard=True)      # heavy glare, blur, noise
    xcal, _ = dataset(256, seed=40, size=SIZE)

    model = GaugeNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * (len(xtr) // 128))
    X, Y = to_tensor(xtr), torch.from_numpy(((ytr - SPEC.min_value) / SPEC.span).astype(np.float32))
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X) - 127, 128):
            idx = perm[i:i + 128]
            loss = nn.functional.smooth_l1_loss(model(X[idx]), Y[idx], beta=0.01)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        print(f"epoch {ep} loss {loss.item():.4f}", flush=True)
    train_s = time.perf_counter() - t0

    model.eval()
    fp32 = OUT / "gauge_fp32.onnx"
    torch.onnx.export(model, torch.zeros(1, 1, SIZE, SIZE), str(fp32), input_names=["image"], output_names=["value01"],
                      dynamic_axes={"image": {0: "n"}, "value01": {0: "n"}}, opset_version=17, dynamo=False)
    pre = OUT / "gauge_fp32_pre.onnx"
    quant_pre_process(str(fp32), str(pre))
    int8 = OUT / "gauge_int8.onnx"
    quantize_static(str(pre), str(int8), Calib(xcal, "image"), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre.unlink()

    s32, s8 = ort_session(fp32), ort_session(int8)
    with torch.no_grad():
        torch_pred = model(to_tensor(xte)).numpy() * SPEC.span + SPEC.min_value
    onnx_pred = ort_predict(s32, xte)

    report = {
        "spec": {"range": f"{SPEC.min_value}-{SPEC.max_value} {SPEC.unit}", "accuracy_class": SPEC.accuracy_class,
                 "tolerance": round(SPEC.tolerance, 3)},
        "train": {"images": len(xtr), "epochs": epochs, "seconds": round(train_s, 1),
                  "params": sum(p.numel() for p in model.parameters())},
        "onnx_vs_torch_max_abs_diff": float(np.abs(onnx_pred - torch_pred).max()),
        "test_normal": {}, "test_hard": {}, "latency_ms_batch1_1thread": {}, "model_kb": {},
    }
    readers = {
        "classic": lambda imgs: np.array([read_classic(i).value for i in imgs]),
        "cnn_fp32": lambda imgs: ort_predict(s32, imgs),
        "cnn_int8": lambda imgs: ort_predict(s8, imgs),
    }
    for name, fn in readers.items():
        report["test_normal"][name] = errors(fn(xte), yte)
        report["test_hard"][name] = errors(fn(xhd), yhd)
        report["latency_ms_batch1_1thread"][name] = latency_ms(lambda im: fn(im[None]), xte[0])
    # the classic reader can say "I'm not sure": readings under the mission's confidence floor go to human review
    for split, (xs, ys) in {"test_normal": (xte, yte), "test_hard": (xhd, yhd)}.items():
        rs = [read_classic(i) for i in xs]
        conf = np.array([r.confidence for r in rs])
        keep = conf >= MIN_CONF
        e = errors(np.array([r.value for r in rs])[keep], ys[keep])
        report[split]["classic_after_review_gate"] = {**e, "sent_to_review_pct": round(100 * float((~keep).mean()), 1)}
    report["model_kb"] = {"cnn_fp32": round(fp32.stat().st_size / 1024, 1), "cnn_int8": round(int8.stat().st_size / 1024, 1)}
    (OUT / "gauge_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
