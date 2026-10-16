# scripts/run_grid.py: the three e501 pipelines (TF-IDF + LR, LaBSE + LR, fine-tuned AfroXLMR) on a list of AfriSenti languages, with the e501 settings.
# Resumable: every finished run appends one row to results/grid.csv and one prediction file, and a run that already has both is skipped.
# Usage, from the repository folder: python scripts/run_grid.py kin swa tso arq por twi pcm ary amh yor ibo hau
import argparse, gc, json, os, platform, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, sklearn, torch, transformers, sentence_transformers, datasets
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parents[1] # the repository folder, one level above scripts/
NORM_PATH = ROOT / "data" / "afrisenti_normalized.parquet" # every tweet with its normalized text, written by notebook Cell 1.1
AUDIT_PATH = ROOT / "data" / "afrisenti_audit.csv" # the parquet commit, written by Cell 1.1
FERT_PATH = ROOT / "data" / "afrisenti_fertility.csv" # the tokenizer commits, written by Cell 1.3
RETYPED_PATH = ROOT / "data" / "kin_test_retyped.csv" # the retyped Kinyarwanda test set, written by Cell 1.2
GRID_PATH = ROOT / "results" / "grid.csv" # one row per finished run and split
PRED_DIR = ROOT / "results" / "predictions" # one file per language, pipeline and seed
MODEL_DIR = ROOT / "models" # the kept models, ignored by git
RUN_DIR = ROOT / "runs" # Trainer checkpoints, deleted after scoring, ignored by git

SEED = 42 # the seed of e501 for everything that is not a fine-tuning seed
AFROXLMR_ID = os.environ.get("GRID_AFROXLMR", "Davlan/afro-xlmr-base") # the environment variable only serves the smoke test with a tiny local model
LABSE_ID = os.environ.get("GRID_LABSE", "sentence-transformers/LaBSE")
MAX_LEN = 128 # the AfroXLMR truncation limit of e501
N_TIMING = 200 # test tweets drawn for the latency test, as in e501
LABEL_NAMES = ["positive", "neutral", "negative"] # ClassLabel ids 0, 1, 2
ID2LABEL = dict(enumerate(LABEL_NAMES))
LABEL2ID = {n: i for i, n in ID2LABEL.items()}
TOKEN_PATTERN = r"\w+(?:['’‘\-–]\w+)*['’‘]?" # the e501 word pattern: apostrophes and hyphens stay inside a word
EXPECTED_KIN = {"tfidf": (0.5965, 0.6006), "labse": (0.6433, 0.6484), "afroxlmr": (0.659, 0.662)} # e501 test accuracy and macro F1; AfroXLMR as the mean over seeds 42, 43, 44
TOLERANCE = {"tfidf": 0.0005, "labse": 0.005, "afroxlmr": 0.015} # exact for TF-IDF, a little room for GPU encoding, the e501 rerun spread for the fine-tune

parser = argparse.ArgumentParser(description="run the three e501 pipelines on AfriSenti languages")
parser.add_argument("langs", nargs="+", help="language codes in the order to run, for example kin swa tso")
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="fine-tuning seeds")
parser.add_argument("--pipelines", nargs="+", default=["tfidf", "labse", "afroxlmr"], choices=["tfidf", "labse", "afroxlmr"])
parser.add_argument("--latency-langs", nargs="*", default=["kin"], help="languages that get the latency test (one at a time and in a batch, GPU and CPU)")
parser.add_argument("--keep-langs", nargs="*", default=["kin", "swa"], help="languages whose models are saved under models/")
parser.add_argument("--force", action="store_true", help="run again even if the rows and predictions already exist")
parser.add_argument("--no-strict-kin", action="store_true", help="do not stop when kin fails to reproduce e501")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
datasets.disable_progress_bars()
transformers.logging.set_verbosity_error() # the "newly initialized classifier" notice would repeat 36 times

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True) # flush so nohup's log file shows progress at once

def revision_for(model_id, commit):
    return None if Path(model_id).exists() else commit # a local path (the smoke test) has no Hub revision

# ---------- data and provenance ----------
audit = pd.read_csv(AUDIT_PATH).set_index("lang")
fert = pd.read_csv(FERT_PATH).set_index("lang")
AFROXLMR_COMMIT = fert["afroxlmr_commit"].iloc[0] # the commits Cell 1.3 measured the tokenizers at
LABSE_COMMIT = fert["labse_commit"].iloc[0]
data = pd.read_parquet(NORM_PATH, columns=["lang", "split", "label", "text"])
ENV = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
       "torch": torch.__version__, "transformers": transformers.__version__, "sentence_transformers": sentence_transformers.__version__,
       "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "none",
       "parquet_commit": audit["parquet_commit"].iloc[0], "afroxlmr_commit": AFROXLMR_COMMIT, "labse_commit": LABSE_COMMIT} # recorded on every row
for d in [GRID_PATH.parent, PRED_DIR, MODEL_DIR, RUN_DIR]:
    d.mkdir(parents=True, exist_ok=True)
log(f"device {DEVICE}; {json.dumps(ENV)}")

def splits_for(lang):
    part = data[data["lang"] == lang]
    X = {s: part.loc[part["split"] == s, "text"].tolist() for s in ["train", "validation", "test"]} # train is the final training set
    y = {s: part.loc[part["split"] == s, "label"].to_numpy() for s in X}
    assert len(X["train"]) == audit.loc[lang, "train_final"] and len(X["test"]) == audit.loc[lang, "test"], f"{lang}: sizes differ from the audit"
    retyped = None
    if lang == "kin":
        r = pd.read_csv(RETYPED_PATH)
        assert r["text"].tolist() == X["test"], "the retyped file does not match the kin test split" # same tweets, same order
        retyped = r["retyped"].tolist()
    rng = np.random.default_rng(SEED)
    timing = [X["test"][i] for i in rng.choice(len(X["test"]), size=min(N_TIMING, len(X["test"])), replace=False)] # the e501 draw
    return X, y, retyped, timing

# ---------- bookkeeping ----------
def load_grid():
    return pd.read_csv(GRID_PATH) if GRID_PATH.exists() else pd.DataFrame(columns=["lang", "pipeline", "seed", "split"])

def record(lang, pipeline, seed, split, **metrics):
    row = {"lang": lang, "pipeline": pipeline, "seed": seed, "split": split, **metrics, **ENV, "recorded_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    old = load_grid()
    same = (old["lang"] == lang) & (old["pipeline"] == pipeline) & (old["seed"] == seed) & (old["split"] == split)
    pd.concat([old[~same], pd.DataFrame([row])], ignore_index=True).to_csv(GRID_PATH, index=False) # replace the row if it exists, else append
    shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
    log(f"recorded {lang} {pipeline} seed {seed} {split}: {shown}")

def pred_path(lang, pipeline, seed):
    return PRED_DIR / f"{lang}_{pipeline}_seed{seed}.csv"

def model_path(lang, pipeline, seed):
    return MODEL_DIR / {"tfidf": f"tfidf_{lang}.joblib", "labse": f"labse_lr_{lang}.joblib", "afroxlmr": f"afroxlmr_{lang}_seed{seed}"}[pipeline]

def done(lang, pipeline, seed):
    if args.force: return False
    g = load_grid()
    has_row = ((g["lang"] == lang) & (g["pipeline"] == pipeline) & (g["seed"] == seed) & (g["split"] == "test")).any() # the test row is written last
    has_model = lang not in args.keep_langs or model_path(lang, pipeline, seed).exists() # a kept language also needs its saved model
    return bool(has_row) and pred_path(lang, pipeline, seed).exists() and has_model

def save_predictions(lang, pipeline, seed, y_test, pred, pred_retyped=None):
    out = pd.DataFrame({"idx": range(len(y_test)), "label": y_test, "pred": pred})
    if pred_retyped is not None: out["pred_retyped"] = pred_retyped
    out.to_csv(pred_path(lang, pipeline, seed), index=False)

def scores(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")

def time_per_sample(predict_one, texts, warmup=5):
    for t in texts[:warmup]: predict_one(t) # the first calls pay one-off costs, not counted
    start = time.perf_counter()
    for t in texts: predict_one(t)
    return (time.perf_counter() - start) / len(texts) * 1000 # milliseconds per tweet

def peak_gib():
    return torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else float("nan")

def reset_peak():
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()

# ---------- pipeline 1: TF-IDF + logistic regression ----------
def run_tfidf(lang, X, y, retyped, timing):
    word = TfidfVectorizer(analyzer="word", token_pattern=TOKEN_PATTERN, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)
    pipe = Pipeline([("features", FeatureUnion([("word", word), ("char", char)])), ("clf", clf)])
    t0 = time.perf_counter(); pipe.fit(X["train"], y["train"]); fit_s = time.perf_counter() - t0
    n_features = len(pipe.named_steps["features"].get_feature_names_out())
    val_acc, val_f1 = scores(y["validation"], pipe.predict(X["validation"]))
    test_pred = pipe.predict(X["test"]); test_acc, test_f1 = scores(y["test"], test_pred)
    extra = {}
    if retyped is not None:
        pred_r = pipe.predict(retyped); extra["retyped_accuracy"], extra["retyped_macro_f1"] = scores(y["test"], pred_r)
    else: pred_r = None
    if lang in args.latency_langs:
        extra["ms_per_sample_cpu"] = time_per_sample(lambda t: pipe.predict([t])[0], timing)
        t0 = time.perf_counter(); pipe.predict(timing); extra["ms_per_sample_batch_cpu"] = (time.perf_counter() - t0) / len(timing) * 1000
    if lang in args.keep_langs: joblib.dump(pipe, model_path(lang, "tfidf", SEED)) # kept for the suite and the latency test
    record(lang, "tfidf", SEED, "validation", accuracy=val_acc, macro_f1=val_f1)
    record(lang, "tfidf", SEED, "test", accuracy=test_acc, macro_f1=test_f1, train_time_s=fit_s, n_features=n_features, device="cpu", **extra)
    save_predictions(lang, "tfidf", SEED, y["test"], test_pred, pred_r)

# ---------- pipeline 2: LaBSE embeddings + the same logistic regression ----------
_encoder = None
def encoder():
    global _encoder
    if _encoder is None:
        t0 = time.perf_counter()
        _encoder = SentenceTransformer(LABSE_ID, device=DEVICE, revision=revision_for(LABSE_ID, LABSE_COMMIT))
        log(f"loaded {LABSE_ID} on {DEVICE} in {time.perf_counter() - t0:.1f} s")
    return _encoder

def free_encoder():
    global _encoder
    _encoder = None; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache() # so the fine-tune's peak memory is its own

def run_labse(lang, X, y, retyped, timing):
    cache = ROOT / "data" / f"labse_{lang}.npz" # the embeddings, ignored by git
    parts = {**X, **({"retyped": retyped} if retyped is not None else {})}
    if cache.exists() and not args.force:
        z = np.load(cache); E = {k: z[k] for k in parts}; encode_s = float(z["encode_s"]); peak = float(z["peak_gib"])
        log(f"{lang}: LaBSE embeddings read from {cache.name}")
    else:
        enc = encoder(); torch.manual_seed(SEED); reset_peak()
        t0 = time.perf_counter()
        E = {k: enc.encode(v, batch_size=64, device=DEVICE, convert_to_numpy=True, show_progress_bar=False) for k, v in parts.items()}
        encode_s = time.perf_counter() - t0; peak = peak_gib()
        np.savez_compressed(cache, encode_s=encode_s, peak_gib=peak, **E)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)
    t0 = time.perf_counter(); clf.fit(E["train"], y["train"]); fit_s = time.perf_counter() - t0
    val_acc, val_f1 = scores(y["validation"], clf.predict(E["validation"]))
    test_pred = clf.predict(E["test"]); test_acc, test_f1 = scores(y["test"], test_pred)
    extra = {}
    if retyped is not None:
        pred_r = clf.predict(E["retyped"]); extra["retyped_accuracy"], extra["retyped_macro_f1"] = scores(y["test"], pred_r)
    else: pred_r = None
    if lang in args.latency_langs:
        enc = encoder()
        one = lambda t, dev: clf.predict(enc.encode([t], device=dev, convert_to_numpy=True, show_progress_bar=False))[0]
        extra["ms_per_sample"] = time_per_sample(lambda t: one(t, DEVICE), timing)
        t0 = time.perf_counter(); clf.predict(enc.encode(timing, batch_size=64, device=DEVICE, convert_to_numpy=True, show_progress_bar=False))
        extra["ms_per_sample_batch"] = (time.perf_counter() - t0) / len(timing) * 1000
        extra["ms_per_sample_cpu"] = time_per_sample(lambda t: one(t, "cpu"), timing)
        enc.to(DEVICE)
    if lang in args.keep_langs: joblib.dump(clf, model_path(lang, "labse", SEED)) # the classifier; the encoder is the public model
    record(lang, "labse", SEED, "validation", accuracy=val_acc, macro_f1=val_f1)
    record(lang, "labse", SEED, "test", accuracy=test_acc, macro_f1=test_f1, train_time_s=encode_s + fit_s, encode_time_s=encode_s, fit_time_s=fit_s,
           peak_gpu_gib=peak, n_features=E["train"].shape[1], device=DEVICE, **extra)
    save_predictions(lang, "labse", SEED, y["test"], test_pred, pred_r)

# ---------- pipeline 3: fine-tuned AfroXLMR ----------
_tok = None
def tokenizer():
    global _tok
    if _tok is None: _tok = AutoTokenizer.from_pretrained(AFROXLMR_ID, revision=revision_for(AFROXLMR_ID, AFROXLMR_COMMIT))
    return _tok

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(pred.label_ids, preds), "macro_f1": f1_score(pred.label_ids, preds, average="macro")}

@torch.no_grad()
def predict_one_xlmr(t, model, dev):
    enc = tokenizer()(t, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(dev)
    return int(model(**enc).logits.argmax(-1).item())

@torch.no_grad()
def predict_batch_xlmr(texts, model, dev):
    enc = tokenizer()(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(dev)
    return model(**enc).logits.argmax(-1).cpu().numpy()

def run_afroxlmr(lang, X, y, retyped, timing, seeds):
    if all(done(lang, "afroxlmr", s) for s in seeds):
        log(f"{lang} afroxlmr: all seeds already done, skipped"); return
    tok = tokenizer()
    def tokenize(batch): return tok(batch["text"], truncation=True, max_length=MAX_LEN)
    parts = {**X, **({"retyped": retyped} if retyped is not None else {})}
    ds = {k: Dataset.from_dict({"text": v, "label": y["test" if k == "retyped" else k].tolist()}).map(tokenize, batched=True, remove_columns=["text"]) for k, v in parts.items()}
    for seed in seeds:
        if done(lang, "afroxlmr", seed):
            log(f"{lang} afroxlmr seed {seed}: already done, skipped"); continue
        set_seed(seed); reset_peak()
        model = AutoModelForSequenceClassification.from_pretrained(AFROXLMR_ID, revision=revision_for(AFROXLMR_ID, AFROXLMR_COMMIT),
                                                                   num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)
        out_dir = RUN_DIR / f"afroxlmr_{lang}_seed{seed}"
        targs = TrainingArguments(output_dir=str(out_dir), seed=seed, data_seed=seed,
                                  num_train_epochs=3, learning_rate=2e-5, per_device_train_batch_size=32, per_device_eval_batch_size=64,
                                  eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
                                  load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
                                  save_total_limit=1, save_only_model=True, bf16=(DEVICE == "cuda"), report_to="none", disable_tqdm=True)
        trainer = Trainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["validation"], processing_class=tok, compute_metrics=compute_metrics)
        trainer.remove_callback(transformers.PrinterCallback) # one summary line per seed below instead of seven dictionaries
        train_s = trainer.train().metrics["train_runtime"]; peak = peak_gib()
        tr = [r for r in trainer.state.log_history if "loss" in r and "eval_loss" not in r] # training loss per epoch
        ev = [r for r in trainer.state.log_history if "eval_macro_f1" in r] # validation metrics per epoch
        best = max(ev, key=lambda r: r["eval_macro_f1"]) # the epoch the Trainer kept (the first best on a tie, as the Trainer does)
        log(f"{lang} afroxlmr seed {seed}: trained in {train_s:.0f} s, peak {peak:.2f} GiB; train loss by epoch " + ", ".join(f"{r['loss']:.4f}" for r in tr)
            + "; validation macro F1 by epoch " + ", ".join(f"{r['eval_macro_f1']:.4f}" for r in ev) + f"; best epoch {int(round(best['epoch']))}")
        test_pred = trainer.predict(ds["test"]).predictions.argmax(-1); test_acc, test_f1 = scores(y["test"], test_pred)
        extra = {}
        if retyped is not None:
            pred_r = trainer.predict(ds["retyped"]).predictions.argmax(-1); extra["retyped_accuracy"], extra["retyped_macro_f1"] = scores(y["test"], pred_r)
        else: pred_r = None
        m = trainer.model.eval()
        if lang in args.latency_langs and seed == seeds[0]:
            extra["ms_per_sample"] = time_per_sample(lambda t: predict_one_xlmr(t, m, DEVICE), timing)
            t0 = time.perf_counter(); predict_batch_xlmr(timing, m, DEVICE); extra["ms_per_sample_batch"] = (time.perf_counter() - t0) / len(timing) * 1000
            m.to("cpu"); extra["ms_per_sample_cpu"] = time_per_sample(lambda t: predict_one_xlmr(t, m, "cpu"), timing); m.to(DEVICE)
        if lang in args.keep_langs: trainer.save_model(str(model_path(lang, "afroxlmr", seed))) # about 1.1 GB per seed
        record(lang, "afroxlmr", seed, "validation", accuracy=float(best["eval_accuracy"]), macro_f1=float(best["eval_macro_f1"]), best_epoch=int(round(best["epoch"])))
        record(lang, "afroxlmr", seed, "test", accuracy=test_acc, macro_f1=test_f1, train_time_s=train_s, peak_gpu_gib=peak, best_epoch=int(round(best["epoch"])), device=DEVICE, **extra)
        save_predictions(lang, "afroxlmr", seed, y["test"], test_pred, pred_r)
        del trainer, model, m; gc.collect(); shutil.rmtree(out_dir, ignore_errors=True) # the checkpoints go, the saved model stays
        if DEVICE == "cuda": torch.cuda.empty_cache()

# ---------- the kin check against e501 ----------
def check_kin():
    g = load_grid(); g = g[(g["lang"] == "kin") & (g["split"] == "test")]
    ok = True
    for p, (acc, f1) in EXPECTED_KIN.items():
        rows = g[g["pipeline"] == p]
        if p == "afroxlmr": rows = rows[rows["seed"].isin([42, 43, 44])]
        if len(rows) < (3 if p == "afroxlmr" else 1):
            log(f"kin check {p}: not enough rows yet"); ok = False; continue
        got_acc, got_f1 = rows["accuracy"].mean(), rows["macro_f1"].mean()
        passed = abs(got_acc - acc) <= TOLERANCE[p] and abs(got_f1 - f1) <= TOLERANCE[p]
        extra = f", seeds {rows['macro_f1'].round(4).tolist()}, sd {rows['macro_f1'].std(ddof=1):.4f}" if p == "afroxlmr" else ""
        log(f"kin check {p}: accuracy {got_acc:.4f} against e501 {acc:.4f}, macro F1 {got_f1:.4f} against {f1:.4f}{extra}: {'PASS' if passed else 'FAIL'}")
        ok &= passed
    return ok

# ---------- main loop ----------
for lang in args.langs:
    X, y, retyped, timing = splits_for(lang)
    log(f"=== {lang}: {len(X['train'])} train, {len(X['validation'])} validation, {len(X['test'])} test ===")
    if "tfidf" in args.pipelines:
        if done(lang, "tfidf", SEED): log(f"{lang} tfidf: already done, skipped")
        else: run_tfidf(lang, X, y, retyped, timing)
    if "labse" in args.pipelines:
        if done(lang, "labse", SEED): log(f"{lang} labse: already done, skipped")
        else: run_labse(lang, X, y, retyped, timing)
    if "afroxlmr" in args.pipelines:
        free_encoder() # the fine-tune measures its own memory
        run_afroxlmr(lang, X, y, retyped, timing, args.seeds)
    if lang == "kin" and set(args.pipelines) == {"tfidf", "labse", "afroxlmr"} and set(args.seeds) >= {42, 43, 44}:
        if not check_kin() and not args.no_strict_kin:
            log("kin does not reproduce e501, so the grid stops here; see the rows above"); sys.exit(1)
log("done")