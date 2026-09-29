import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip() + "\n"}


cells = [
md('''# Professional Speech Emotion Recognition

This notebook builds a reproducible six-class SER system from RAVDESS, CREMA-D, TESS, and SAVEE. It uses speaker-disjoint splits, train-only augmentation, validation-based feature/model selection, a locked test set, saved inference artifacts, and a Gradio demo.

**Important:** the historical 78.57% result is not carried forward because its evaluation protocol was compromised. This notebook reports only values produced by the clean pipeline.'''),
md('''## 1. Environment and experiment rules

Use a Colab GPU runtime. Secrets are read from Colab Secrets; no credentials are embedded in code. Expensive stages are cached under `/content/ser_project`.'''),
code('''# Keep Colab's GPU-matched TensorFlow/NumPy scientific stack.
%pip install -q --upgrade "librosa>=0.10.2,<0.12" "soundfile>=0.12,<0.14" "gradio>=6,<7" "kaggle>=1.6,<2" "joblib>=1.4,<2" "tqdm>=4.66,<5" "jedi>=0.19,<1"
'''),
code('''import os, re, json, random, shutil, warnings, hashlib, gc, subprocess, sys
from dataclasses import dataclass, asdict
from pathlib import Path
import joblib, librosa, numpy as np, pandas as pd, seaborn as sns
import soundfile as sf, gradio as gr
import matplotlib.pyplot as plt
from IPython.display import display, Audio
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from importlib.metadata import version, requires, PackageNotFoundError
from packaging.version import Version
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

def ensure_jedi():
    try:
        return version("jedi")
    except PackageNotFoundError:
        pass
    try:
        from google import colab
    except ImportError:
        raise RuntimeError("IPython requires jedi. Run the package setup cell before imports.") from None
    print("Installing missing IPython dependency in this Colab runtime: jedi")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jedi>=0.19,<1"], check=True)
        installed = version("jedi")
    except (subprocess.CalledProcessError, PackageNotFoundError):
        raise RuntimeError("Could not install jedi. Rerun the package setup cell, then restart the runtime if Colab requests it.") from None
    print("Installed jedi", installed)
    return installed

ensure_jedi()

def verify_environment():
    queue = ["tensorflow", "librosa", "soundfile", "scikit-learn", "pandas",
             "numpy", "matplotlib", "seaborn", "gradio", "kaggle", "joblib", "tqdm", "jedi", "packaging", "ipython"]
    seen, problems = set(), []
    while queue:
        package = canonicalize_name(queue.pop())
        if package in seen: continue
        seen.add(package)
        try: version(package)
        except PackageNotFoundError:
            problems.append(f"Missing package: {package}"); continue
        for spec in requires(package) or []:
            dependency = Requirement(spec)
            if dependency.marker and not dependency.marker.evaluate({"extra": ""}): continue
            try: dependency_version = version(dependency.name)
            except PackageNotFoundError:
                problems.append(f"{package} requires missing {dependency.name}"); continue
            if dependency.specifier and not dependency.specifier.contains(dependency_version, prereleases=True):
                problems.append(f"{package} requires {dependency}; found {dependency_version}")
            queue.append(dependency.name)
    if problems:
        raise RuntimeError("Project dependency conflicts. Restart with a fresh Colab runtime, then run setup again: " + "; ".join(problems))
    print("Project dependency checks passed.")
verify_environment()

@dataclass(frozen=True)
class Config:
    seed: int = 42
    sample_rate: int = 16000
    duration: float = 4.0
    top_db: int = 30
    n_mfcc: int = 13
    n_fft: int = 512
    hop_length: int = 160
    frame_length: int = 400
    batch_size: int = 32
    max_epochs: int = 60
    learning_rate: float = 1e-3
    ablation_epochs: int = 15
    max_input_seconds: float = 60.0
    project_dir: str = "/content/ser_project"

CFG = Config()
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
PROJECT_DIR = Path(CFG.project_dir)
DATA_DIR, CACHE_DIR, ARTIFACT_DIR = [PROJECT_DIR / x for x in ("data", "cache", "artifacts")]
for directory in (DATA_DIR, CACHE_DIR, ARTIFACT_DIR): directory.mkdir(parents=True, exist_ok=True)

def set_seed(seed=CFG.seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass
set_seed()
gpus=tf.config.list_physical_devices("GPU")
if not gpus: warnings.warn("No GPU detected. Training will work but will be substantially slower.")
if not (Version("2.17") <= Version(tf.__version__) < Version("2.22")):
    warnings.warn(f"TensorFlow {tf.__version__} is outside the declared compatibility range.")
print("TensorFlow:",tf.__version__,"GPU devices:",gpus)
print({name:version(name) for name in ["numpy","pandas","scikit-learn","librosa","gradio","kaggle"]})'''),
md('''## 2. Reproducible dataset acquisition

The handles below point to common Kaggle mirrors. If a mirror changes, update only `KAGGLE_DATASETS`. Dataset parsing is filename-based and searches recursively.'''),
code('''KAGGLE_DATASETS = {
    "ravdess": "uwrfkaggler/ravdess-emotional-speech-audio",
    "cremad": "ejlok1/cremad",
    "tess": "ejlok1/toronto-emotional-speech-set-tess",
    "savee": "ejlok1/surrey-audiovisual-expressed-emotion-savee",
}

def configure_kaggle():
    try:
        from google.colab import userdata
    except ImportError:
        username, key = os.getenv("KAGGLE_USERNAME"), os.getenv("KAGGLE_KEY")
    else:
        try: username, key = userdata.get("KAGGLE_USERNAME"), userdata.get("KAGGLE_KEY")
        except Exception:
            raise RuntimeError("Add KAGGLE_USERNAME and KAGGLE_KEY to Colab Secrets and enable notebook access for both.") from None
    if not isinstance(username, str) or not isinstance(key, str) or not username.strip() or not key.strip():
        raise RuntimeError("Both Kaggle Secrets must contain non-empty strings.")
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = username.strip(), key.strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()  # Reads the environment variables above; no kaggle.json is required.
    return api

def download_datasets(force=False):
    api = configure_kaggle()
    roots = {}
    for name, handle in KAGGLE_DATASETS.items():
        target = DATA_DIR / name
        if force and target.exists(): shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        complete = target / ".download_complete"
        if not complete.exists() or complete.read_text().strip() != handle or not any(target.rglob("*.wav")):
            try: api.dataset_download_files(handle, path=str(target), unzip=True, quiet=False, force=True)
            except Exception:
                raise RuntimeError(f"Download failed for {handle}. Check your legacy Kaggle username/API key, dataset access, and available disk space.") from None
            if not any(target.rglob("*.wav")): raise RuntimeError(f"No WAV files found after downloading {handle}")
            complete.write_text(handle)
        roots[name] = target
    return roots

DATA_ROOTS = download_datasets(force=False)'''),
md('''## 3. Metadata parsing and label harmonization

All datasets are reduced to the same six-label taxonomy. Calm and surprise are intentionally excluded. Speaker IDs are prefixed by dataset so they cannot collide.'''),
code('''RAV_MAP={"01":"neutral","03":"happy","04":"sad","05":"angry","06":"fear","07":"disgust"}
CREMA_MAP={"ANG":"angry","DIS":"disgust","FEA":"fear","HAP":"happy","NEU":"neutral","SAD":"sad"}
TESS_MAP={"angry":"angry","disgust":"disgust","fear":"fear","happy":"happy","neutral":"neutral","sad":"sad"}
SAVEE_MAP={"a":"angry","d":"disgust","f":"fear","h":"happy","n":"neutral","sa":"sad"}

def record(path, dataset, speaker, sex, emotion):
    return {"path":str(path), "dataset":dataset, "speaker_id":f"{dataset}:{speaker}",
            "sex":sex, "emotion":emotion, "source_id":f"{dataset}:{speaker}:{path.stem.lower()}"}

def parse_ravdess(root):
    rows=[]
    for p in root.rglob("*.wav"):
        parts=p.stem.split("-")
        if len(parts)==7 and parts[0]=="03" and parts[1]=="01" and parts[2] in RAV_MAP:
            actor=parts[6]; rows.append(record(p,"ravdess",actor,"female" if int(actor)%2==0 else "male",RAV_MAP[parts[2]]))
    return rows

def parse_cremad(root):
    female={"1002","1003","1004","1006","1007","1008","1009","1010","1012","1013","1018","1020","1021","1024","1025","1028","1029","1030","1037","1043","1046","1047","1049","1052","1053","1054","1055","1056","1058","1060","1061","1063","1072","1073","1074","1075","1076","1078","1079","1082","1084","1089","1091"}
    rows=[]
    for p in root.rglob("*.wav"):
        parts=p.stem.split("_")
        if len(parts)>=4 and parts[2] in CREMA_MAP:
            rows.append(record(p,"cremad",parts[0],"female" if parts[0] in female else "male",CREMA_MAP[parts[2]]))
    return rows

def parse_tess(root):
    rows=[]
    for p in root.rglob("*.wav"):
        token=p.stem.split("_")[-1].lower(); speaker=p.stem.split("_")[0].upper()
        if token in TESS_MAP: rows.append(record(p,"tess",speaker,"female",TESS_MAP[token]))
    return rows

def parse_savee(root):
    rows=[]
    for p in root.rglob("*.wav"):
        m=re.match(r"^(?:(DC|JE|JK|KL)_)?(sa|su|[adfhn])\\d+$",p.stem,re.I)
        if m and m.group(2).lower() in SAVEE_MAP:
            speaker=(m.group(1) or p.parent.name).upper()
            if speaker in {"DC","JE","JK","KL"}:
                rows.append(record(p,"savee",speaker,"male",SAVEE_MAP[m.group(2).lower()]))
    return rows

def prepare_dataset(roots=DATA_ROOTS):
    rows=parse_ravdess(roots["ravdess"])+parse_cremad(roots["cremad"])+parse_tess(roots["tess"])+parse_savee(roots["savee"])
    if not rows: raise RuntimeError("No recognized recordings. Check DATA_ROOTS and the Kaggle download output.")
    df=pd.DataFrame(rows).sort_values(["dataset","speaker_id","path"])
    duplicate_count=int(df.source_id.duplicated().sum())
    if duplicate_count: print(f"Ignoring {duplicate_count} duplicate source recordings in dataset mirrors.")
    df=df.drop_duplicates("source_id").reset_index(drop=True)
    for dataset, count in {"ravdess":24, "cremad":91, "tess":2, "savee":4}.items():
        found=df.loc[df.dataset==dataset, "speaker_id"].nunique()
        if found!=count: raise RuntimeError(f"{dataset}: expected {count} speakers, parsed {found}. Check the corpus download and filenames.")
    assert len(df)>0 and set(df.emotion)==set(LABELS)
    assert df.path.map(Path).map(Path.exists).all() and not df.path.duplicated().any()
    return df

manifest=prepare_dataset()
display(manifest.head(), pd.crosstab(manifest.dataset, manifest.emotion), manifest.groupby("dataset").speaker_id.nunique())'''),
md('''## 4. EDA and speaker-disjoint split

TESS has only two speakers, so one is assigned to train and one to test. SAVEE's four speakers are assigned 2/1/1. Larger datasets use deterministic 70/15/15 speaker partitions.'''),
code('''def assign_speaker_splits(df, seed=CFG.seed):
    rng=np.random.default_rng(seed); mapping={}
    for dataset, group in df.groupby("dataset"):
        speakers=np.array(sorted(group.speaker_id.unique())); rng.shuffle(speakers); n=len(speakers)
        if dataset=="tess":
            if n!=2: raise ValueError("TESS split requires two speakers")
            cuts=(1,1)
        elif dataset=="savee":
            if n!=4: raise ValueError("SAVEE split requires four speakers")
            cuts=(2,3)
        else:
            if n<3: raise ValueError(f"At least three speakers are required for {dataset}")
            n_train=max(1,round(.70*n)); n_val=max(1,round(.15*n)); cuts=(n_train,min(n-1,n_train+n_val))
        for s in speakers[:cuts[0]]: mapping[s]="train"
        for s in speakers[cuts[0]:cuts[1]]: mapping[s]="validation"
        for s in speakers[cuts[1]:]: mapping[s]="test"
    out=df.copy(); out["split"]=out.speaker_id.map(mapping)
    assert out.split.notna().all()
    sets={k:set(g.speaker_id) for k,g in out.groupby("split")}
    assert sets["train"].isdisjoint(sets["validation"]|sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])
    assert not out.path.duplicated().any()
    for split in ("train", "validation", "test"):
        assert set(out.loc[out.split==split, "emotion"])==set(LABELS), f"Missing emotion in {split}"
    assert out.groupby("source_id").split.nunique().max()==1
    return out

manifest=assign_speaker_splits(manifest)
split_fingerprint=manifest.groupby("speaker_id").split.first().sort_index().to_dict()
assert split_fingerprint==assign_speaker_splits(manifest.drop(columns="split")).groupby("speaker_id").split.first().sort_index().to_dict()
manifest.to_csv(ARTIFACT_DIR/"manifest.csv", index=False)
display(pd.crosstab(manifest.split,manifest.emotion),pd.crosstab(manifest.split,manifest.dataset))
sns.catplot(data=manifest,x="emotion",col="split",kind="count",sharey=False,height=3,aspect=1.4)
plt.show()'''),
md('''## 5. Audio standardization and training-only augmentation'''),
code('''def preprocess_audio(audio, sample_rate, config=CFG):
    if not isinstance(sample_rate, (int, float, np.integer)) or not np.isfinite(sample_rate) or sample_rate<=0:
        raise ValueError("Sample rate must be a positive number")
    y=np.asarray(audio,dtype=np.float32)
    if y.ndim not in (1,2): raise ValueError("Expected mono or stereo audio")
    if y.ndim==2: y=y.mean(axis=0 if y.shape[0]<y.shape[1] else 1)
    y=np.ravel(y)
    if y.size==0 or not np.isfinite(y).all(): raise ValueError("Audio is empty or non-finite")
    original_seconds=float(y.size/sample_rate)
    if original_seconds>config.max_input_seconds: raise ValueError(f"Audio must be at most {config.max_input_seconds:g} seconds")
    if sample_rate!=config.sample_rate: y=librosa.resample(y,orig_sr=sample_rate,target_sr=config.sample_rate)
    y=y-y.mean(); y,_=librosa.effects.trim(y,top_db=config.top_db)
    if y.size==0 or np.max(np.abs(y))<1e-6: raise ValueError("Audio is silent")
    y=.95*y/np.max(np.abs(y)); target=int(config.sample_rate*config.duration)
    if len(y)>target:
        start=(len(y)-target)//2; y=y[start:start+target]
    else:
        total=target-len(y); y=np.pad(y,(total//2,total-total//2))
    return y.astype(np.float32), {"sample_rate":config.sample_rate,"samples":target,"original_seconds":original_seconds}

def load_audio_file(path, config=CFG):
    path=Path(path)
    if not path.is_file(): raise ValueError("Audio file does not exist")
    if path.suffix.lower() not in {".wav", ".flac", ".ogg"}:
        raise ValueError("Use a WAV, FLAC, or OGG recording")
    try:
        info=sf.info(str(path))
        if info.frames==0: raise ValueError("Audio is empty")
        if info.duration>config.max_input_seconds:
            raise ValueError(f"Audio must be at most {config.max_input_seconds:g} seconds")
        wave, sr=sf.read(str(path), dtype="float32", always_2d=False)
    except ValueError: raise
    except (OSError, RuntimeError):
        raise ValueError("Audio could not be decoded; upload a valid recording") from None
    return wave, sr

def augment_audio(y, sr, rng):
    choices=list(rng.choice(["noise","stretch","pitch","gain"],size=int(rng.integers(1,3)),replace=False))
    out=y.copy()
    for name in choices:
        if name=="noise":
            snr=float(rng.uniform(15,30)); rms=np.sqrt(np.mean(out**2)+1e-12)
            noise=rng.normal(size=len(out)).astype(np.float32); noise*=rms/(10**(snr/20)*(np.sqrt(np.mean(noise**2))+1e-12)); out+=noise
        elif name=="stretch": out=librosa.effects.time_stretch(out,rate=float(rng.uniform(.9,1.1)))
        elif name=="pitch": out=librosa.effects.pitch_shift(out,sr=sr,n_steps=float(rng.uniform(-1.5,1.5)))
        else: out*=10**(float(rng.uniform(-3,3))/20)
    target=int(CFG.sample_rate*CFG.duration); out=librosa.util.fix_length(out,size=target)
    return np.clip(out,-1.0,1.0).astype(np.float32)  # Preserve gain augmentation.

sample,sr=load_audio_file(manifest.iloc[0].path)
clean,meta=preprocess_audio(sample,sr); assert clean.shape==(int(CFG.sample_rate*CFG.duration),) and np.isfinite(clean).all()
for seconds in (.1, 7.0):
    t=np.arange(int(seconds*22050))/22050
    tone=(.2*np.sin(2*np.pi*220*t)).astype(np.float32)
    for wave in (tone, np.column_stack([tone,tone]), np.stack([tone,tone])):
        checked,_=preprocess_audio(wave,22050)
        assert checked.shape==clean.shape and np.isfinite(checked).all()
for invalid in (np.array([],dtype=np.float32), np.zeros(800), np.array([np.nan])):
    try: preprocess_audio(invalid,16000)
    except ValueError: pass
    else: raise AssertionError("Invalid audio was accepted")
print("Audio preprocessing smoke checks passed."); meta'''),
md('''## 6. Time-major feature extraction and caching

The expanded tensor contains 44 channels: 13 MFCC, 13 delta, 13 delta-delta, RMS, ZCR, centroid, bandwidth, and rolloff. It is transposed to `(frames, channels)` before entering recurrent layers.'''),
code('''FEATURE_SETS={
 "mfcc":list(range(CFG.n_mfcc)),
 "mfcc_deltas":list(range(3*CFG.n_mfcc)),
 "core":list(range(3*CFG.n_mfcc+2)),
 "expanded":list(range(3*CFG.n_mfcc+5)),
}

def extract_features(y,sr=CFG.sample_rate,config=CFG):
    kw=dict(n_fft=config.n_fft,hop_length=config.hop_length)
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=config.n_mfcc,**kw)
    channels=[mfcc,librosa.feature.delta(mfcc),librosa.feature.delta(mfcc,order=2),
              librosa.feature.rms(y=y,frame_length=config.frame_length,hop_length=config.hop_length),
              librosa.feature.zero_crossing_rate(y,frame_length=config.frame_length,hop_length=config.hop_length),
              librosa.feature.spectral_centroid(y=y,sr=sr,**kw),
              librosa.feature.spectral_bandwidth(y=y,sr=sr,**kw),
              librosa.feature.spectral_rolloff(y=y,sr=sr,**kw)]
    frames=min(x.shape[1] for x in channels); x=np.concatenate([a[:,:frames] for a in channels],axis=0).T.astype(np.float32)
    if not np.isfinite(x).all(): raise ValueError("Non-finite features")
    return x

def build_feature_cache(df,force=False):
    signature={"pipeline_version":2,"config":asdict(CFG),"labels":LABELS,
               "librosa":version("librosa"),"numpy":version("numpy"),
               "manifest":df[["path","source_id","speaker_id","emotion","split"]].to_dict("records"),
               "files":[(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in df.path]}
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
    cache=CACHE_DIR/f"expanded_features_{fingerprint}.npz"
    if cache.exists() and not force:
        with np.load(cache,allow_pickle=False) as z:
            return tuple(z[k] for k in ("X","y","splits","datasets","paths","augmented"))
    total=len(df)+int((df.split=="train").sum())
    X=None; ys=[]; splits=[]; datasets=[]; paths=[]; augmented=[]; cursor=0
    for i,row in enumerate(tqdm(df.itertuples(index=False),total=len(df),desc="Extracting features")):
        try:
            raw,sr=load_audio_file(row.path); clean,_=preprocess_audio(raw,sr)
        except ValueError as exc:
            raise ValueError(f"Invalid corpus recording {row.path}: {exc}") from exc
        variants=[(clean,False)]
        if row.split=="train": variants.append((augment_audio(clean,CFG.sample_rate,np.random.default_rng(CFG.seed+i)),True))
        for wave,is_aug in variants:
            feature=extract_features(wave)
            if X is None: X=np.empty((total,*feature.shape),dtype=np.float32)
            if feature.shape!=X.shape[1:]: raise ValueError(f"Unexpected feature shape for {row.path}")
            X[cursor]=feature; cursor+=1
            ys.append(LABEL_TO_ID[row.emotion]); splits.append(row.split)
            datasets.append(row.dataset); paths.append(row.path); augmented.append(is_aug)
    arrays=(X,np.array(ys),np.array(splits),np.array(datasets),np.array(paths),np.array(augmented))
    temporary=cache.with_suffix(".partial.npz")
    np.savez_compressed(temporary,X=arrays[0],y=arrays[1],splits=arrays[2],datasets=arrays[3],paths=arrays[4],augmented=arrays[5])
    temporary.replace(cache)
    return arrays

X_all,y_all,splits_all,datasets_all,paths_all,augmented_all=build_feature_cache(manifest)
assert X_all.ndim==3 and X_all.shape[1:]==(1+int(CFG.sample_rate*CFG.duration)//CFG.hop_length,3*CFG.n_mfcc+5)
assert np.isfinite(X_all).all()
assert not augmented_all[splits_all!="train"].any()
assert set(paths_all)==set(manifest.path)
expected_split=manifest.set_index("path").split.to_dict()
assert all(expected_split[p]==s for p,s in zip(paths_all,splits_all))
X_all.shape'''),
md('''## 7. Training-only scaling and validation feature ablation'''),
code('''def scaled_views(feature_name):
    cols=FEATURE_SETS[feature_name]; X=np.ascontiguousarray(X_all[:,:,cols]); train=splits_all=="train"
    scaler=StandardScaler()
    for start in range(0,len(X),128):
        batch=X[start:start+128][train[start:start+128]]
        if len(batch): scaler.partial_fit(batch.reshape(-1,len(cols)))
    assert scaler.n_samples_seen_==int(train.sum())*X.shape[1]
    for start in range(0,len(X),128):
        batch=X[start:start+128]
        scaler.transform(batch.reshape(-1,len(cols)),copy=False)
        assert np.isfinite(batch).all()
    return X,scaler

def make_model(name,input_shape,num_classes=len(LABELS)):
    inputs=keras.Input(shape=input_shape)
    if name=="lstm": x=layers.LSTM(64,return_sequences=True)(inputs); x=layers.LSTM(64)(x)
    elif name=="bilstm": x=layers.Bidirectional(layers.LSTM(64,return_sequences=True))(inputs); x=layers.Bidirectional(layers.LSTM(64))(x)
    elif name=="cnn_bilstm":
        x=layers.Conv1D(64,5,padding="same",activation="relu")(inputs); x=layers.BatchNormalization()(x)
        x=layers.MaxPooling1D(2)(x); x=layers.Bidirectional(layers.LSTM(64))(x)
    else: raise ValueError(name)
    x=layers.Dropout(.35)(x); outputs=layers.Dense(num_classes,activation="softmax")(x)
    model=keras.Model(inputs,outputs,name=name)
    model.compile(keras.optimizers.Adam(CFG.learning_rate),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

def macro_f1(y_true,probs):
    return precision_recall_fscore_support(y_true,np.argmax(probs,axis=1),labels=np.arange(len(LABELS)),average="macro",zero_division=0)[2]

def feature_ablation():
    rows=[]; train=splits_all=="train"; val=splits_all=="validation"
    weights=compute_class_weight("balanced",classes=np.arange(len(LABELS)),y=y_all[train])
    class_weights={i:float(w) for i,w in enumerate(weights)}
    for name in FEATURE_SETS:
        print("Feature ablation:",name)
        set_seed(); X,_=scaled_views(name); model=make_model("lstm",X.shape[1:])
        cb=keras.callbacks.EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)
        model.fit(X[train],y_all[train],validation_data=(X[val],y_all[val]),epochs=CFG.ablation_epochs,batch_size=CFG.batch_size,class_weight=class_weights,verbose=2,callbacks=[cb])
        probs=model.predict(X[val],verbose=0); rows.append({"feature_set":name,"channels":X.shape[-1],"val_macro_f1":macro_f1(y_all[val],probs)})
        keras.backend.clear_session()
        del X,model; gc.collect()
    result=pd.DataFrame(rows); best=result.val_macro_f1.max()
    selected=result[result.val_macro_f1>=best-.005].sort_values("channels").iloc[0].feature_set
    return result,selected

expected_frames=1+int(CFG.sample_rate*CFG.duration)//CFG.hop_length
for name in ("lstm","bilstm","cnn_bilstm"):
    set_seed(); candidate=make_model(name,(expected_frames,len(FEATURE_SETS["core"])))
    probabilities=candidate(np.zeros((2,expected_frames,len(FEATURE_SETS["core"])),dtype=np.float32),training=False).numpy()
    assert probabilities.shape==(2,len(LABELS)) and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1),1,atol=1e-5)
    del candidate; keras.backend.clear_session()
print("All three model forward-pass checks passed.")
ablation_results,SELECTED_FEATURE_SET=feature_ablation()
display(ablation_results.sort_values("val_macro_f1",ascending=False)); print("Selected:",SELECTED_FEATURE_SET)'''),
md('''## 8. Fair LSTM, BiLSTM, and CNN-BiLSTM comparison'''),
code('''X,scaler=scaled_views(SELECTED_FEATURE_SET)
train=splits_all=="train"; val=splits_all=="validation"; test=splits_all=="test"
weights=compute_class_weight("balanced",classes=np.arange(len(LABELS)),y=y_all[train])
class_weights={i:float(w) for i,w in enumerate(weights)}

def evaluate_probabilities(y_true,probs):
    assert probs.shape==(len(y_true),len(LABELS)) and np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1),1,atol=1e-4)
    pred=probs.argmax(1); p,r,f,_=precision_recall_fscore_support(y_true,pred,labels=np.arange(len(LABELS)),average="macro",zero_division=0)
    wp,wr,wf,_=precision_recall_fscore_support(y_true,pred,labels=np.arange(len(LABELS)),average="weighted",zero_division=0)
    return {"accuracy":accuracy_score(y_true,pred),"macro_precision":p,"macro_recall":r,"macro_f1":f,
            "weighted_precision":wp,"weighted_recall":wr,"weighted_f1":wf}

def train_model(name):
    set_seed(); model=make_model(name,X.shape[1:]); checkpoint=ARTIFACT_DIR/f"{name}.keras"
    callbacks=[keras.callbacks.ModelCheckpoint(checkpoint,monitor="val_loss",save_best_only=True),
      keras.callbacks.EarlyStopping(monitor="val_loss",patience=8,restore_best_weights=True),
      keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=4,min_lr=1e-6)]
    history=model.fit(X[train],y_all[train],validation_data=(X[val],y_all[val]),epochs=CFG.max_epochs,
                      batch_size=CFG.batch_size,class_weight=class_weights,callbacks=callbacks,verbose=1)
    (ARTIFACT_DIR/f"{name}_history.json").write_text(json.dumps(history.history),encoding="utf-8")
    return keras.models.load_model(checkpoint,compile=False),history

models={}; histories={}; rows=[]
for name in ("lstm","bilstm","cnn_bilstm"):
    model,history=train_model(name); models[name]=model; histories[name]=history
    vm=evaluate_probabilities(y_all[val],model.predict(X[val],verbose=0))
    rows.append({"model":name,"parameters":model.count_params(),"best_epoch":int(np.argmin(history.history["val_loss"])+1),
                 "val_accuracy":vm["accuracy"],"val_macro_f1":vm["macro_f1"]})
comparison=pd.DataFrame(rows).sort_values(["val_macro_f1","val_accuracy"],ascending=False).reset_index(drop=True)
BEST_MODEL_NAME=comparison.iloc[0].model
comparison.to_csv(ARTIFACT_DIR/"validation_comparison.csv",index=False)
display(comparison); print("Frozen selection:",BEST_MODEL_NAME)'''),
md('''## 9. Locked test evaluation

Run this cell only after the feature set and model selection above are complete.'''),
code('''for i,row in comparison.iterrows():
    probs=models[row.model].predict(X[test],verbose=0); metrics=evaluate_probabilities(y_all[test],probs)
    for key,value in metrics.items(): comparison.loc[i,"test_"+key]=value
comparison.to_csv(ARTIFACT_DIR/"model_comparison.csv",index=False)
display(comparison)

best_model=models[BEST_MODEL_NAME]; test_probs=best_model.predict(X[test],verbose=0); test_pred=test_probs.argmax(1)
print(classification_report(y_all[test],test_pred,labels=np.arange(len(LABELS)),target_names=LABELS,digits=4,zero_division=0))
report=classification_report(y_all[test],test_pred,labels=np.arange(len(LABELS)),target_names=LABELS,output_dict=True,zero_division=0)
pd.DataFrame(report).T.to_csv(ARTIFACT_DIR/"test_classification_report.csv")
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,norm,title in zip(axes,[None,"true"],["Counts","Row-normalized"]):
    sns.heatmap(confusion_matrix(y_all[test],test_pred,labels=np.arange(len(LABELS)),normalize=norm),annot=True,fmt="d" if norm is None else ".2f",cmap="Blues",xticklabels=LABELS,yticklabels=LABELS,ax=ax)
    ax.set(title=title,xlabel="Predicted",ylabel="True")
plt.tight_layout(); plt.savefig(ARTIFACT_DIR/"confusion_matrices.png",dpi=150); plt.show()

per_dataset=[]
for source in sorted(set(datasets_all[test])):
    mask=test & (datasets_all==source); m=evaluate_probabilities(y_all[mask],best_model.predict(X[mask],verbose=0)); m["dataset"]=source; m["samples"]=int(mask.sum()); per_dataset.append(m)
per_dataset=pd.DataFrame(per_dataset).set_index("dataset")
per_dataset.to_csv(ARTIFACT_DIR/"test_per_dataset.csv"); display(per_dataset)
errors=np.flatnonzero(test_pred!=y_all[test])[:3]
for i in errors:
    path=paths_all[test][i]
    print(f"True: {LABELS[y_all[test][i]]}; predicted: {LABELS[test_pred[i]]}; source: {datasets_all[test][i]}")
    display(Audio(filename=path))'''),
code('''fig,axes=plt.subplots(len(histories),2,figsize=(12,4*len(histories)))
for row,(name,h) in enumerate(histories.items()):
    axes[row,0].plot(h.history["loss"],label="train"); axes[row,0].plot(h.history["val_loss"],label="validation"); axes[row,0].set_title(f"{name}: loss"); axes[row,0].legend()
    axes[row,1].plot(h.history["accuracy"],label="train"); axes[row,1].plot(h.history["val_accuracy"],label="validation"); axes[row,1].set_title(f"{name}: accuracy"); axes[row,1].legend()
plt.tight_layout(); plt.savefig(ARTIFACT_DIR/"training_curves.png",dpi=150); plt.show()

confidence=test_probs.max(1); correct=(test_pred==y_all[test]); bins=np.linspace(0,1,11); ids=np.clip(np.digitize(confidence,bins)-1,0,9)
ece=sum(np.mean(ids==i)*abs(correct[ids==i].mean()-confidence[ids==i].mean()) for i in range(10) if np.any(ids==i))
print(f"Expected calibration error: {ece:.4f}")
occupied=[i for i in range(10) if np.any(ids==i)]
plt.figure(figsize=(5,5)); plt.plot([0,1],[0,1],"--",color="gray")
plt.plot([confidence[ids==i].mean() for i in occupied],[correct[ids==i].mean() for i in occupied],"o-")
plt.xlabel("Mean predicted confidence"); plt.ylabel("Empirical accuracy"); plt.title("Test reliability diagram")
plt.tight_layout(); plt.savefig(ARTIFACT_DIR/"reliability_diagram.png",dpi=150); plt.show()
(ARTIFACT_DIR/"calibration.json").write_text(json.dumps({"expected_calibration_error":float(ece)}))'''),
md('''## 10. Save a complete inference bundle'''),
code('''best_model.save(ARTIFACT_DIR/"best_model.keras")
joblib.dump(scaler,ARTIFACT_DIR/"feature_scaler.joblib")
metadata={"labels":LABELS,"label_to_id":LABEL_TO_ID,"config":asdict(CFG),"feature_set":SELECTED_FEATURE_SET,
          "feature_columns":FEATURE_SETS[SELECTED_FEATURE_SET],"model_name":BEST_MODEL_NAME,
          "input_shape":list(X.shape[1:]),"pipeline_version":2,
          "package_versions":{p:version(p) for p in ("tensorflow","librosa","soundfile","numpy","scikit-learn","gradio")}}
(ARTIFACT_DIR/"metadata.json").write_text(json.dumps(metadata,indent=2))
ablation_results.to_csv(ARTIFACT_DIR/"feature_ablation.csv",index=False)
print(list(ARTIFACT_DIR.iterdir()))'''),
md('''## 11. Clean single-file inference and reload verification'''),
code('''class EmotionPredictor:
    def __init__(self,artifact_dir=ARTIFACT_DIR):
        root=Path(artifact_dir); self.model=keras.models.load_model(root/"best_model.keras",compile=False)
        self.scaler=joblib.load(root/"feature_scaler.joblib"); self.meta=json.loads((root/"metadata.json").read_text())
        self.config=Config(**self.meta["config"])
        labels=self.meta["labels"]; columns=self.meta["feature_columns"]
        if len(set(labels))!=len(labels) or self.meta["label_to_id"]!={k:i for i,k in enumerate(labels)}:
            raise ValueError("Saved label metadata is inconsistent")
        if self.scaler.n_features_in_!=len(columns) or self.model.input_shape[-1]!=len(columns) or self.model.output_shape[-1]!=len(labels):
            raise ValueError("Model, scaler, and labels do not match")
        if tuple(self.model.input_shape[1:])!=tuple(self.meta["input_shape"]):
            raise ValueError("Saved input shape does not match the model")
    def predict_emotion(self,audio_input):
        if audio_input is None: raise ValueError("Provide an audio file")
        if isinstance(audio_input,tuple):
            if len(audio_input)!=2: raise ValueError("Audio tuple must contain (sample_rate, waveform)")
            sr,wave=audio_input
        else: wave,sr=load_audio_file(audio_input,config=self.config)
        wave,_=preprocess_audio(wave,sr,config=self.config)
        feat=extract_features(wave,sr=self.config.sample_rate,config=self.config)[:,self.meta["feature_columns"]]
        if feat.shape!=tuple(self.meta["input_shape"]): raise ValueError("Audio features do not match the saved model")
        feat=self.scaler.transform(feat).astype(np.float32)[None,...]
        probs=self.model.predict(feat,verbose=0)[0]; labels=self.meta["labels"]
        if not np.isfinite(probs).all() or not np.isclose(probs.sum(),1,atol=1e-4): raise RuntimeError("Invalid probabilities")
        idx=int(np.argmax(probs)); return {"emotion":labels[idx],"confidence":float(probs[idx]),"probabilities":{k:float(v) for k,v in zip(labels,probs)}}

predictor=EmotionPredictor()
def predict_emotion(audio_input):
    return predictor.predict_emotion(audio_input)

sample_path=paths_all[np.flatnonzero(test)[0]]; before=best_model.predict(X[np.flatnonzero(test)[0]:np.flatnonzero(test)[0]+1],verbose=0)[0]
after=predictor.predict_emotion(sample_path); assert np.allclose(before,list(after["probabilities"].values()),atol=1e-5)
print("Saved-model predictions match the in-memory model.")
after'''),
md('''## 12. Gradio demo'''),
code('''import gradio as gr

def gradio_predict(audio):
    try: result=predict_emotion(audio)
    except ValueError as exc: raise gr.Error(str(exc)) from None
    return result["emotion"],f"{result['confidence']:.1%}",result["probabilities"]

if "demo" in globals(): demo.close()
with gr.Blocks(title="Speech Emotion Recognition") as demo:
    gr.Markdown("# Speech Emotion Recognition")
    audio=gr.Audio(sources=["upload","microphone"],type="filepath",label="Speech sample")
    run=gr.Button("Analyze emotion",variant="primary")
    with gr.Row(): predicted=gr.Textbox(label="Predicted emotion"); confidence=gr.Textbox(label="Confidence")
    probabilities=gr.Label(num_top_classes=6,label="Emotion probabilities")
    run.click(gradio_predict,audio,[predicted,confidence,probabilities],concurrency_limit=1)
demo.launch(share=True, debug=False)'''),
md('''## 13. Interpretation and limitations

- Speaker-disjoint evaluation tests generalization to unseen voices instead of memorization.
- Time-major tensors let recurrent layers model prosodic change through time.
- Feature ablation keeps additional descriptors only when validation macro F1 supports them.
- The locked test set is not used for preprocessing, tuning, early stopping, or architecture selection.
- These acted English corpora differ in microphones, scripts, speakers, and recording conditions. TESS has only two speakers. Results do not imply reliable real-world emotion or mental-state detection.

### Interview explanation

“I began by fixing the experimental protocol, not by making the network larger. I grouped recordings by speaker before splitting, kept augmentation inside training, and fitted scaling only on training frames. The acoustic tensor is time-major, with MFCC dynamics and prosodic features as channels. I used a validation ablation to justify the feature set, then compared LSTM, BiLSTM, and CNN-BiLSTM under identical conditions. The architecture was selected on validation macro F1, and the untouched test set was evaluated once. Finally, I serialized the model, scaler, labels, and preprocessing configuration so notebook evaluation and Gradio inference execute the same pipeline.”''')
]

notebook = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "colab": {"name": "ser_professional_colab.ipynb", "provenance": []},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.x"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
for index, cell in enumerate(cells): cell["id"] = f"ser-cell-{index:02d}"
(Path(__file__).resolve().parent/"ser_professional_colab.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8", newline="\n")
print(f"Wrote ser_professional_colab.ipynb with {len(cells)} cells")

from generate_documentation import generate

updated = generate(Path(__file__).resolve().parent)
print("Updated documentation: " + ", ".join(updated) if updated else "Documentation is up to date.")
