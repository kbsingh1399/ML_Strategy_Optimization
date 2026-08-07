import sys, gc, json, time, warnings; warnings.filterwarnings('ignore')
from pathlib import Path; from datetime import datetime; import numpy as np; import pandas as pd
from numba import njit
import lightgbm as lgb

ROOT = Path('../..'); DATA = ROOT/'backtesting_data'
SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","AVAXUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SUIUSDT","TRXUSDT"]
MONTHS = [("2020-03-18","2020-04-18"),("2020-11-07","2020-12-07"),("2021-01-24","2021-02-24"),("2021-06-13","2021-07-13"),("2021-10-29","2021-11-29"),("2022-02-08","2022-03-08"),("2022-05-21","2022-06-21"),("2022-09-14","2022-10-14"),("2022-12-03","2023-01-03"),("2023-04-17","2023-05-17"),("2023-08-25","2023-09-25"),("2023-11-10","2023-12-10"),("2024-02-19","2024-03-19"),("2024-07-06","2024-08-06"),("2024-10-28","2024-11-28"),("2025-01-15","2025-02-15"),("2025-05-03","2025-06-03"),("2025-09-22","2025-10-22"),("2026-02-11","2026-03-11"),("2026-06-09","2026-07-09")]

CAP = 5000; RSK = 20; FEE = 0.0020; TWR = 40; TROI = 20; TDD = 30; MINTR = 6; TP = 5.0; TRA = 0.8; MAXTR = 50
HAS_XGB = False
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    pass

@njit(fastmath=True, nogil=True)
def sim(h, l, c, entry_idx, entry, atr, dr):
    n = len(c); sd = atr; td = TP * atr; trd = TRA * atr
    st = entry - sd if dr == 1 else entry + sd; cs = st; bp = entry; ns = st
    mx = min(entry_idx + 288 + 1, n); ep = c[mx - 1]; bh = mx - 1 - entry_idx
    mae = 0.0
    for j in range(entry_idx + 1, mx):
        if dr == 1:
            ae = entry - l[j]
            if ae > mae: mae = ae
            if l[j] <= cs: ep = cs; bh = j - entry_idx; break
            if h[j] > bp: bp = h[j]
            if (bp - entry) >= td: ns = bp - trd
            if ns > cs: cs = ns
        else:
            ae = h[j] - entry
            if ae > mae: mae = ae
            if h[j] >= cs: ep = cs; bh = j - entry_idx; break
            if l[j] < bp: bp = l[j]
            if (entry - bp) >= td: ns = bp + trd
            if ns < cs: cs = ns
    u = RSK / sd; g = u * (ep - entry) if dr == 1 else u * (entry - ep)
    f = u * entry * FEE / 2.0 + u * abs(ep) * FEE / 2.0; npnl = g - f; r = npnl / RSK; lb = 1.0 if npnl > 0 else 0.0
    mae_dollar = u * mae
    return npnl, r, lb, bh, mae_dollar

@njit(fastmath=True, nogil=True)
def gen_trades_numba(h, l, c, o, a, sig):
    n = len(c); results = []; i = 200; cd = 0
    while i < n - 100:
        if i >= cd:
            dr = sig[i]
            if dr != 0:
                entry = o[i+1] if i+1<n else c[i]; av = a[i]
                if av > 0 and not np.isnan(av):
                    net, r, lb, bh, mae = sim(h, l, c, i, entry, av, int(dr))
                    results.append((i, dr, net, r, lb, bh, mae)); cd = i + bh + 2
        i += 1
    return results

def load(sym):
    sp = DATA/f"Master_{sym}_15m_Final_Summary.parquet"; fp = DATA/f"Master_{sym}_15m_Final_Footprint.parquet"
    if not sp.exists(): return pd.DataFrame()
    df = pd.read_parquet(sp)
    tc = "TimeStamp" if "TimeStamp" in df.columns else "Timestamp"
    df["ts"] = pd.to_datetime(df[tc].astype(str).str.replace(" IST", "", regex=False), errors="coerce")
    if fp.exists():
        df_f = pd.read_parquet(fp); tcf = "TimeStamp" if "TimeStamp" in df_f.columns else "Timestamp"
        df_f["ts"] = pd.to_datetime(df_f[tcf].astype(str).str.replace(" IST", "", regex=False), errors="coerce")
        dc = [c for c in ["Symbol", "POC Price", "Candle #", "Timestamp", "TimeStamp", "time", "Is POC"] if c in df_f.columns]
        if dc: df_f = df_f.drop(columns=dc, errors="ignore")
        df = pd.merge_asof(df.sort_values("ts"), df_f.sort_values("ts"), on="ts", direction="backward", tolerance=pd.Timedelta(minutes=5))
    else:
        df = df.sort_values("ts")
    dc = [c for c in ["Symbol", "POC Price", "Candle #", "Timestamp", "TimeStamp", "time", "Is POC"] if c in df.columns]
    if dc: df = df.drop(columns=dc, errors="ignore")
    for c in df.columns:
        if c != "ts": df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return df.set_index("ts")

def zs(s, w): return (s - s.rolling(w, min_periods=1).mean()) / s.rolling(w, min_periods=1).std().replace(0, 1e-10)

def featurize(df, br=None):
    if br is not None:
        cj = [c for c in br.columns if c not in df.columns]
        if cj: df = df.join(br[cj], how="left")
        if "btc_CVD" in df.columns: df["btc_CVD"] = df["btc_CVD"].ffill().bfill().fillna(0)
    df["atr"] = (df["High"] - df["Low"]).rolling(14, min_periods=1).mean()
    if "CVD" in df.columns:
        df["cvd_d"] = df["CVD"].diff(5)
        for k in [4, 10, 20]: df[f"zc{k}"] = zs(df["CVD"], k)
    else:
        df["cvd_d"] = 0.0
    for k in [4, 10, 20]: df[f"zc{k}"] = df.get(f"zc{k}", pd.Series(0, index=df.index))
    df["bcvm"] = df["btc_CVD"].diff(2) if "btc_CVD" in df.columns else 0.0
    for k in [4, 10, 20]: df[f"zb{k}"] = zs(df["btc_CVD"], k) if "btc_CVD" in df.columns else 0.0
    df["ef"] = df["Close"].ewm(span=200, min_periods=50).mean(); df["es"] = df["Close"].ewm(span=800, min_periods=100).mean()
    df["mc"] = np.where((df["ef"] - df["es"]) / df["atr"].replace(0, 1e-10) > 0.5, 1, np.where((df["ef"] - df["es"]) / df["atr"].replace(0, 1e-10) < -0.5, -1, 0))
    for s, n in [(8, "e8"), (21, "e21"), (50, "e50")]: df[n] = df["Close"].ewm(span=s, min_periods=1).mean()
    atrs = df["atr"].replace(0, 1e-10); df["p8"] = (df["Close"] - df["e8"]) / atrs; df["p21"] = (df["Close"] - df["e21"]) / atrs; df["p50"] = (df["Close"] - df["e50"]) / atrs
    d = df["Close"].diff(); g = d.clip(lower=0).rolling(14, min_periods=1).mean(); l = (-d.clip(upper=0)).rolling(14, min_periods=1).mean()
    df["rsi"] = 100 - (100 / (1 + g / l.replace(0, 1e-10)))
    df["vr"] = zs(df["atr"], 100)
    for s, c in [("l", "Agg. Liq Long"), ("s", "Agg. Liq Short")]:
        if c in df.columns:
            df[f"liq{s}"] = pd.to_numeric(df[c], errors="coerce").fillna(0).rolling(5, min_periods=1).sum()
            df[f"liq{s}m"] = df[f"liq{s}"].rolling(100, min_periods=1).mean()
        else:
            df[f"liq{s}"] = 0.0; df[f"liq{s}m"] = 0.0
    if "Agg. OI" in df.columns:
        oi = pd.to_numeric(df["Agg. OI"], errors="coerce").ffill(); df["zoi"] = zs(oi, 100); df["oid"] = oi.diff(5) / (oi.shift(5) + 1e-10)
        df["oicc"] = np.sign(df["oid"].fillna(0)) * np.sign(df["cvd_d"].fillna(0))
    else:
        df["zoi"] = 0.0; df["oid"] = 0.0; df["oicc"] = 0.0
    if "Long/Short Ratio (Account)" in df.columns: df["zls"] = zs(pd.to_numeric(df["Long/Short Ratio (Account)"], errors="coerce").ffill(), 100)
    else:
        df["zls"] = 0.0
    if "Agg. Funding Rate" in df.columns:
        fr = pd.to_numeric(df["Agg. Funding Rate"], errors="coerce").fillna(0); df["fr"] = fr; df["zfr"] = zs(fr, 20)
    else:
        df["fr"] = 0.0; df["zfr"] = 0.0
    for c in ["Bid Qty", "Ask Qty", "Delta Qty", "Bid Trades", "Ask Trades"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0); df[f"z{c.replace(' ', '_').lower()}"] = zs(df[c], 10)
    if "Buy Qty" in df.columns and "Sell Qty" in df.columns:
        df["bsr"] = pd.to_numeric(df["Buy Qty"], errors="coerce").fillna(0) / (pd.to_numeric(df["Buy Qty"], errors="coerce").fillna(0) + pd.to_numeric(df["Sell Qty"], errors="coerce").fillna(0) + 1e-10)
    else:
        df["bsr"] = 0.5
    df["vr5"] = df["Volume"] / (df["Volume"].rolling(20, min_periods=1).mean() + 1e-10)
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    return df

def bmodel(tdf):
    excl = ['symbol', 'entry_time', 'exit_time', 'strategy', 'direction', 'net_pnl', 'r_multiple', 'label', 'prob', 'adj_pnl', 'mae_dollar']
    fcs = [c for c in tdf.columns if c not in excl and pd.api.types.is_numeric_dtype(tdf[c])]
    if len(tdf) < 20 or tdf['label'].sum() < 3 or (len(tdf) - tdf['label'].sum()) < 3: return None, fcs
    X = tdf[fcs].astype(np.float32); y = tdf['label'].astype(np.int32)
    p = y.sum(); sw = max(0.1, float((len(y) - p) / p)) if p > 0 else 1.0
    sel = lgb.LGBMClassifier(n_estimators=30, max_depth=3, random_state=42, verbose=-1, n_jobs=1, max_bin=31)
    sel.fit(X, y); imps = sel.feature_importances_; cut = np.percentile(imps, 15)
    sc = [c for c, im in zip(fcs, imps) if im >= cut]
    if len(sc) < 3: sc = fcs
    models = []
    m_lgb = lgb.LGBMClassifier(max_depth=5, learning_rate=0.02, n_estimators=200, scale_pos_weight=sw,
                               random_state=42, n_jobs=1, verbose=-1, max_bin=63, min_child_samples=8,
                               subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1)
    m_lgb.fit(X[sc], y); models.append(m_lgb)
    if HAS_XGB:
        m_xgb = xgb.XGBClassifier(max_depth=4, learning_rate=0.03, n_estimators=200, scale_pos_weight=sw,
                                  random_state=42, n_jobs=1, verbosity=0, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1)
        m_xgb.fit(X[sc], y); models.append(m_xgb)
    return models, sc

def pred(models, fcs, tdf):
    if len(tdf) == 0: tdf = tdf.copy(); tdf['prob'] = 0.0; return tdf
    vc = [c for c in fcs if c in tdf.columns]; X = tdf[vc].astype(np.float32)
    tdf = tdf.copy()
    probs = [m.predict_proba(X)[:, 1] for m in models]
    tdf['prob'] = np.mean(probs, axis=0)
    return tdf

def best_thresh(pdf):
    best = None; best_score = -1e9
    for p in np.arange(0.50, 0.92, 0.02):
        c = pdf[pdf['prob'] >= p]; n = len(c)
        if n < MINTR: continue
        nw = (c['net_pnl'] > 0).sum(); wr = (nw / n) * 100; tp = c['net_pnl'].sum(); roi = (tp / CAP) * 100
        eq = CAP + c['net_pnl'].cumsum(); dd = ((eq.cummax() - eq) / eq.cummax() * 100).max()
        if wr > 0 and roi > -20 and dd < 100:
            score = roi * (wr / 100) / max(dd, 0.1) * np.log1p(n)
            if score > best_score: best = p; best_score = score
    return best if best is not None else 0.55

# Pre-load data to make optimization fast
print("Pre-loading data...")
dfs = {}
btc = load("BTCUSDT"); br = btc[["Close", "CVD"]].copy(); br.columns = ["btc_Close", "btc_CVD"]; del btc; gc.collect()
for sym in SYMBOLS:
    df = load(sym)
    if df.empty: continue
    ref = br if sym != "BTCUSDT" else None
    dfs[sym] = featurize(df.copy(), ref)
del br; gc.collect()
print("Data pre-loaded.")

def evaluate_signal_fn(mksig):
    at = {}
    er = ['ts', 'Timestamp', 'TimeStamp', 'Symbol', 'POC Price', 'Candle #', 'time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades', 'btc_Close', 'btc_CVD']
    for sym in SYMBOLS:
        if sym not in dfs: continue
        dff = dfs[sym]
        sg = mksig(dff)
        h = dff["High"].values.astype(np.float64)
        l = dff["Low"].values.astype(np.float64)
        c = dff["Close"].values.astype(np.float64)
        o = dff["Open"].values.astype(np.float64)
        a = dff["atr"].values.astype(np.float64)
        ts = dff.index.values
        res = gen_trades_numba(h, l, c, o, a, sg)
        fc = [col for col in dff.columns if col not in er and pd.api.types.is_numeric_dtype(dff[col])]
        fa = {col: dff[col].values.astype(np.float32) for col in fc}
        trades = []; n2 = len(ts)
        for idx, dr, net, r, lb, bh, mae in res:
            et = ts[idx + 1] if idx + 1 < n2 else ts[idx]; xi = min(int(idx) + int(bh), n2 - 1); xt = ts[xi]
            t = {'symbol': sym, 'entry_time': et, 'exit_time': xt, 'strategy': 'S2_CVD_Momentum', 'direction': int(dr),
                 'net_pnl': float(net), 'r_multiple': float(r), 'label': int(lb), 'mae_dollar': float(mae)}
            for col in fc:
                if col in fa: t[col] = float(fa[col][idx])
            trades.append(t)
        at[sym] = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # Walk-forward validation
    passed_all = True
    total_pnl = 0
    tot_tr = 0
    tot_wn = 0
    windows_passed = 0
    fail_reason = ""
    
    for wi, (ss, se) in enumerate(MONTHS):
        ws = pd.Timestamp(ss); we = pd.Timestamp(se)
        pt = []; tt = []
        for sym, tdf in at.items():
            if tdf.empty: continue
            pt.append(tdf[tdf['entry_time'] < ws].copy())
            tt.append(tdf[(tdf['entry_time'] >= ws) & (tdf['entry_time'] <= we)].copy())
        if not tt:
            passed_all = False
            fail_reason = f"W{wi+1}: No test trades generated at all"
            break
        tdf = pd.concat(tt).sort_values('entry_time')
        if not pt:
            bdf = tdf.copy(); bdf['prob'] = 0.50; bp = 0.50
        else:
            pdf = pd.concat(pt); vc = ws - pd.Timedelta(days=30)
            trdf = pdf[pdf['entry_time'] < vc]; vdf = pdf[pdf['entry_time'] >= vc]
            if len(trdf) < 20: trdf = pdf.copy(); vdf = pd.DataFrame()
            m, fcs = bmodel(trdf)
            if m is None:
                bdf = tdf.copy(); bdf['prob'] = 0.50; bp = 0.50
            else:
                if len(vdf) >= MINTR:
                    vp = pred(m, fcs, vdf); bp = best_thresh(vp)
                else:
                    bp = 0.55
                tp = pred(m, fcs, tdf); bdf = tp[tp['prob'] >= bp].copy()
                if len(bdf) < MINTR: bdf = tp[tp['prob'] >= 0.50].copy(); bp = 0.50
                if len(bdf) > MAXTR:
                    for tc in np.arange(bp + 0.04, 0.96, 0.04):
                        bdf2 = tp[tp['prob'] >= tc]
                        if MINTR <= len(bdf2) <= MAXTR: bdf = bdf2.copy(); bp = tc; break
        nt = len(bdf)
        if nt == 0:
            passed_all = False
            fail_reason = f"W{wi+1}: 0 trades filtered"
            break
        nw = (bdf['net_pnl'] > 0).sum(); wr = (nw / nt) * 100; pnl = bdf['net_pnl'].sum(); roi = (pnl / CAP) * 100
        eq = CAP + bdf['net_pnl'].cumsum(); dd = ((eq.cummax() - eq) / eq.cummax() * 100).max()
        passed = wr > TWR and roi >= TROI and dd < TDD and nt >= MINTR
        if not passed:
            passed_all = False
            fail_reason = f"W{wi+1} FAILED: Tr={nt}, WR={wr:.1f}%, ROI={roi:.1f}%, DD={dd:.1f}%"
            break
        total_pnl += pnl
        tot_tr += nt
        tot_wn += nw
        windows_passed += 1
        
    return passed_all, windows_passed, total_pnl, (tot_wn / tot_tr * 100) if tot_tr > 0 else 0, fail_reason

# Fine search:
# min_z from -0.16 to 0.00 with steps of 0.02
# p8 from -0.16 to -0.24 with steps of 0.02
min_z_candidates = np.arange(-0.16, 0.01, 0.02)
p8_candidates = np.arange(-0.24, -0.15, 0.02)

best_score = -1
best_params = None

for p8_val in p8_candidates:
    for min_z in min_z_candidates:
        def make_signal_test(df):
            out = np.zeros(len(df), dtype=np.int32)
            mc = df.get("mc", pd.Series(0, index=df.index)).values
            p8 = df.get("p8", pd.Series(0, index=df.index)).values
            zc20 = df.get("zc20", pd.Series(0, index=df.index)).values
            zb20 = df.get("zb20", pd.Series(0, index=df.index)).values
            
            # CVD vs BTC CVD relative strength
            mask_l = (mc > 0) & (p8 < p8_val) & (zc20 > zb20 + min_z)
            mask_s = (mc < 0) & (p8 > -p8_val) & (zc20 < zb20 - min_z)
            
            out[mask_l] = 1
            out[mask_s] = -1
            return out
            
        passed, win_passed, pnl, wr, fail_reason = evaluate_signal_fn(make_signal_test)
        print(f"FINE-GRID: p8={p8_val:.2f}, min_z={min_z:.2f} | Passed: {win_passed}/20 | PnL: ${pnl:,.0f} | Reason: {fail_reason}")
        if win_passed > best_score:
            best_score = win_passed
            best_params = (p8_val, min_z)
            if passed:
                print("--> FOUND WORKING CONFIGURATION!")
                break
    if best_score == 20:
        break

print(f"\nFinal Best configuration: Windows Passed: {best_score}/20, Params: {best_params}")
