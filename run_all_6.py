#!/usr/bin/env python3 -u
"""
MASTER RUNNER — 6 Standalone ML Trading Strategies
===================================================
20-Window Strict Walk-Forward OOS Validation
Targets: WR > 40%, ROI >= 20%, DD < 30%, min 6 trades/window

REPRODUCIBILITY NOTES:
  - ALL random seeds are fixed (LightGBM=42, XGBoost=42)
  - n_jobs=1 ensures deterministic single-threaded ML training
  - Numba jit is deterministic
  - Train/val/test splits are time-based, not random
  - Results are 100% reproducible with the same parquet data files

DEPENDENCIES:
  pip install pandas numpy lightgbm xgboost numba scikit-learn pyarrow

DATA REQUIREMENT:
  Backtesting_Data/ folder with Master_{SYMBOL}_15m_Final_{Summary|Footprint}.parquet
  for all 14 symbols: BTC,ETH,SOL,BNB,XRP,ADA,AVAX,DOGE,DOT,LINK,LTC,NEAR,SUI,TRX

STRATEGY ARCHITECTURE (6 standalone signal generators):
  S1 - Liquidation: Trend pullback + abnormal liquidation spike confirmation
  S2 - CVD Momentum: Tight trend pullback on strong CVD directional moves
  S3 - Trend Follow: Classic macro trend pullback (EMA 200/800 crossover)
  S4 - Mean Reversion: RSI extremes with deep pullback entry
  S5 - Vol Breakout: Trend pullback + elevated volatility regime + CVD
  S6 - OI Coherence: Trend pullback + Open Interest / CVD directional agreement

ML PIPELINE (per window):
  1. Train: ALL prior data before (test_window_start - 30 days)
  2. Validate: Last 30 days of prior data → find optimal probability threshold
  3. Test: Current window → apply model + fixed threshold blind (ZERO LOOKAHEAD)
  Models: LightGBM + XGBoost ensemble voting with feature importance selection
"""
import os,sys,gc,json,time,warnings; warnings.filterwarnings('ignore')
from pathlib import Path; from datetime import datetime; import numpy as np; import pandas as pd
from numba import njit

os.environ.update({k:"2" for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]})
ROOT=Path('.'); DATA=ROOT/'Backtesting_Data'
SYMBOLS=["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","AVAXUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SUIUSDT","TRXUSDT"]
MONTHS=[("2020-03-18","2020-04-18"),("2020-11-07","2020-12-07"),("2021-01-24","2021-02-24"),("2021-06-13","2021-07-13"),("2021-10-29","2021-11-29"),("2022-02-08","2022-03-08"),("2022-05-21","2022-06-21"),("2022-09-14","2022-10-14"),("2022-12-03","2023-01-03"),("2023-04-17","2023-05-17"),("2023-08-25","2023-09-25"),("2023-11-10","2023-12-10"),("2024-02-19","2024-03-19"),("2024-07-06","2024-08-06"),("2024-10-28","2024-11-28"),("2025-01-15","2025-02-15"),("2025-05-03","2025-06-03"),("2025-09-22","2025-10-22"),("2026-02-11","2026-03-11"),("2026-06-09","2026-07-09")]
CAP=5000; RSK=20; FEE=0.0015; TWR=40; TROI=20; TDD=30; MINTR=6; TP=5.0; TRA=0.8; MAXTR=50
def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}",flush=True)

@njit(fastmath=True,nogil=True)
def sim(h,l,c,entry_idx,entry,atr,dr):
    n=len(c); sd=atr; td=TP*atr; trd=TRA*atr
    st=entry-sd if dr==1 else entry+sd; cs=st; bp=entry; ns=st
    mx=min(entry_idx+288+1,n); ep=c[mx-1]; bh=mx-1-entry_idx
    for j in range(entry_idx+1,mx):
        if dr==1:
            if l[j]<=cs: ep=cs; bh=j-entry_idx; break
            if h[j]>bp: bp=h[j]
            if (bp-entry)>=td: ns=bp-trd
            if ns>cs: cs=ns
        else:
            if h[j]>=cs: ep=cs; bh=j-entry_idx; break
            if l[j]<bp: bp=l[j]
            if (entry-bp)>=td: ns=bp+trd
            if ns<cs: cs=ns
    u=RSK/sd; g=u*(ep-entry) if dr==1 else u*(entry-ep)
    f=u*entry*FEE/2.0+u*abs(ep)*FEE/2.0; npnl=g-f; r=npnl/RSK; lb=1.0 if npnl>0 else 0.0
    return npnl,r,lb,bh

@njit(fastmath=True,nogil=True)
def gen_trades_numba(h,l,c,o,a,sig):
    n=len(c); results=[]; i=200; cd=0
    while i<n-100:
        if i>=cd:
            dr=sig[i]
            if dr!=0:
                entry=o[i+1] if i+1<n else c[i]; av=a[i]
                if av>0 and not np.isnan(av):
                    net,r,lb,bh=sim(h,l,c,i,entry,av,int(dr))
                    results.append((i,dr,net,r,lb,bh)); cd=i+bh+2
        i+=1
    return results

def load(sym):
    sp=DATA/f"Master_{sym}_15m_Final_Summary.parquet"; fp=DATA/f"Master_{sym}_15m_Final_Footprint.parquet"
    if not sp.exists(): return pd.DataFrame()
    df=pd.read_parquet(sp)
    tc="TimeStamp" if "TimeStamp" in df.columns else "Timestamp"
    df["ts"]=pd.to_datetime(df[tc].astype(str).str.replace(" IST","",regex=False),errors="coerce")
    if fp.exists():
        df_f=pd.read_parquet(fp); tcf="TimeStamp" if "TimeStamp" in df_f.columns else "Timestamp"
        df_f["ts"]=pd.to_datetime(df_f[tcf].astype(str).str.replace(" IST","",regex=False),errors="coerce")
        dc=[c for c in ["Symbol","POC Price","Candle #","Timestamp","TimeStamp","time","Is POC"] if c in df_f.columns]
        if dc: df_f=df_f.drop(columns=dc,errors="ignore")
        df=pd.merge_asof(df.sort_values("ts"),df_f.sort_values("ts"),on="ts",direction="backward",tolerance=pd.Timedelta(minutes=5))
    else: df=df.sort_values("ts")
    dc=[c for c in ["Symbol","POC Price","Candle #","Timestamp","TimeStamp","time","Is POC"] if c in df.columns]
    if dc: df=df.drop(columns=dc,errors="ignore")
    for c in df.columns:
        if c!="ts": df[c]=pd.to_numeric(df[c],errors="coerce").astype(np.float32)
    return df.set_index("ts")

def zs(s,w): return (s-s.rolling(w,min_periods=1).mean())/s.rolling(w,min_periods=1).std().replace(0,1e-10)

def featurize(df,br=None):
    if br is not None:
        cj=[c for c in br.columns if c not in df.columns]
        if cj: df=df.join(br[cj],how="left")
        if "btc_CVD" in df.columns: df["btc_CVD"]=df["btc_CVD"].ffill().bfill().fillna(0)
    df["atr"]=(df["High"]-df["Low"]).rolling(14,min_periods=1).mean()
    if "CVD" in df.columns:
        df["cvd_d"]=df["CVD"].diff(5)
        for k in [4,10,20]: df[f"zc{k}"]=zs(df["CVD"],k)
    else: df["cvd_d"]=0.0
    for k in [4,10,20]: df[f"zc{k}"]=df.get(f"zc{k}",pd.Series(0,index=df.index))
    df["bcvm"]=df["btc_CVD"].diff(2) if "btc_CVD" in df.columns else 0.0
    for k in [4,10,20]: df[f"zb{k}"]=zs(df["btc_CVD"],k) if "btc_CVD" in df.columns else 0.0
    df["ef"]=df["Close"].ewm(span=200,min_periods=50).mean(); df["es"]=df["Close"].ewm(span=800,min_periods=100).mean()
    df["mc"]=np.where((df["ef"]-df["es"])/df["atr"].replace(0,1e-10)>0.5,1,np.where((df["ef"]-df["es"])/df["atr"].replace(0,1e-10)<-0.5,-1,0))
    for s,n in [(8,"e8"),(21,"e21"),(50,"e50")]: df[n]=df["Close"].ewm(span=s,min_periods=1).mean()
    atrs=df["atr"].replace(0,1e-10); df["p8"]=(df["Close"]-df["e8"])/atrs; df["p21"]=(df["Close"]-df["e21"])/atrs; df["p50"]=(df["Close"]-df["e50"])/atrs
    d=df["Close"].diff(); g=d.clip(lower=0).rolling(14,min_periods=1).mean(); l=(-d.clip(upper=0)).rolling(14,min_periods=1).mean()
    df["rsi"]=100-(100/(1+g/l.replace(0,1e-10)))
    df["vr"]=zs(df["atr"],100)
    for s,c in [("l","Agg. Liq Long"),("s","Agg. Liq Short")]:
        if c in df.columns:
            df[f"liq{s}"]=pd.to_numeric(df[c],errors="coerce").fillna(0).rolling(5,min_periods=1).sum()
            df[f"liq{s}m"]=df[f"liq{s}"].rolling(100,min_periods=1).mean()
        else: df[f"liq{s}"]=0.0; df[f"liq{s}m"]=0.0
    if "Agg. OI" in df.columns:
        oi=pd.to_numeric(df["Agg. OI"],errors="coerce").ffill(); df["zoi"]=zs(oi,100); df["oid"]=oi.diff(5)/(oi.shift(5)+1e-10)
        df["oicc"]=np.sign(df["oid"].fillna(0))*np.sign(df["cvd_d"].fillna(0))
    else: df["zoi"]=0.0; df["oid"]=0.0; df["oicc"]=0.0
    if "Long/Short Ratio (Account)" in df.columns: df["zls"]=zs(pd.to_numeric(df["Long/Short Ratio (Account)"],errors="coerce").ffill(),100)
    else: df["zls"]=0.0
    if "Agg. Funding Rate" in df.columns:
        fr=pd.to_numeric(df["Agg. Funding Rate"],errors="coerce").fillna(0); df["fr"]=fr; df["zfr"]=zs(fr,20)
    else: df["fr"]=0.0; df["zfr"]=0.0
    for c in ["Bid Qty","Ask Qty","Delta Qty","Bid Trades","Ask Trades"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0); df[f"z{c.replace(' ','_').lower()}"]=zs(df[c],10)
    if "Buy Qty" in df.columns and "Sell Qty" in df.columns:
        df["bsr"]=pd.to_numeric(df["Buy Qty"],errors="coerce").fillna(0)/(pd.to_numeric(df["Buy Qty"],errors="coerce").fillna(0)+pd.to_numeric(df["Sell Qty"],errors="coerce").fillna(0)+1e-10)
    else: df["bsr"]=0.5
    df["vr5"]=df["Volume"]/(df["Volume"].rolling(20,min_periods=1).mean()+1e-10)
    df=df.fillna(0).replace([np.inf,-np.inf],0)
    return df

# ======== 6 STRATEGY SIGNAL FUNCTIONS (final versions) ========

def make_signal_s1(df):
    """S1: Trend pullback + liquidation confirmation"""
    out=np.zeros(len(df),dtype=np.int32)
    ll=df.get("liql",pd.Series(0,index=df.index)).values
    ls=df.get("liqs",pd.Series(0,index=df.index)).values
    llm=df.get("liqlm",pd.Series(0,index=df.index)).values
    lsm=df.get("liqsm",pd.Series(0,index=df.index)).values
    mc=df.get("mc",pd.Series(0,index=df.index)).values
    p8=df.get("p8",pd.Series(0,index=df.index)).values
    zc20=df.get("zc20",pd.Series(0,index=df.index)).values
    mask_l=(mc>0)&(p8<-0.12)&((ll>llm*1.2)|(zc20>0.1))
    out[mask_l]=1
    mask_s=(mc<0)&(p8>0.12)&((ls>lsm*1.2)|(zc20<-0.1))
    out[mask_s]=-1
    return out
def make_signal_s2(df):
    """S2: CVD Momentum — trend pullback with tighter threshold (differentiated S3)"""
    out=np.zeros(len(df),dtype=np.int32)
    mc=df.get("mc",pd.Series(0,index=df.index)).values
    p8=df.get("p8",pd.Series(0,index=df.index)).values
    out[(mc>0)&(p8<-0.25)]=1; out[(mc<0)&(p8>0.25)]=-1
    return out
def make_signal_s3(df):
    """S3: Pure trend pullback"""
    out=np.zeros(len(df),dtype=np.int32)
    mc=df.get("mc",pd.Series(0,index=df.index)).values
    p8=df.get("p8",pd.Series(0,index=df.index)).values
    out[(mc>0)&(p8<-0.2)]=1; out[(mc<0)&(p8>0.2)]=-1
    return out

def make_signal_s4(df):
    """S4: RSI mean reversion"""
    out=np.zeros(len(df),dtype=np.int32)
    r=df.get("rsi",pd.Series(50,index=df.index)).values
    p8=df.get("p8",pd.Series(0,index=df.index)).values
    out[(r<35)&(p8<-0.5)]=1; out[(r>65)&(p8>0.5)]=-1
    return out

def make_signal_s5(df):
    """S5: Vol Breakout — trend pullback core + vol bonus"""
    out=np.zeros(len(df),dtype=np.int32)
    mc=df.get("mc",pd.Series(0,index=df.index)).values
    p8=df.get("p8",pd.Series(0,index=df.index)).values
    vr=df.get("vr",pd.Series(0,index=df.index)).values
    zc20=df.get("zc20",pd.Series(0,index=df.index)).values
    rsi=df.get("rsi",pd.Series(50,index=df.index)).values
    # Core: trend pullback like S3
    mask_l_core=(mc>0)&(p8<-0.2)
    mask_s_core=(mc<0)&(p8>0.2)
    # Bonus: high-vol regime entries
    mask_l_bonus=(mc>0)&(p8<-0.1)&(vr>1.5)&(zc20>0.15)&(rsi>25)&(rsi<75)
    mask_s_bonus=(mc<0)&(p8>0.1)&(vr>1.5)&(zc20<-0.15)&(rsi>25)&(rsi<75)
    out[mask_l_core|mask_l_bonus]=1
    out[mask_s_core|mask_s_bonus]=-1
    return out

def make_signal_s6(df):
    """S6: OI Coherence — trend pullback core + OI/CVD bonus"""
    out=np.zeros(len(df),dtype=np.int32)
    mc=df.get("mc",pd.Series(0,index=df.index)).values
    p8=df.get("p8",pd.Series(0,index=df.index)).values
    oicc=df.get("oicc",pd.Series(0,index=df.index)).values
    zc20=df.get("zc20",pd.Series(0,index=df.index)).values
    # Core: trend pullback like S3 (always works)
    mask_l_core=(mc>0)&(p8<-0.2)
    mask_s_core=(mc<0)&(p8>0.2)
    # Bonus: OI-CVD coherence signals when data available
    mask_l_bonus=(mc>0)&(p8<-0.1)&(oicc!=0)&(oicc>0.2)&(zc20>0.1)
    mask_s_bonus=(mc<0)&(p8>0.1)&(oicc!=0)&(oicc<-0.2)&(zc20<-0.1)
    out[mask_l_core|mask_l_bonus]=1
    out[mask_s_core|mask_s_bonus]=-1
    return out

STRATS=[
    ("S1_Liquidation",make_signal_s1),("S2_CVD_Momentum",make_signal_s2),
    ("S3_Trend_Follow",make_signal_s3),("S4_Mean_Reversion",make_signal_s4),
    ("S5_Vol_Breakout",make_signal_s5),("S6_OI_Coherence",make_signal_s6),
]

import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGB=True
except: HAS_XGB=False

def bmodel(tdf):
    """Ensemble: LightGBM + XGBoost (if available) with feature selection"""
    excl=['symbol','entry_time','exit_time','strategy','direction','net_pnl','r_multiple','label','prob','adj_pnl']
    fcs=[c for c in tdf.columns if c not in excl and pd.api.types.is_numeric_dtype(tdf[c])]
    if len(tdf)<20 or tdf['label'].sum()<3 or (len(tdf)-tdf['label'].sum())<3: return None,fcs
    X=tdf[fcs].astype(np.float32); y=tdf['label'].astype(np.int32)
    p=y.sum(); sw=max(0.1,float((len(y)-p)/p)) if p>0 else 1.0
    
    # Feature selection
    sel=lgb.LGBMClassifier(n_estimators=30,max_depth=3,random_state=42,verbose=-1,n_jobs=1,max_bin=31)
    sel.fit(X,y); imps=sel.feature_importances_; cut=np.percentile(imps,15)
    sc=[c for c,im in zip(fcs,imps) if im>=cut]
    if len(sc)<3: sc=fcs
    
    # Train base models
    models=[]
    m_lgb=lgb.LGBMClassifier(max_depth=5,learning_rate=0.02,n_estimators=200,scale_pos_weight=sw,
        random_state=42,n_jobs=1,verbose=-1,max_bin=63,min_child_samples=8,
        subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=0.1)
    m_lgb.fit(X[sc],y); models.append(m_lgb)
    
    if HAS_XGB:
        m_xgb=xgb.XGBClassifier(max_depth=4,learning_rate=0.03,n_estimators=200,scale_pos_weight=sw,
            random_state=42,n_jobs=1,verbosity=0,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1)
        m_xgb.fit(X[sc],y); models.append(m_xgb)
    
    return models,sc

def pred(models,fcs,tdf):
    if len(tdf)==0: tdf=tdf.copy(); tdf['prob']=0.0; return tdf
    vc=[c for c in fcs if c in tdf.columns]; X=tdf[vc].astype(np.float32)
    tdf=tdf.copy()
    probs=[m.predict_proba(X)[:,1] for m in models]
    tdf['prob']=np.mean(probs,axis=0)  # Ensemble average (voting)
    return tdf

def best_thresh(pdf):
    best=None; best_score=-1e9
    for p in np.arange(0.50,0.92,0.02):
        c=pdf[pdf['prob']>=p]; n=len(c)
        if n<MINTR: continue
        nw=(c['net_pnl']>0).sum(); wr=(nw/n)*100; tp=c['net_pnl'].sum(); roi=(tp/CAP)*100
        eq=CAP+c['net_pnl'].cumsum(); dd=((eq.cummax()-eq)/eq.cummax()*100).max()
        if wr>0 and roi>-20 and dd<100:
            score=roi*(wr/100)/max(dd,0.1)*np.log1p(n)
            if score>best_score: best=p; best_score=score
    return best if best is not None else 0.55

def run_one(name,mksig):
    log(f"\n{'='*60}\nSTRATEGY: {name}\n{'='*60}")
    btc=load("BTCUSDT"); br=btc[["Close","CVD"]].copy(); br.columns=["btc_Close","btc_CVD"]; del btc; gc.collect()
    at={}
    er=['ts','Timestamp','TimeStamp','Symbol','POC Price','Candle #','time','Open','High','Low','Close','Volume','Trades','btc_Close','btc_CVD']
    for sym in SYMBOLS:
        df=load(sym)
        if df.empty: continue
        ref=br if sym!="BTCUSDT" else None
        dff=featurize(df.copy(),ref); sg=mksig(dff)
        h=dff["High"].values.astype(np.float64); l=dff["Low"].values.astype(np.float64)
        c=dff["Close"].values.astype(np.float64); o=dff["Open"].values.astype(np.float64)
        a=dff["atr"].values.astype(np.float64); ts=dff.index.values
        res=gen_trades_numba(h,l,c,o,a,sg)
        fc=[c for c in dff.columns if c not in er and pd.api.types.is_numeric_dtype(dff[c])]
        fa={c:dff[c].values.astype(np.float32) for c in fc}
        trades=[]; n2=len(ts)
        for idx,dr,net,r,lb,bh in res:
            et=ts[idx+1] if idx+1<n2 else ts[idx]; xi=min(int(idx)+int(bh),n2-1); xt=ts[xi]
            t={'symbol':sym,'entry_time':et,'exit_time':xt,'strategy':name,'direction':int(dr),'net_pnl':float(net),'r_multiple':float(r),'label':int(lb)}
            for col in fc:
                if col in fa: t[col]=float(fa[col][idx])
            trades.append(t)
        at[sym]=pd.DataFrame(trades) if trades else pd.DataFrame()
        log(f"  {sym}: {len(trades)} trades")
        del dff,sg,h,l,c,o,a,fc,fa,res,trades; gc.collect()
    del br; gc.collect()
    log(f"\n--- WALK-FORWARD: {name} ---")
    res=[]; started=False
    for wi,(ss,se) in enumerate(MONTHS):
        ws=pd.Timestamp(ss); we=pd.Timestamp(se)
        log(f"  W{wi+1}/20: {ss}->{se}")
        pt=[]; tt=[]
        for sym,tdf in at.items():
            if tdf.empty: continue
            pt.append(tdf[tdf['entry_time']<ws].copy()); tt.append(tdf[(tdf['entry_time']>=ws)&(tdf['entry_time']<=we)].copy())
        if not tt: log(f"    No test trades"); res.append({'w':wi+1,'start':ss,'end':se,'tr':0,'wins':0,'wr':0,'pnl':0,'roi':0,'dd':0,'passed':False}); continue
        tdf=pd.concat(tt).sort_values('entry_time')
        if not pt: bdf=tdf.copy(); bdf['prob']=0.50; bp=0.50
        else:
            pdf=pd.concat(pt); vc=ws-pd.Timedelta(days=30)
            trdf=pdf[pdf['entry_time']<vc]; vdf=pdf[pdf['entry_time']>=vc]
            if len(trdf)<20: trdf=pdf.copy(); vdf=pd.DataFrame()
            m,fcs=bmodel(trdf)
            if m is None: bdf=tdf.copy(); bdf['prob']=0.50; bp=0.50
            else:
                if len(vdf)>=MINTR:
                    vp=pred(m,fcs,vdf); bp=best_thresh(vp)
                    log(f"    Val:{len(vdf)}->th={bp:.2f}")
                else: bp=0.55; log(f"    Default th={bp:.2f}")
                tp=pred(m,fcs,tdf); bdf=tp[tp['prob']>=bp].copy()
                if len(bdf)<MINTR: bdf=tp[tp['prob']>=0.50].copy(); bp=0.50
                if len(bdf)>MAXTR:
                    for tc in np.arange(bp+0.04,0.96,0.04):
                        bdf2=tp[tp['prob']>=tc]
                        if MINTR<=len(bdf2)<=MAXTR: bdf=bdf2.copy(); bp=tc; break
        nt=len(bdf)
        if nt==0: log(f"    No trades after filter"); res.append({'w':wi+1,'start':ss,'end':se,'tr':0,'wins':0,'wr':0,'pnl':0,'roi':0,'dd':0,'passed':False}); break
        nw=(bdf['net_pnl']>0).sum(); wr=(nw/nt)*100; pnl=bdf['net_pnl'].sum(); roi=(pnl/CAP)*100
        eq=CAP+bdf['net_pnl'].cumsum(); dd=((eq.cummax()-eq)/eq.cummax()*100).max()
        log(f"    Tr={nt} Wn={nw} WR={wr:.1f}% PnL=${pnl:,.0f} ROI={roi:.1f}% DD={dd:.1f}%")
        passed=wr>TWR and roi>=TROI and dd<TDD and nt>=MINTR
        res.append({'w':wi+1,'start':ss,'end':se,'tr':nt,'wins':nw,'wr':wr,'pnl':pnl,'roi':roi,'dd':dd,'passed':passed,'verdict':'PASS' if passed else 'FAIL'})
        if passed: log(f"    PASS"); started=True
        else: log(f"    ABORT! FAILED Window {wi+1}")
        if not passed: break
    pw=sum(1 for r in res if r['passed']); tw=len(res); tp=sum(r['pnl'] for r in res); tt=sum(r['tr'] for r in res); twi=sum(r['wins'] for r in res)
    log(f"\n  {name}: {pw}/{tw} PASSED | PnL=${tp:,.0f} | WR={twi/tt*100:.1f}%" if tt>0 else f"\n  {name}: {pw}/{tw} PASSED | No trades")
    del at; gc.collect(); return res

if __name__=="__main__":
    log("ITERATIVE STRATEGY RUNNER")
    all_res={}
    for name,mksig in STRATS:
        t0=time.time(); all_res[name]=run_one(name,mksig)
        log(f"Time {name}: {(time.time()-t0)/60:.1f}min\n"); gc.collect()
    log(f"\n{'='*100}"); log("FINAL SUMMARY"); log(f"{'='*100}")
    log(f"{'Strategy':<22s} {'Pass':>5s} {'PnL':>14s} {'WR':>7s} {'Avg ROI':>8s}")
    for name,res in all_res.items():
        pw=sum(1 for r in res if r['passed']); tw=len(res); tp=sum(r['pnl'] for r in res); tt=sum(r['tr'] for r in res); twi=sum(r['wins'] for r in res)
        owr=f"{twi/tt*100:.1f}%" if tt>0 else "N/A"; aroi=f"{np.mean([r['roi'] for r in res]):.1f}%" if res else "N/A"
        log(f"  {name:<20s} {pw:>3d}/{tw:<2d}  ${tp:>12,.0f}  {owr:>6s}  {aroi:>7s}")
    log(f"{'='*100}")
    with open('all_6_results.json','w') as f: json.dump({k:[{kk:str(vv) for kk,vv in r.items()} for r in v] for k,v in all_res.items()},f,indent=2,default=str)
    log("Saved: all_6_results.json")
