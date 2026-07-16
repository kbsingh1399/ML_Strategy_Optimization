# Quant Audit & Optimization Report (Merge-Ready)

**Status:** PASS
**Base:** origin/master (backtesting_data path fallback preserved)
**Simulator:** `optimization/oos_simulator.py` with full ATR trailing-stop logic
**Parquet files:** 32

## Mean Average R

| Strategy | Mean Avg R |
|---|---|
| AlphaSqueezer_V17 | **5.470R** |
| ML_Trend_Pull | **5.432R** |
| ML_Liquidation_Runner | **5.944R** |

## Reproduce

```bash
python scripts/generate_backtesting_data.py   # if parquets missing
python optimization/run_full_optimization_loop.py
# or:
python optimization/oos_simulator.py --strategy both --trials 16 --write
python optimization/oos_simulator_liquidation.py --trials 12 --write
```

## Scripts committed

- optimization/oos_simulator.py
- optimization/oos_simulator_liquidation.py
- optimization/run_full_optimization_loop.py
- scripts/generate_backtesting_data.py
- data_paths.py

## Merge conflict resolution

Preserved master's `local_backtest_dir` / `DEFAULT_DIR` path fallback in:
- alpha_squeezer_v17/unified_backtest.py
- alpha_squeezer_v17/model_trainer.py
- ml_trend_pull/unified_backtest.py
- ml_trend_pull/model_trainer.py
- Liquidation/features.py
