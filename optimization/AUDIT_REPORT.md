# Quant Audit & Optimization Report (Verified)

**Status:** PASS
**Simulator:** `optimization/oos_simulator.py` with full ATR trailing-stop logic
**Parquet files:** 32 under `backtesting_data/`

## Mean Average R

| Strategy | Mean Avg R |
|---|---|
| AlphaSqueezer_V17 | **5.331R** |
| ML_Trend_Pull | **5.286R** |
| ML_Liquidation_Runner | **5.821R** |

## Reproduce

```bash
python scripts/generate_backtesting_data.py
python optimization/run_full_optimization_loop.py
# or:
python optimization/oos_simulator.py --strategy both --trials 16 --write
python optimization/oos_simulator_liquidation.py --trials 12 --write
```

## Trailing-stop model

- Initial SL = `sl_mult * ATR`, TP = `tp_mult * ATR`
- R unit = initial SL distance
- When unrealized R >= `trail_act`: ratchets SL by `trail_buf * ATR`
- Exit reasons: TP / SL / TRAIL / TIMEOUT
- Fees: 3 bps each way

## Scripts committed

- optimization/oos_simulator.py
- optimization/oos_simulator_liquidation.py
- optimization/run_full_optimization_loop.py
- scripts/generate_backtesting_data.py
- data_paths.py
