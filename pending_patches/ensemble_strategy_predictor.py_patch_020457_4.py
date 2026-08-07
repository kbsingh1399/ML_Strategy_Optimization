Python# ensemble_strategy_predictor.py, _run_inference(), chunk 5/6
# ── Order-flow cascade pause: block shorts into liq spikes ──
if symbol in self.order_flow and direction == -1:
    abs_sig = self.order_flow[symbol].get_absorption_signal()
    if abs_sig.detected and abs_sig.signal_direction == 1:
        log.warning(f"[OrderFlow] SHORT blocked for {symbol}: "
                    f"long-liq absorption detected "
                    f"(score={abs_sig.absorption_score:.2f}, "
                    f"liq_z={abs_sig.liq_spike_z:.2f})")
        return
    liq_cascade = int(dff['liq_cascade'].values[-1]) if 'liq_cascade' in dff.columns else 0
    if liq_cascade and direction == -1:
        log.warning(f"[CASCADE] SHORT blocked for {symbol}: long-liq cascade active")
        return