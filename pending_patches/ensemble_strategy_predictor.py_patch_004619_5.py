Python# TARGET: ensemble_strategy_predictor.py
# FIND in _run_inference() the block after computing direction/confidence:
#     if not self.ensemble.should_enter(direction, confidence, agreeing): return
#     if trade_tracker is None: return
# ADD between them:

            # ── Order-flow cascade pause: block shorts into liq spikes ──
            if symbol in self.order_flow and direction == -1:
                abs_sig = self.order_flow[symbol].get_absorption_signal()
                if abs_sig.detected and abs_sig.signal_direction == 1:
                    log.warning(
                        f"[OrderFlow] SHORT blocked for {symbol}: "
                        f"long-liq absorption detected "
                        f"(score={abs_sig.absorption_score:.2f}, "
                        f"liq_z={abs_sig.liq_spike_z:.2f})"
                    )
                    return
                # Also check Coinglass liq_cascade feature
                liq_cascade = int(dff['liq_cascade'].values[-1]) if 'liq_cascade' in dff.columns else 0
                if liq_cascade and direction == -1:
                    log.warning(
                        f"[OrderFlow] SHORT blocked for {symbol}: "
                        f"liq_cascade flag active (long liqs > 2.5σ)"
                    )
                    return