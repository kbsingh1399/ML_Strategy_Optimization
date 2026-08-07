Python# TARGET: Engine_1.py — ALREADY PRESENT at 92f33909
# Rolling 1-hour DD circuit breaker
if self.rolling_dd_halt:
    if time.time() < self.rolling_dd_halt_until:
        remaining_m = (self.rolling_dd_halt_until - time.time()) / 60.0
        log.warning(f"[Risk] Entry blocked: 1h rolling DD circuit "
                    f"breaker active ({remaining_m:.0f}m remaining)")
        return
    else:
        self.rolling_dd_halt = False
        log.info("[Risk] 1h rolling DD halt timer expired — resuming entries")