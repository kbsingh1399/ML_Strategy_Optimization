Python# TARGET: Engine_1.py — ALREADY PRESENT at 92f33909
if self.consecutive_losses > 0:
    anti_mart_scale = max(
        self.anti_martingale_floor,
        self.anti_martingale_factor ** self.consecutive_losses)
else:
    anti_mart_scale = 1.0
effective_risk_mult = risk_mult * anti_mart_scale
risk_capital = max(0.0, self.current_capital) * zeno_risk_pct * effective_risk_mult