import numpy as np

# Standard Strategy Configuration Map
# Key layout: STRATEGY_NAME -> (default | ASSETUSDT) -> PARAM_NAME
STRATEGY_CONFIG = {
    "ML_Trend_Pull": {
        "default": {
            "cooldown_seconds": 900,
            "confidence_threshold": 0.5261,
            "tp_mult": 5.0,  # Minimum 5R target as requested
            "sl_mult": 1.0,
            "trail_activation_r": 3.0,
            "trail_buffer_atr": 0.5,
            "max_hold_bars": 96
        }
    }
}

def get_param(symbol: str, strategy_name: str, param_name: str, fallback=None):
    strat_dict = STRATEGY_CONFIG.get(strategy_name)
    if not strat_dict:
        return fallback

    # Check asset override
    if symbol in strat_dict and param_name in strat_dict[symbol]:
        return strat_dict[symbol][param_name]

    # Check strategy default
    if "default" in strat_dict and param_name in strat_dict["default"]:
        return strat_dict["default"][param_name]

    return fallback
