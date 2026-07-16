import numpy as np

# Standard Strategy Configuration Map
# Key layout: STRATEGY_NAME -> (default | ASSETUSDT) -> PARAM_NAME
STRATEGY_CONFIG = {
    "VWAP": {
        "default": {
            "cooldown_seconds": 1800
        }
    },
    "POC_R": {
        "default": {
            "cooldown_seconds": 1800
        }
    },
    "POC_M": {
        "default": {
            "cooldown_seconds": 1800
        }
    },
    "CVD_C": {
        "default": {
            "cooldown_seconds": 3600
        }
    },
    "LIQ_R": {
        "default": {
            "cooldown_seconds": 1800
        }
    },
    "CVD_D": {
        "default": {
            "cooldown_seconds": 1800
        }
    },
    "D_ACC": {
        "default": {
            "cooldown_seconds": 1800
        }
    },
    "D_MOM": {
        "default": {
            "cooldown_seconds": 1800
        }
    },
    "AlphaSqueezer_V17": {
        "default": {
            "cooldown_seconds": 900,
            "confidence_threshold": 0.51,
            "tp_mult": 3.5,
            "sl_mult": 1.0,
            "trail_activation_r": 2.5,
            "trail_buffer_atr": 0.5,
            "max_hold_bars": 96
        },
        "ADAUSDT": {
            "confidence_threshold": 0.5397,
            "tp_mult": 1.74,
            "sl_mult": 0.57,
            "trail_activation_r": 2.11
        },
        "AVAXUSDT": {
            "confidence_threshold": 0.5252,
            "tp_mult": 4.24,
            "sl_mult": 1.15,
            "trail_activation_r": 1.59
        },
        "BNBUSDT": {
            "confidence_threshold": 0.52,
            "tp_mult": 3.04,
            "sl_mult": 1.06,
            "trail_activation_r": 1.62
        },
        "BTCUSDT": {
            "confidence_threshold": 0.5334,
            "tp_mult": 3.53,
            "sl_mult": 1.3,
            "trail_activation_r": 1.66
        },
        "DOGEUSDT": {
            "confidence_threshold": 0.5852,
            "tp_mult": 5.01,
            "sl_mult": 1.32,
            "trail_activation_r": 2.7
        },
        "DOTUSDT": {
            "confidence_threshold": 0.5108,
            "tp_mult": 4.15,
            "sl_mult": 1.36,
            "trail_activation_r": 2.14
        },
        "ETHUSDT": {
            "confidence_threshold": 0.5538,
            "tp_mult": 4.1,
            "sl_mult": 1.47,
            "trail_activation_r": 2.53
        },
        "LINKUSDT": {
            "confidence_threshold": 0.6482,
            "tp_mult": 7.18,
            "sl_mult": 1.48,
            "trail_activation_r": 3.27
        },
        "LTCUSDT": {
            "confidence_threshold": 0.6377,
            "tp_mult": 5.55,
            "sl_mult": 1.39,
            "trail_activation_r": 1.69
        },
        "NEARUSDT": {
            "confidence_threshold": 0.5918,
            "tp_mult": 3.16,
            "sl_mult": 1.17,
            "trail_activation_r": 2.81
        },
        "SOLUSDT": {
            "confidence_threshold": 0.5393,
            "tp_mult": 3.41,
            "sl_mult": 1.16,
            "trail_activation_r": 2.53
        },
        "SUIUSDT": {
            "confidence_threshold": 0.5282,
            "tp_mult": 6.27,
            "sl_mult": 1.31,
            "trail_activation_r": 2.91
        },
        "TRXUSDT": {
            "confidence_threshold": 0.5514,
            "tp_mult": 5.4,
            "sl_mult": 1.34,
            "trail_activation_r": 1.59
        },
        "XAGUSDT": {
            "confidence_threshold": 0.5678,
            "tp_mult": 5.12,
            "sl_mult": 1.18,
            "trail_activation_r": 2.04
        },
        "XAUUSDT": {
            "confidence_threshold": 0.5885,
            "tp_mult": 2.01,
            "sl_mult": 0.64,
            "trail_activation_r": 1.54
        },
        "XRPUSDT": {
            "confidence_threshold": 0.5164,
            "tp_mult": 1.45,
            "sl_mult": 0.55,
            "trail_activation_r": 1.61
        }
    }
}

def get_param(symbol: str, strategy_name: str, param_name: str, fallback=None):
    """
    Looks up a parameter for a given symbol and strategy name.
    1. Checks the asset-specific override under STRATEGY_CONFIG[strategy_name][symbol][param_name]
    2. Checks the strategy default under STRATEGY_CONFIG[strategy_name]['default'][param_name]
    3. Returns fallback value if not found.
    """
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
