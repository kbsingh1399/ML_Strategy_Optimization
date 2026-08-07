Python# TARGET: live_model_trainer.py
# ═══════════════════════════════════════════════════════════════════
# PATCH 3 — OnlineModelUpdater: incremental LightGBM refit every
# N candles. ADD this class at the bottom of live_model_trainer.py
# or in a new file imported by ensemble_strategy_predictor.py.
# ═══════════════════════════════════════════════════════════════════

class OnlineModelUpdater:
    """Incremental model updater using LightGBM refit().

    Every `update_every_bars` (default 96 = ~24h of 15m candles),
    the model is refit on the most recent `window_bars` candles.
    This adapts to regime shifts without full retraining.

    Usage in EnsembleStrategyPredictor._run_inference():
        if symbol not in self.online_updater:
            self.online_updater[symbol] = OnlineModelUpdater(symbol, ...)
        self.online_updater[symbol].on_new_candle(dff, labels)
    """
    def __init__(self, symbol: str, model_dir: str = None,
                 update_every_bars: int = 96,
                 window_bars: int = 500,
                 strategy_name: str = "Ensemble_6Strategy"):
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.update_every_bars = update_every_bars
        self.window_bars = window_bars
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), "models")
        self.bars_since_update: int = 0
        self._feature_buffer: deque = deque(maxlen=window_bars)
        self._label_buffer: deque = deque(maxlen=window_bars)
        self._model: Optional[lgb.Booster] = None
        self._feature_cols: List[str] = []
        self._load_or_init_model()

    def _load_or_init_model(self):
        """Load existing model from disk or train from buffer."""
        try:
            path = os.path.join(
                self.model_dir,
                f"{self.strategy_name}_{self.symbol}_online_lgb.txt")
            if os.path.exists(path):
                self._model = lgb.Booster(model_file=path)
                cols_path = path.replace("_lgb.txt", "_cols.json")
                if os.path.exists(cols_path):
                    with open(cols_path) as f:
                        self._feature_cols = json.load(f)
                log.info(f"[OnlineUpdater] Loaded existing model for "
                         f"{self.symbol}: {len(self._feature_cols)} features")
        except Exception as e:
            log.warning(f"[OnlineUpdater] Could not load model: {e}")

    def on_new_candle(self, features: dict, label: int = None):
        """Feed a new candle's features and (optional) realized label.

        Called from _run_inference() when a candle closes and we have
        a resolved trade outcome.
        """
        self._feature_buffer.append(features)
        if label is not None:
            self._label_buffer.append(label)

        self.bars_since_update += 1
        if self.bars_since_update >= self.update_every_bars:
            self._refit()
            self.bars_since_update = 0

    def _refit(self):
        """Refit the model on the in-memory buffer window."""
        if len(self._feature_buffer) < 50:
            return
        if len(self._label_buffer) < 10:
            return

        X = pd.DataFrame(list(self._feature_buffer))
        y = pd.Series(list(self._label_buffer))

        # Align to same length
        min_len = min(len(X), len(y))
        X = X.iloc[-min_len:].reset_index(drop=True)
        y = y.iloc[-min_len:].reset_index(drop=True)

        # Keep only numeric features
        self._feature_cols = [c for c in X.columns
                              if pd.api.types.is_numeric_dtype(X[c])]
        if len(self._feature_cols) < 2:
            return

        X_sub = X[self._feature_cols].astype(np.float32)
        pos_weight = max(1, int((len(y) - y.sum()) / max(y.sum(), 1)))

        try:
            if self._model is not None:
                # Incremental refit (keeps tree structure, updates leaves)
                train_data = lgb.Dataset(
                    X_sub, label=y,
                    feature_name=self._feature_cols)
                self._model = lgb.train(
                    {'objective': 'binary', 'verbose': -1,
                     'scale_pos_weight': pos_weight,
                     'num_leaves': 31, 'learning_rate': 0.02,
                     'max_depth': 4, 'min_child_samples': 10,
                     'subsample': 0.8, 'colsample_bytree': 0.8,
                     'reg_alpha': 0.1, 'n_jobs': 1},
                    train_data,
                    num_boost_round=20,
                    init_model=self._model,
                    keep_training_booster=True)
            else:
                # First train
                train_data = lgb.Dataset(
                    X_sub, label=y,
                    feature_name=self._feature_cols)
                self._model = lgb.train(
                    {'objective': 'binary', 'verbose': -1,
                     'scale_pos_weight': pos_weight,
                     'num_leaves': 31, 'learning_rate': 0.03,
                     'max_depth': 4, 'min_child_samples': 10,
                     'n_jobs': 1},
                    train_data,
                    num_boost_round=50)

            # Save to disk
            path = os.path.join(
                self.model_dir,
                f"{self.strategy_name}_{self.symbol}_online_lgb.txt")
            cols_path = path.replace("_lgb.txt", "_cols.json")
            self._model.save_model(path)
            with open(cols_path, 'w') as f:
                json.dump(self._feature_cols, f)

            log.info(f"[OnlineUpdater] {self.symbol}: refit on "
                     f"{min_len} samples, {len(self._feature_cols)} features")
        except Exception as e:
            log.warning(f"[OnlineUpdater] {self.symbol}: refit failed — {e}")

    def predict_proba(self, features: dict) -> Optional[float]:
        """Get probability for a single feature dict."""
        if self._model is None or not self._feature_cols:
            return None
        try:
            X = pd.DataFrame([features])
            for c in self._feature_cols:
                if c not in X.columns:
                    X[c] = 0.0
            X_sub = X[self._feature_cols].astype(np.float32)
            return float(self._model.predict(X_sub)[0])
        except Exception:
            return None