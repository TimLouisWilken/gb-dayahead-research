"""Forecasting members of the online LSTM + Random-Forest hybrid.

Every member exposes
    fit(data, train_idx)          full (re)calibration on the given days
    predict(data, idx) -> (k, 24) day-ahead forecasts
and optionally
    update(data, train_idx)       cheap incremental update (LSTM fine-tuning)
"""
import os

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoLarsIC
from sklearn.preprocessing import StandardScaler

RF_JOBS = int(os.environ.get("RF_JOBS", "-1"))   # limit when running several members in parallel


# ------------------------------------------------------------- helpers
class AsinhScaler:
    """Variance-stabilising transform for prices (Uniejewski, Weron & Ziel, 2018):
    z = asinh((p - median) / MAD)."""

    def fit(self, p):
        p = np.asarray(p, dtype=np.float64).ravel()
        self.med = np.median(p)
        self.mad = np.median(np.abs(p - self.med)) / 0.6745 + 1e-6
        return self

    def transform(self, p):
        return np.arcsinh((p - self.med) / self.mad)

    def inverse(self, z):
        return np.sinh(z) * self.mad + self.med


def recency_weights(n_days, half_life):
    """Exponential down-weighting of older days (Pesaran, Pick & Pranovich, 2013)."""
    if not half_life:
        return np.ones(n_days)
    age = np.arange(n_days)[::-1]
    return 0.5 ** (age / half_life)


def naive_forecast(data, idx):
    """Lago et al. (2021) naive benchmark: d-7 for Mon/Sat/Sun, else d-1."""
    idx = np.atleast_1d(idx)
    dow = data.days[idx].dayofweek.values
    p_d1 = data.tab[idx, :, data.tab_cols.index("p_d1")]
    p_d7 = data.tab[idx, :, data.tab_cols.index("p_d7")]
    return np.where(np.isin(dow, [0, 5, 6])[:, None], p_d7, p_d1)


# ------------------------------------------------------------- Random Forest
class RFMember:
    """Random forest on per-hour tabular rows.

    The target is the price relative to yesterday's mean price, which keeps the
    trees usable when the price level drifts (trees cannot extrapolate levels).
    `window` limits the calibration window in days, `half_life` adds exponential
    recency weights inside it.
    """

    def __init__(self, window=None, half_life=None, n_estimators=300, min_samples_leaf=5,
                 max_features=0.4, seed=0, extra_cols=0):
        self.window, self.half_life = window, half_life
        self.kw = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                       max_features=max_features, n_jobs=RF_JOBS, random_state=seed)

    def _xy(self, data, idx, extra=None):
        X = data.tab[idx]                                     # (k, 24, f)
        if extra is not None:
            X = np.concatenate([X, extra[..., None]], axis=-1)
        level = data.tab[idx, :, data.tab_cols.index("p_d1_mean")]
        return X.reshape(-1, X.shape[-1]), level

    def fit(self, data, train_idx, extra=None):
        train_idx = np.asarray(train_idx)
        if self.window:
            train_idx = train_idx[-self.window:]
            extra = None if extra is None else extra[-self.window:]
        X, level = self._xy(data, train_idx, extra)
        y = (data.y[train_idx] - level).ravel()
        w = np.repeat(recency_weights(len(train_idx), self.half_life), 24)
        self.model = RandomForestRegressor(**self.kw).fit(X, y, sample_weight=w)
        return self

    def predict(self, data, idx, extra=None):
        idx = np.atleast_1d(idx)
        X, level = self._xy(data, idx, extra)
        return self.model.predict(X).reshape(len(idx), 24) + level


class GBMember(RFMember):
    """Histogram gradient boosting (the LightGBM algorithm, scikit-learn implementation) on the
    same per-hour rows and level-relative target as the random forest.

    Overfitting control: shallow trees (max 31 leaves), learning rate 0.05, L2 penalty, min 40
    rows per leaf, and the number of boosting rounds is chosen by early stopping on the most
    recent 10 % of days (time-ordered, never shuffled); the model is then refit on all days."""

    def __init__(self, window=1092, half_life=365, max_iter=1500, loss="absolute_error", quantile=None, seed=0):
        self.window, self.half_life, self.quantile = window, half_life, quantile
        self.kw = dict(loss=loss, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                       max_features=0.7, early_stopping=False, random_state=seed)
        if quantile is not None:                     # tail member: predicts a quantile of the price
            self.kw.update(loss="quantile", quantile=quantile)
        self.max_iter = max_iter

    def _val_err(self, p, y, w):
        if self.quantile is None:
            return np.average(np.abs(p - y), weights=w)
        d = y - p                                     # pinball loss at the member's own quantile
        return np.average(np.maximum(self.quantile * d, (self.quantile - 1) * d), weights=w)

    def fit(self, data, train_idx, extra=None):
        from sklearn.ensemble import HistGradientBoostingRegressor
        idx = np.asarray(train_idx)[-self.window:]
        X, level = self._xy(data, idx)
        y = (data.y[idx] - level).ravel()
        w = np.repeat(recency_weights(len(idx), self.half_life), 24)
        n_val = max(28, len(idx) // 10) * 24
        m = HistGradientBoostingRegressor(max_iter=self.max_iter, **self.kw).fit(X[:-n_val], y[:-n_val], sample_weight=w[:-n_val])
        val_err = [self._val_err(p, y[-n_val:], w[-n_val:]) for p in m.staged_predict(X[-n_val:])]
        self.n_iter = int(np.argmin(val_err)) + 1
        self.model = HistGradientBoostingRegressor(max_iter=self.n_iter, **self.kw).fit(X, y, sample_weight=w)
        return self


class MeritOrderMember:
    """Fundamental 'merit-order' regression: the price is the marginal cost of the plant that
    clears the residual load. The supply curve is modelled as a smooth (cubic spline) function of
    forecast residual load whose height scales with the CCGT fuel cost, plus hour-of-day effects:

        price ≈ Σ_k b_k(res_load) · (α_k + β_k · fuel_cost) + hour effects

    Ridge regression on the last `window` days, recalibrated daily."""

    def __init__(self, window=182, alpha=1.0):
        self.window, self.alpha = window, alpha

    def _X(self, data, idx):
        c = data.tab_cols.index
        T = data.tab[idx]
        rl = T[:, :, c("res_load")].reshape(-1, 1) / 1e4
        fc = T[:, :, c("fuel_cost")].reshape(-1, 1) / 100
        B = self.spline.transform(np.clip(rl, self.lo, self.hi))
        H = np.tile(np.eye(24), (len(idx), 1))
        return np.hstack([B, B * fc, H, H * fc])

    def fit(self, data, train_idx):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import SplineTransformer
        idx = np.asarray(train_idx)[-self.window:]
        rl = data.tab[idx, :, data.tab_cols.index("res_load")].reshape(-1, 1) / 1e4
        self.lo, self.hi = np.quantile(rl, [0.005, 0.995])
        self.spline = SplineTransformer(n_knots=8, degree=3, extrapolation="constant").fit(np.clip(rl, self.lo, self.hi))
        self.model = Ridge(alpha=self.alpha).fit(self._X(data, idx), data.y[idx].ravel())
        return self

    def predict(self, data, idx):
        idx = np.atleast_1d(idx)
        return self.model.predict(self._X(data, idx)).reshape(len(idx), 24)


class ResidualRF:
    """Error-correction stage (cf. Ghimire et al., 2024): a random forest learns the
    LSTM's out-of-sample errors from the tabular features plus the LSTM forecast
    itself. Final forecast = LSTM + predicted error."""

    def __init__(self, window=365, half_life=90, n_estimators=300, min_samples_leaf=20, max_features=0.4, seed=0):
        self.window, self.half_life = window, half_life
        self.kw = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                       max_features=max_features, n_jobs=RF_JOBS, random_state=seed)

    @staticmethod
    def _X(data, idx, base):
        X = np.concatenate([data.tab[idx], base[..., None]], axis=-1)
        return X.reshape(-1, X.shape[-1])

    def fit(self, data, idx, base):
        idx, base = np.asarray(idx)[-self.window:], base[-self.window:]
        w = np.repeat(recency_weights(len(idx), self.half_life), 24)
        self.model = RandomForestRegressor(**self.kw).fit(self._X(data, idx, base), (data.y[idx] - base).ravel(),
                                                          sample_weight=w)
        return self

    def predict(self, data, idx, base):
        idx = np.atleast_1d(idx)
        return base + self.model.predict(self._X(data, idx, base)).reshape(len(idx), 24)


# ------------------------------------------------------------- LSTM
class Seq2SeqLSTM(nn.Module):
    """Encoder LSTM reads the last 7 days of hourly history; its final state
    initialises a decoder LSTM that runs over the 24 delivery hours, consuming
    the known day-ahead inputs (renewable/weather forecasts, calendar, price lags)."""

    def __init__(self, n_hist, n_fut, hidden=64, layers=1, dropout=0.2, skip=False, out_dim=1):
        super().__init__()
        # optional linear path from the day-ahead inputs (incl. price lags) to the output,
        # so the LSTM only has to learn the non-linear remainder
        self.skip = nn.Linear(n_fut, 1) if skip else None
        self.out_dim = out_dim
        self.enc = nn.LSTM(n_hist, hidden, num_layers=layers, batch_first=True)
        self.dec = nn.LSTM(n_fut, hidden, num_layers=layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden + n_fut, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, hist, fut):
        _, state = self.enc(hist)
        out, _ = self.dec(fut, state)
        out = self.drop(out)
        y = self.head(torch.cat([out, fut], dim=-1))                     # (B, 24, out_dim)
        if self.skip is not None:
            y = y + self.skip(fut)
        return y.squeeze(-1) if self.out_dim == 1 else y


class TCNNet(nn.Module):
    """Temporal convolutional network (Bai, Kolter & Koltun, 2018): a stack of dilated causal
    convolutions encodes the 7-day history; the last time step is the context for a per-hour
    head over the 24 delivery hours."""

    def __init__(self, n_hist, n_fut, hidden=64, dilations=(1, 2, 4, 8, 16, 32, 64), kernel=3, dropout=0.2, out_dim=1):
        super().__init__()
        self.inp = nn.Conv1d(n_hist, hidden, 1)
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, kernel, dilation=d) for d in dilations)
        self.pads = [(kernel - 1) * d for d in dilations]
        self.drop = nn.Dropout(dropout)
        self.out_dim = out_dim
        self.head = nn.Sequential(nn.Linear(hidden + n_fut, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, hist, fut):
        x = self.inp(hist.transpose(1, 2))                                # (B, C, T)
        for conv, pad in zip(self.convs, self.pads):
            x = x + self.drop(nn.functional.gelu(conv(nn.functional.pad(x, (pad, 0)))))   # causal + residual
        ctx = x[:, :, -1].unsqueeze(1).expand(-1, fut.shape[1], -1)
        y = self.head(torch.cat([ctx, fut], -1))
        return y.squeeze(-1) if self.out_dim == 1 else y


class TransformerNet(nn.Module):
    """Encoder-decoder transformer (Vaswani et al., 2017): self-attention over the 168 history
    hours, the 24 delivery hours attend to that memory through cross-attention."""

    def __init__(self, n_hist, n_fut, hidden=64, heads=4, layers=2, dropout=0.1, out_dim=1):
        super().__init__()
        self.enc_in, self.dec_in = nn.Linear(n_hist, hidden), nn.Linear(n_fut, hidden)
        self.pos_e = nn.Parameter(torch.randn(1, 168, hidden) * 0.02)
        self.pos_d = nn.Parameter(torch.randn(1, 24, hidden) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout, batch_first=True), layers)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(hidden, heads, 2 * hidden, dropout, batch_first=True), layers)
        self.out_dim = out_dim
        self.head = nn.Sequential(nn.Linear(hidden + n_fut, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, hist, fut):
        mem = self.enc(self.enc_in(hist) + self.pos_e[:, :hist.shape[1]])
        out = self.dec(self.dec_in(fut) + self.pos_d, mem)
        y = self.head(torch.cat([out, fut], -1))
        return y.squeeze(-1) if self.out_dim == 1 else y


class MLPNet(nn.Module):
    """Feed-forward DNN in the spirit of Lago et al. (2021): all inputs flattened, two hidden
    layers, 24 outputs at once (no recurrence, no convolution)."""

    def __init__(self, n_hist, n_fut, hidden=256, dropout=0.2, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(nn.Linear(168 * n_hist + 24 * n_fut, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 24 * out_dim))

    def forward(self, hist, fut):
        y = self.net(torch.cat([hist.flatten(1), fut.flatten(1)], 1)).view(-1, 24, self.out_dim)
        return y.squeeze(-1) if self.out_dim == 1 else y


ARCHS = {"lstm": Seq2SeqLSTM, "tcn": TCNNet, "transformer": TransformerNet, "mlp": MLPNet}


# ---- Johnson's SU distribution (used by the distributional network, Marcjasz et al., 2023)
def jsu_params(out):
    """Map raw network outputs (..., 4) to JSU parameters: location, scale>0, skew, tail weight>0."""
    xi, lam, gam, dlt = out.unbind(-1)
    return xi, nn.functional.softplus(lam) + 1e-3, gam, nn.functional.softplus(dlt) + 0.1


def jsu_nll(out, y):
    xi, lam, gam, dlt = jsu_params(out)
    u = (y - xi) / lam
    z = gam + dlt * torch.asinh(u)
    return (-torch.log(dlt) + torch.log(lam) + 0.5 * torch.log1p(u ** 2) + 0.5 * z ** 2 + 0.9189385).mean(-1)


def jsu_quantile(out, tau):
    xi, lam, gam, dlt = jsu_params(out)
    zq = float(stats.norm.ppf(tau))
    return xi + lam * torch.sinh((zq - gam) / dlt)


class NeuralMember:
    """Neural network member with continual learning (arch: lstm | tcn | transformer | mlp).

    fit():    cold training on all available days (recency-weighted loss,
              early stopping on the most recent 10 % of days)
    update(): daily warm-start fine-tuning on the newest days mixed with an
              experience-replay sample from the whole history (Rolnick et al., 2019)
    dist="jsu": distributional output (Johnson's SU per hour, NLL loss); the point forecast
              is the median and predict_quantiles() returns any quantiles.
    """

    def __init__(self, arch="lstm", hidden=64, skip=False, revin=False, dist=None, epochs=150, lr=1e-3, batch=32,
                 half_life=365, patience=20, ft_steps=8, ft_lr=2e-4, ft_recent=28, ft_replay=16, n_seeds=1, seed=0,
                 verbose=False):
        self.__dict__.update(locals()); del self.__dict__["self"]
        self.lag_cols = ["p_d1", "p_d2", "p_d7"]

    def _make_net(self, n_hist, n_fut):
        kw = dict(out_dim=4 if self.dist == "jsu" else 1)
        if self.arch == "lstm":
            kw["skip"] = self.skip
        hidden = self.hidden * 4 if self.arch == "mlp" else self.hidden
        return ARCHS[self.arch](n_hist, n_fut, hidden, **kw)

    def _loss(self, out, y):                                             # per-sample loss, shape (B,)
        if self.dist == "jsu":
            return jsu_nll(out, y)
        return nn.functional.smooth_l1_loss(out, y, reduction="none", beta=0.5).mean(1)

    def _point(self, out):
        return jsu_quantile(out, 0.5) if self.dist == "jsu" else out

    # -- scaling (fitted on the cold-training window, frozen during fine-tuning)
    def _fit_scalers(self, data, idx):
        self.ps = AsinhScaler().fit(data.y[idx])
        self.hs = StandardScaler().fit(data.hist[idx][..., 1:].reshape(-1, data.hist.shape[-1] - 1))
        self.fs = StandardScaler().fit(data.fut[idx].reshape(-1, data.fut.shape[-1]))

    def _price_transform(self, data, idx):
        """Returns (forward, inverse) price transforms for the samples idx.

        revin=True: reversible instance normalisation (Kim et al., 2022) — every
        sample is normalised by the mean/std of its own 7-day price history, then
        asinh-stabilised, so the network learns price *shapes* rather than levels
        that drift between regimes. Otherwise a global asinh scaler is used."""
        if not self.revin:
            return self.ps.transform, self.ps.inverse
        hp = data.hist[idx][..., 0]
        lvl = hp.mean(1, keepdims=True)
        sc = np.maximum(hp.std(1, keepdims=True), 5.0)
        return (lambda p: np.arcsinh((p - lvl) / sc)), (lambda z: np.sinh(z) * sc + lvl)

    def _tensors(self, data, idx):
        tf, _ = self._price_transform(data, idx)
        h = data.hist[idx]
        h = np.concatenate([tf(h[..., 0])[..., None],
                            self.hs.transform(h[..., 1:].reshape(-1, h.shape[-1] - 1)).reshape(*h.shape[:2], -1)], -1)
        f = self.fs.transform(data.fut[idx].reshape(-1, data.fut.shape[-1])).reshape(len(idx), 24, -1)
        lags = np.stack([tf(data.tab[idx, :, data.tab_cols.index(c)]) for c in self.lag_cols], -1)
        f = np.concatenate([f, lags], -1)
        y = tf(data.y[idx])
        T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
        return T(h), T(f), T(y)

    def _train(self, net, H, F, Y, w, epochs, lr, val=None):
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        w = torch.tensor(w / w.mean(), dtype=torch.float32)
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Y), generator=g)
            for b in range(0, len(Y), self.batch):
                j = perm[b:b + self.batch]
                loss = (self._loss(net(H[j], F[j]), Y[j]) * w[j]).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if val is not None:
                net.eval()
                with torch.no_grad():
                    v = (self._point(net(val[0], val[1])) - val[2]).abs().mean().item()
                if v < best - 1e-4:
                    best, bad, best_state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        if self.verbose:
            print(f"    {self.arch} trained {ep + 1} epochs, best val MAE(z) = {best:.4f}")
        return net

    def fit(self, data, train_idx):
        train_idx = np.asarray(train_idx)
        self._fit_scalers(data, train_idx)
        H, F, Y = self._tensors(data, train_idx)
        w = recency_weights(len(train_idx), self.half_life)
        n_val = max(28, len(train_idx) // 10)
        tr, va = slice(0, len(Y) - n_val), slice(len(Y) - n_val, None)
        self.nets = []
        for s in range(self.n_seeds):
            torch.manual_seed(self.seed + s)
            net = self._make_net(H.shape[-1], F.shape[-1])
            net = self._train(net, H[tr], F[tr], Y[tr], w[tr], self.epochs, self.lr, val=(H[va], F[va], Y[va]))
            # short final pass that also sees the validation (most recent) days
            net = self._train(net, H, F, Y, w, 3, self.lr * 0.3)
            self.nets.append(net)
        return self

    def update(self, data, train_idx, rng):
        """One day of continual learning: a few gradient steps on recent days + replay."""
        train_idx = np.asarray(train_idx)
        recent = train_idx[-self.ft_recent:]
        replay = rng.choice(train_idx[:-self.ft_recent], size=self.ft_replay, replace=False)
        for net in self.nets:
            opt = torch.optim.AdamW(net.parameters(), lr=self.ft_lr, weight_decay=1e-4)
            net.train()
            for _ in range(self.ft_steps):
                j = np.concatenate([rng.choice(recent, size=min(16, len(recent)), replace=False), replay])
                H, F, Y = self._tensors(data, j)
                loss = self._loss(net(H, F), Y).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        return self

    def predict(self, data, idx):
        idx = np.atleast_1d(idx)
        H, F, _ = self._tensors(data, idx)
        out = []
        for net in self.nets:
            net.eval()
            with torch.no_grad():
                out.append(self._point(net(H, F)).numpy())
        return self._price_transform(data, idx)[1](np.mean(out, 0))

    def predict_quantiles(self, data, idx, taus):
        """(k, 24, len(taus)) quantiles; seeds are combined by averaging quantiles (Vincentization)."""
        idx = np.atleast_1d(idx)
        H, F, _ = self._tensors(data, idx)
        inv = self._price_transform(data, idx)[1]
        qs = []
        for net in self.nets:
            net.eval()
            with torch.no_grad():
                out = net(H, F)
                qs.append(np.stack([jsu_quantile(out, t).numpy() for t in taus], -1))
        q = np.mean(qs, 0)
        return np.stack([inv(q[..., i]) for i in range(len(taus))], -1)


# ------------------------------------------------------------- LEAR benchmark
class LEARMember:
    """LASSO-estimated autoregressive model (Lago et al., 2021), simplified:
    one LassoLarsIC per hour on asinh-transformed prices, using the full daily
    profiles of d-1, d-2, d-7 prices and of the day-ahead residual load plus the
    hour's own exogenous inputs. Forecast = mean over calibration windows."""

    def __init__(self, windows=(182, 728)):
        self.windows = windows

    def _X(self, data, idx, ps):
        c = data.tab_cols.index
        T = data.tab[idx]
        prof = [ps.transform(T[:, :, c(k)]) for k in ("p_d1", "p_d2", "p_d7")] + [T[:, :, c("res_load")] / 1e4,
                                                                                    T[:, :, c("wind_da")] / 1e4]
        prof = np.concatenate(prof, 1)                                  # (k, 120)
        daily = T[:, 0, [c(k) for k in ("fuel_cost", "dow_sin", "dow_cos", "is_holiday")]]
        # note: the EXTRA_FEATS sets are deliberately NOT used here - on the burn-in period both
        # made LEAR worse (neighbours 12.21, TTF/EUA/wind-error 12.35, vs 11.87 without)
        own = [c(k) for k in ("wind_da", "solar_da", "temperature", "demand_fc", "demand_d7", "demand_d2", "res_load")]
        return [np.concatenate([prof, daily, T[:, h, own]], 1) for h in range(24)]

    def fit(self, data, train_idx):
        """Windows are fitted from longest to shortest. A window shorter than the number of
        regressors cannot estimate the noise variance needed by the AIC; it borrows the
        estimate of the longest window (as in epftoolbox's LEAR for short calibration windows)."""
        self.models, sigma2 = [], {}
        for win in sorted(self.windows, reverse=True):
            idx = np.asarray(train_idx)[-win:]
            ps = AsinhScaler().fit(data.y[idx])
            Xs = self._X(data, idx, ps)
            Z = ps.transform(data.y[idx])
            per_h = []
            for h in range(24):
                keep = Xs[h].std(0) > 1e-8                 # drop columns constant in this window
                sc = StandardScaler().fit(Xs[h][:, keep])
                Xh = sc.transform(Xs[h][:, keep])
                try:
                    if Xh.shape[0] > Xh.shape[1] + 10:
                        m = LassoLarsIC(criterion="aic", max_iter=500).fit(Xh, Z[:, h])
                        sigma2.setdefault(h, m.noise_variance_)
                    else:
                        m = LassoLarsIC(criterion="aic", max_iter=500, noise_variance=sigma2[h]).fit(Xh, Z[:, h])
                except (ValueError, KeyError):              # rare LARS degeneracy -> plain Lasso
                    m = Lasso(alpha=1e-3, max_iter=5000).fit(Xh, Z[:, h])
                per_h.append((keep, sc, m))
            self.models.append((ps, per_h))
        return self

    def predict(self, data, idx):
        idx = np.atleast_1d(idx)
        preds = []
        for ps, per_h in self.models:
            Xs = self._X(data, idx, ps)
            z = np.stack([m.predict(sc.transform(Xs[h][:, k])) for h, (k, sc, m) in enumerate(per_h)], 1)
            preds.append(ps.inverse(z))
        return np.mean(preds, 0)


# ------------------------------------------------------------- online combiners
class BOACombiner:
    """Bernstein Online Aggregation (Wintenberger, 2017) with expert-specific
    adaptive learning rates and optional exponential forgetting (forget<1),
    run separately for each delivery hour. Default forget=1.0 (no forgetting):
    selected on 2024, confirmed on the 2025-26 holdout (DM p=0.0012 for the
    6-member set). Loss: absolute error, linearised via its gradient."""

    def __init__(self, n_experts, n_hours=24, forget=1.0, smooth=1):
        self.K, self.forget, self.smooth = n_experts, forget, smooth
        self.R = np.zeros((n_hours, n_experts))        # cumulative linearised regret
        self.V = np.zeros((n_hours, n_experts))        # cumulative squared regret
        self.E = np.full(n_hours, 1e-9)                # range of regrets seen
        self.w = np.full((n_hours, n_experts), 1.0 / n_experts)

    def _w_served(self):
        """Per-hour weights are noisy; averaging each hour's weights with its +/-smooth
        neighbours reduces that variance (2024 selection -0.03 MAE, holdout confirmed,
        both p<0.001). Serving only - the online update keeps the raw per-hour weights."""
        if not self.smooth:
            return self.w
        H = len(self.w)
        w = np.stack([self.w[max(0, h - self.smooth):min(H, h + self.smooth + 1)].mean(0)
                      for h in range(H)])
        return w / w.sum(1, keepdims=True)

    def combine(self, F):                               # F: (24, K)
        return (self._w_served() * F).sum(1)

    def update(self, F, y):
        yhat = (self.w * F).sum(1)                     # raw weights: learning dynamics unchanged
        g = np.sign(yhat - y)[:, None]                  # gradient of |yhat - y|
        r = g * (yhat[:, None] - F)                     # regret of each expert (positive = expert better)
        self.E = np.maximum(self.E, np.abs(r).max(1))
        self.R = self.forget * self.R + r
        self.V = self.forget * self.V + r ** 2
        eta = np.minimum(1 / (2 * self.E[:, None]), np.sqrt(np.log(self.K) / (self.V + 1e-9)))
        logit = np.log(eta) + eta * self.R - eta ** 2 * self.V
        logit -= logit.max(1, keepdims=True)
        w = np.exp(logit)
        self.w = w / w.sum(1, keepdims=True)


class InverseErrorCombiner:
    """Weights proportional to 1 / MAE^p of each expert over the last `window` days, per hour."""

    def __init__(self, n_experts, window=28, power=2):
        self.window, self.power, self.hist = window, power, []
        self.K = n_experts

    def combine(self, F):
        if not self.hist:
            return F.mean(1)
        mae = np.mean(self.hist[-self.window:], 0)      # (24, K)
        w = 1 / (mae ** self.power + 1e-9)
        return (w / w.sum(1, keepdims=True) * F).sum(1)

    def update(self, F, y):
        self.hist.append(np.abs(F - y[:, None]))


LSTMMember = NeuralMember          # backwards-compatible name
