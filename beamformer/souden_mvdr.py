import cupy as cp

from cached_property import cached_property

from beamformer.utils import morph
from beamformer.beamform import (
    get_power_spectral_density_matrix,
    get_mvdr_vector_souden,
    blind_analytic_normalization,
    apply_beamforming_vector,
)


class _Beamformer:
    def __init__(
        self,
        Y,
        X_mask,
        N_mask,
    ):
        if cp.ndim(Y) == 4:
            self.Y = morph("1DTF->FDT", Y)
        else:
            self.Y = morph("DTF->FDT", Y)

        if cp.ndim(X_mask) == 4:
            self.X_mask = morph("1DTF->FT", X_mask, reduce=cp.median)
            self.N_mask = morph("1DTF->FT", N_mask, reduce=cp.median)
        elif cp.ndim(X_mask) == 3:
            self.X_mask = morph("DTF->FT", X_mask, reduce=cp.median)
            self.N_mask = morph("DTF->FT", N_mask, reduce=cp.median)
        elif cp.ndim(X_mask) == 2:
            self.X_mask = morph("TF->FT", X_mask, reduce=cp.median)
            self.N_mask = morph("TF->FT", N_mask, reduce=cp.median)
        else:
            raise NotImplementedError(X_mask.shape)

        assert self.Y.ndim == 3, self.Y.shape
        F, D, T = self.Y.shape
        assert D < 30, (D, self.Y.shape)
        assert self.X_mask.shape == (F, T), (self.X_mask.shape, F, T)
        assert self.N_mask.shape == (F, T), (self.N_mask.shape, F, T)

    @cached_property
    def _Cov_X(self):
        Cov_X = get_power_spectral_density_matrix(self.Y, self.X_mask)
        return Cov_X

    @cached_property
    def _Cov_N(self):
        Cov_N = get_power_spectral_density_matrix(self.Y, self.N_mask)
        return Cov_N

    @cached_property
    def _w_mvdr_souden(self):
        w_mvdr_souden = get_mvdr_vector_souden(self._Cov_X, self._Cov_N, eps=1e-10)
        return w_mvdr_souden

    @cached_property
    def _w_mvdr_souden_ban(self):
        w_mvdr_souden_ban = blind_analytic_normalization(
            self._w_mvdr_souden, self._Cov_N
        )
        return w_mvdr_souden_ban

    @cached_property
    def X_hat_mvdr_souden(self):
        return apply_beamforming_vector(self._w_mvdr_souden, self.Y).T

    @cached_property
    def X_hat_mvdr_souden_ban(self):
        return apply_beamforming_vector(self._w_mvdr_souden_ban, self.Y).T


def beamform_mvdr(Y, X_mask, N_mask, ban=False):
    bf = _Beamformer(
        Y=Y,
        X_mask=X_mask,
        N_mask=N_mask,
    )
    if ban:
        return bf.X_hat_mvdr_souden_ban
    else:
        return bf.X_hat_mvdr_souden
