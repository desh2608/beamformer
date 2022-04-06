import functools
import operator
import cupy as cp
import numpy as np


__all__ = [
    "get_power_spectral_density_matrix",
    "get_mvdr_vector_souden",
    "blind_analytic_normalization",
    "apply_beamforming_vector",
]


def get_power_spectral_density_matrix(
    observation,
    mask=None,
    sensor_dim=-2,
    source_dim=-2,
    time_dim=-1,
    normalize=True,
):
    """
    Calculates the weighted power spectral density matrix.
    It's also called covariance matrix.
    With the dim parameters you can change the sort of the dims of the
    observation and mask, but not every combination is allowed.

    :param observation: Complex observations with shape (..., sensors, frames)
    :param mask: Masks with shape (bins, frames) or (..., sources, frames)
    :param sensor_dim: change sensor dimension index (Default: -2)
    :param source_dim: change source dimension index (Default: -2),
        source_dim = 0 means mask shape (sources, ..., frames)
    :param time_dim:  change time dimension index (Default: -1),
        this index must match for mask and observation
    :param normalize: Boolean to decide if normalize the mask
    :return: PSD matrix with shape (..., sensors, sensors)
        or (..., sources, sensors, sensors) or
        (sources, ..., sensors, sensors)
        if source_dim % observation.ndim < -2 respectively
        mask shape (sources, ..., frames)

    Examples
    --------
    >>> F, T, D, K = 51, 31, 6, 2
    >>> X = np.random.randn(F, D, T) + 1j * np.random.randn(F, D, T)
    >>> mask = np.random.randn(F, K, T)
    >>> mask = mask / np.sum(mask, axis=0, keepdims=True)
    >>> get_power_spectral_density_matrix(X, mask=mask).shape
    (51, 2, 6, 6)
    >>> mask = np.random.randn(F, T)
    >>> mask = mask / np.sum(mask, axis=0, keepdims=True)
    >>> get_power_spectral_density_matrix(X, mask=mask).shape
    (51, 6, 6)
    """
    # ensure negative dim indexes
    sensor_dim, source_dim, time_dim = (
        d % observation.ndim - observation.ndim
        for d in (sensor_dim, source_dim, time_dim)
    )

    # ensure observation shape (..., sensors, frames)
    obs_transpose = [
        i for i in range(-observation.ndim, 0) if i not in [sensor_dim, time_dim]
    ] + [sensor_dim, time_dim]
    observation = observation.transpose(obs_transpose)

    if mask is None:
        psd = cp.einsum("...dt,...et->...de", observation, observation.conj())

        # normalize
        psd /= observation.shape[-1]

    else:
        # Unfortunately, this function changes `mask`.
        mask = cp.copy(mask)

        # normalize
        if mask.dtype == cp.bool:
            mask = cp.asfarray(mask)

        if normalize:
            mask /= cp.maximum(
                cp.sum(mask, axis=time_dim, keepdims=True),
                1e-10,
            )

        if mask.ndim + 1 == observation.ndim:
            mask = cp.expand_dims(mask, -2)
            psd = cp.einsum(
                "...dt,...et->...de",
                mask * observation,
                observation.conj(),
            )
        else:
            # ensure shape (..., sources, frames)
            mask_transpose = [
                i
                for i in range(-observation.ndim, 0)
                if i not in [source_dim, time_dim]
            ] + [source_dim, time_dim]
            mask = mask.transpose(mask_transpose)

            psd = cp.einsum(
                "...kt,...dt,...et->...kde", mask, observation, observation.conj()
            )

            if source_dim < -2:
                # Assume PSD shape (sources, ..., sensors, sensors) is desired
                psd = cp.rollaxis(psd, -3, source_dim % observation.ndim)

    return psd


def blind_analytic_normalization(vector, noise_psd_matrix):
    """Reduces distortions by normalizing the beamforming vectors.

    See Section III.A in the following paper:

    Warsitz, Ernst, and Reinhold Haeb-Umbach. "Blind acoustic beamforming
    based on generalized eigenvalue decomposition." IEEE Transactions on
    audio, speech, and language processing 15.5 (2007): 1529-1539.

    Args:
        vector: Beamforming vector with shape (..., sensors)
        noise_psd_matrix: With shape (..., sensors, sensors)

    """
    nominator = cp.einsum(
        "...a,...ab,...bc,...c->...",
        vector.conj(),
        noise_psd_matrix,
        noise_psd_matrix,
        vector,
    )
    nominator = cp.sqrt(nominator)

    denominator = cp.einsum(
        "...a,...ab,...b->...", vector.conj(), noise_psd_matrix, vector
    )
    denominator = cp.sqrt(denominator * denominator.conj())

    nominator = cp.asnumpy(nominator)
    denominator = cp.asnumpy(denominator)
    normalization = np.divide(  # https://stackoverflow.com/a/37977222/5766934
        nominator, denominator, out=np.zeros_like(nominator), where=denominator != 0
    )
    normalization = cp.asarray(normalization)

    return vector * cp.abs(normalization[..., cp.newaxis])


def apply_beamforming_vector(vector, mix):
    """Applies a beamforming vector such that the sensor dimension disappears.

    Although this function may seem simple, it turned out that using it
    reduced implementation errors in practice quite a bit.

    :param vector: Beamforming vector with dimensions ..., sensors
    :param mix: Observed signal with dimensions ..., sensors, time-frames
    :return: A beamformed signal with dimensions ..., time-frames
    """
    assert vector.shape[-1] < 30, (vector.shape, mix.shape)
    return cp.einsum("...a,...at->...t", vector.conj(), mix)


def get_optimal_reference_channel(
    w_mat,
    target_psd_matrix,
    noise_psd_matrix,
    eps=None,
):
    if w_mat.ndim != 3:
        raise ValueError(
            "Estimating the ref_channel expects currently that the input "
            "has 3 ndims (frequency x sensors x sensors). "
            "Considering an independent dim in the SNR estimate is not "
            "unique."
        )
    if eps is None:
        eps = cp.finfo(w_mat.dtype).tiny
    SNR = cp.einsum(
        "...FdR,...FdD,...FDR->...R", w_mat.conj(), target_psd_matrix, w_mat
    ) / cp.maximum(
        cp.einsum("...FdR,...FdD,...FDR->...R", w_mat.conj(), noise_psd_matrix, w_mat),
        eps,
    )
    # Raises an exception when np.inf and/or np.NaN was in target_psd_matrix
    # or noise_psd_matrix
    assert cp.all(cp.isfinite(SNR)), SNR
    return cp.argmax(SNR.real)


def stable_solve(A, B):
    """
    Use np.linalg.solve with fallback to np.linalg.lstsq.
    Equal to np.linalg.lstsq but faster.
    Note: limited currently by A.shape == B.shape
    This function tries np.linalg.solve with independent dimensions,
    when this is not working the function fall back to np.linalg.solve
    for each matrix. If one matrix does not work it falls back to
    np.linalg.lstsq.
    The reason for not using np.linalg.lstsq directly is the execution time.
    Examples:
    A and B have the shape (500, 6, 6), than a loop over lstsq takes
    108 ms and this function 28 ms for the case that one matrix is singular
    else 1 ms.
    >>> def normal(shape):
    ...     return np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    >>> A = normal((6, 6))
    >>> B = normal((6, 6))
    >>> C1 = np.linalg.solve(A, B)
    >>> C2, *_ = np.linalg.lstsq(A, B)
    >>> C3 = stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C1, C2)
    >>> np.testing.assert_allclose(C1, C3)
    >>> np.testing.assert_allclose(C1, C4)
    >>> A = np.zeros((6, 6), dtype=np.complex128)
    >>> B = np.zeros((6, 6), dtype=np.complex128)
    >>> C1 = np.linalg.solve(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.LinAlgError: Singular matrix
    >>> C2, *_ = np.linalg.lstsq(A, B)
    >>> C3 = stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C2, C3)
    >>> np.testing.assert_allclose(C2, C4)
    >>> A = normal((3, 6, 6))
    >>> B = normal((3, 6, 6))
    >>> C1 = np.linalg.solve(A, B)
    >>> C2, *_ = np.linalg.lstsq(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.LinAlgError: 3-dimensional array given. Array must be two-dimensional
    >>> C3 = stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C1, C3)
    >>> np.testing.assert_allclose(C1, C4)
    >>> A[2, 3, :] = 0
    >>> C1 = np.linalg.solve(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.LinAlgError: Singular matrix
    >>> C2, *_ = np.linalg.lstsq(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.LinAlgError: 3-dimensional array given. Array must be two-dimensional
    >>> C3 = stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C3, C4)
    """
    assert A.shape[:-2] == B.shape[:-2], (A.shape, B.shape)
    assert A.shape[-1] == B.shape[-2], (A.shape, B.shape)
    try:
        return cp.linalg.solve(A, B)
    except cp.linalg.LinAlgError:
        shape_A, shape_B = A.shape, B.shape
        assert shape_A[:-2] == shape_A[:-2]
        working_shape_A = [
            functools.reduce(operator.mul, [1, *shape_A[:-2]]),
            *shape_A[-2:],
        ]
        working_shape_B = [
            functools.reduce(operator.mul, [1, *shape_B[:-2]]),
            *shape_B[-2:],
        ]
        A = A.reshape(working_shape_A)
        B = B.reshape(working_shape_B)

        C = cp.zeros_like(B)
        for i in range(working_shape_A[0]):
            # lstsq is much slower, use it only when necessary
            try:
                C[i] = cp.linalg.solve(A[i], B[i])
            except cp.linalg.LinAlgError:
                C[i], *_ = cp.linalg.lstsq(A[i], B[i])
        return C.reshape(*shape_B)


def get_mvdr_vector_souden(
    target_psd_matrix,
    noise_psd_matrix,
    ref_channel=None,
    eps=None,
):
    """
    Returns the MVDR beamforming vector described in [Souden2010MVDR].
    The implementation is based on the description of [Erdogan2016MVDR].

    The ref_channel is selected based of an SNR estimate.

    The eps ensures that the SNR estimation for the ref_channel works
    as long target_psd_matrix and noise_psd_matrix do not contain inf or nan.
    Also zero matrices work. The default eps is the smallest non zero value.

    Note: the frequency dimension is necessary for the ref_channel estimation.
    Note: Currently this function does not support independent dimensions with
          an estimated ref_channel. There is an open point to discuss:
          Should the independent dimension be considered in the SNR estimate
          or not?

    :param target_psd_matrix: Target PSD matrix
        with shape (..., bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (..., bins, sensors, sensors)
    :param ref_channel:
    :param return_ref_channel:
    :param eps: If None use the smallest number bigger than zero.
    :return: Set of beamforming vectors with shape (bins, sensors)

    Returns:

    @article{Souden2010MVDR,
      title={On optimal frequency-domain multichannel linear filtering for noise reduction},
      author={Souden, Mehrez and Benesty, Jacob and Affes, Sofi{\`e}ne},
      journal={IEEE Transactions on audio, speech, and language processing},
      volume={18},
      number={2},
      pages={260--276},
      year={2010},
      publisher={IEEE}
    }
    @inproceedings{Erdogan2016MVDR,
      title={Improved MVDR Beamforming Using Single-Channel Mask Prediction Networks.},
      author={Erdogan, Hakan and Hershey, John R and Watanabe, Shinji and Mandel, Michael I and Le Roux, Jonathan},
      booktitle={Interspeech},
      pages={1981--1985},
      year={2016}
    }

    """
    assert noise_psd_matrix is not None

    phi = stable_solve(noise_psd_matrix, target_psd_matrix)
    lambda_ = cp.trace(phi, axis1=-1, axis2=-2)[..., None, None]
    if eps is None:
        eps = cp.finfo(lambda_.dtype).tiny
    mat = phi / cp.maximum(lambda_.real, eps)

    if ref_channel is None:
        ref_channel = get_optimal_reference_channel(
            mat, target_psd_matrix, noise_psd_matrix, eps=eps
        )

    beamformer = mat[..., ref_channel]
    return beamformer
