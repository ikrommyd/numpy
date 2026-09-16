``np.argmax`` and ``np.argmin`` accept any integer dtype as ``out``
-------------------------------------------------------------------
`numpy.argmax` and `numpy.argmin` now cast ``out`` with same-kind casting, like `numpy.take`.
Integer arrays such as ``uint64``, or ``int64`` on 32-bit platforms, no longer raise a ``TypeError``.
