``np.searchsorted`` accepts any integer dtype as ``sorter``
------------------------------------------------------------
`numpy.searchsorted` now converts its ``sorter`` argument using same-kind
casting, matching the behaviour of indexing and `numpy.take`.  Passing an
array of a different integer dtype than ``intp`` -- an ``uint64`` array on
any platform, or an ``int64`` array on a 32-bit platform -- previously
failed, and the resulting error was reported as ``ValueError: could not
parse sorter argument``.
