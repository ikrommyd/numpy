``np.choose`` accepts any integer dtype as indices
---------------------------------------------------
`numpy.choose` now converts its ``a`` argument using same-kind casting,
matching the behaviour of indexing and `numpy.take`.  Passing an array of a
different integer dtype than ``intp`` -- an ``uint64`` array on any
platform, or an ``int64`` array on a 32-bit platform -- previously raised a
``TypeError`` about unsafe casting, while an equivalent list of Python
integers was accepted.
