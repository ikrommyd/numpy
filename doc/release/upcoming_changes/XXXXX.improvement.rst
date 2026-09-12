``np.put`` accepts any integer dtype as indices
-----------------------------------------------
`numpy.put` now converts its ``ind`` argument using same-kind casting,
matching the behaviour of indexing and `numpy.take`.  Passing an array of a
different integer dtype than ``intp`` -- an ``uint64`` array on any
platform, or an ``int64`` array on a 32-bit platform -- previously raised a
``TypeError`` about unsafe casting, even though `numpy.put` is documented as
being equivalent to ``a.flat[ind] = v``, which accepts those arrays.
