``out`` of any integer dtype for ``np.argmax`` and ``np.argmin``
----------------------------------------------------------------
`numpy.argmax` and `numpy.argmin` now convert the ``out`` argument using
same-kind casting, matching the behaviour of indexing and `numpy.take`.
Passing an array of a different integer dtype than ``intp`` -- an ``uint64``
array on any platform, or an ``int64`` array on a 32-bit platform --
previously raised a ``TypeError`` about unsafe casting, even though narrower
integer dtypes such as ``int16`` were already accepted.
