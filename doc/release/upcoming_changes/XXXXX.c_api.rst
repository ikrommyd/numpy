Reduction loops are now used by ``accumulate`` and ``reduceat``
---------------------------------------------------------------
Ufuncs with more than one output that register a dedicated reduction loop on
their ``ArrayMethod`` now support :meth:`~numpy.ufunc.accumulate` and
:meth:`~numpy.ufunc.reduceat` as well as :meth:`~numpy.ufunc.reduce`. Both
return one array per output. See :ref:`c-api.reduction-loop-tutorial` for a
worked example.
