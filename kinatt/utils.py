def listify(something):
    if isinstance(something, list):
        return something
    return [something]

def scipinize(fun):
    """Scipinize a function returning both f and jac

    For the detail this issue may help:
    https://github.com/scipy/scipy/issues/12692

    Parameters
    ----------
    fun: function
        function maps numpy.ndarray(n_dim,) to tuple[numpy.ndarray(m_dim,),
        numpy.ndarray(m_dim, n_dim)], where the returned tuples is
        composed of function value(vector) and the corresponding jacobian.
    Returns
    -------
    fun_scipinized : function
        function maps numpy.ndarray(n_dim,) to a value numpy.ndarray(m_dim,).
    fun_scipinized_jac : function
        function maps numpy.ndarray(n_dim,) to
        jacobian numpy.ndarray(m_dim, n_dim).
    """

    closure_member = {'jac_cache': None}

    def fun_scipinized(x):
        f, jac = fun(x)
        closure_member['jac_cache'] = jac
        return f

    def fun_scipinized_jac(x):
        return closure_member['jac_cache']
    return fun_scipinized, fun_scipinized_jac
