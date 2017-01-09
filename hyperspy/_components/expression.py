from functools import wraps
from hyperspy.component import Component

_CLASS_DOC = \
    """%s component (created with Expression).

.. math::

    f(x) = %s

"""


def _fill_function_args(fn):
    @wraps(fn)
    def fn_wrapped(self, x):
        return fn(x, *[p.value for p in self.parameters])

    return fn_wrapped


def _fill_function_args_2d(fn):
    @wraps(fn)
    def fn_wrapped(self, x, y):
        return fn(x, y, *[p.value for p in self.parameters])

    return fn_wrapped


class Expression(Component):

    """Create a component from a string expression.
    """

    def __init__(self, expression, name, position=None, module="numpy",
                 autodoc=True, **kwargs):
        """Create a component from a string expression.

        It automatically generates the partial derivatives and the
        class docstring.

        Parameters
        ----------
        expression: str
            Component function in SymPy text expression format. See the SymPy
            documentation for details. The only additional constraint is that
            the variable(s) must be `x` or `x` and `y` for a 2D component.
            Also, in `module` is "numexpr" the
            functions are limited to those that numexpr support. See its
            documentation for details.
        name : str
            Name of the component.
        position: str, optional
            The parameter name that defines the position of the component if
            applicable. It enables adjusting the position of the component
            interactively in a model.
        module: {"numpy", "numexpr"}, default "numpy"
            Module used to evaluate the function. numexpr is often faster but
            it supports less functions.
        add_rotation: bool, default False
            This is only relevant for 2D components. If `True` it automatically
            adds `rotation_angle` parameter.

        **kwargs
             Keyword arguments can be used to initialise the value of the
             parameters.

        Methods
        -------
        recompile: useful to recompile the function and gradient with a
            a different module.

        Examples
        --------

        The following creates a Gaussian component and set the initial value
        of the parameters:

        >>> hs.model.components1D.Expression(
        ... expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2)",
        ... name="Gaussian",
        ... height=1,
        ... fwhm=1,
        ... x0=0,
        ... position="x0",)

        """

        import sympy
        self._add_rotation = kwargs.pop("add_rotation", False)
        self._str_expression = expression
        self.compile_function(module=module, position=position)
        # Initialise component
        Component.__init__(self, self._parameter_strings)
        self._whitelist['expression'] = ('init', expression)
        self._whitelist['name'] = ('init', name)
        self._whitelist['position'] = ('init', position)
        self._whitelist['module'] = ('init', module)
        if self._is2D:
            self._whitelist['add_rotation'] = ('init', self._add_rotation)
        self.name = name
        # Set the position parameter
        if position:
            if self._is2D:
                self._position_x = getattr(self, position[0])
                self._position_y = getattr(self, position[1])
            else:
                self._position = getattr(self, position)
        # Set the initial value of the parameters
        if kwargs:
            for kwarg, value in kwargs.items():
                setattr(getattr(self, kwarg), 'value', value)

        if autodoc:
            self.__doc__ = _CLASS_DOC % (
                name, sympy.latex(sympy.sympify(expression)))

    def compile_function(self, module="numpy", position=False):
        import sympy
        from sympy.utilities.lambdify import lambdify
        expr = sympy.sympify(self._str_expression)
        # Extract x
        x, = [symbol for symbol in expr.free_symbols if symbol.name == "x"]
        # Extract y
        y = [symbol for symbol in expr.free_symbols if symbol.name == "y"]
        self._is2D = True if y else False
        if self._is2D:
            y = y[0]
        if self._is2D and self._add_rotation:
            if position:  # Rotate around the "center" of the function
                rotx = sympy.sympify(
                    "{0} + (x - {0}) * cos(rotation_angle) - (y - {1}) *"
                    " sin(rotation_angle)"
                    .format(*position))
                roty = sympy.sympify(
                    "{1} + (x - {0}) * sin(rotation_angle) + (y - {1}) *"
                    "cos(rotation_angle)"
                    .format(*position))
            else:  # Rotate around the origin
                rotx = sympy.sympify(
                    "x * cos(rotation_angle) - y * sin(rotation_angle)")
                roty = sympy.sympify(
                    "x * sin(rotation_angle) + y * cos(rotation_angle)")
            expr = expr.subs({"x": rotx, "y": roty}, simultaneous=False)
        rvars = sympy.symbols([s.name for s in expr.free_symbols], real=True)
        real_expr = expr.subs(
            {orig: real_ for (orig, real_) in zip(expr.free_symbols, rvars)})
        # just replace with the assumption that all our variables are real
        expr = real_expr

        eval_expr = expr.evalf()
        # Extract parameters
        variables = ("x", "y") if self._is2D else ("x", )
        parameters = [
            symbol for symbol in expr.free_symbols
            if symbol.name not in variables]
        parameters.sort(key=lambda x: x.name)  # to have a reliable order
        # Create compiled function
        variables = [x, y] if self._is2D else [x]
        self._f = lambdify(variables + parameters, eval_expr,
                           modules=module, dummify=False)

        if self._is2D:
            f = lambda x, y: self._f(x, y, *[p.value for p in self.parameters])
        else:
            f = lambda x: self._f(x, *[p.value for p in self.parameters])
        setattr(self, "function", f)
        parnames = [symbol.name for symbol in parameters]
        self._parameter_strings = parnames
        ffargs = _fill_function_args_2d if self._is2D else _fill_function_args
        for parameter in parameters:
            grad_expr = sympy.diff(eval_expr, parameter)
            setattr(self,
                    "_f_grad_%s" % parameter.name,
                    lambdify(variables + parameters,
                             grad_expr.evalf(),
                             modules=module,
                             dummify=False)
                    )

            setattr(self,
                    "grad_%s" % parameter.name,
                    ffargs(
                        getattr(
                            self,
                            "_f_grad_%s" %
                            parameter.name)).__get__(
                        self,
                        Expression)
                    )
