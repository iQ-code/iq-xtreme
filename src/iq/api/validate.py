from typing import Any

import numpy as np
import numpy.typing as npt


def integer(
    number: int | float,
    min_value: int,
    max_value: int,
    name: str | None = None,
    b_return_repr: bool = True,
) -> str | int:
    """Validate that a number is an integer within a given range.

    Parameters
    ----------
    number : int or float
        Value to validate. Will be cast to int.
    min_value : int
        Minimum allowed value (inclusive).
    max_value : int
        Maximum allowed value (inclusive).
    name : str, optional
        Parameter name used in error messages.
    b_return_repr : bool, optional
        If True, return repr(number) as a string; otherwise return the int.

    Returns
    -------
    str or int
        repr of the validated integer if b_return_repr is True, else the int.

    Raises
    ------
    ValueError
        If the number is outside [min_value, max_value].
    """
    number = int(number)
    if number < min_value or number > max_value:
        if name is not None:
            raise ValueError(
                f"The parameter {name} is {number}, which is out of range: ({min_value}, {max_value})"
            )
        else:
            raise ValueError(f"Number: {number} is out of range: ({min_value}, {max_value})")
    if b_return_repr:
        return repr(number)
    else:
        return number


def integer_or_string(obj: int | float | str) -> str:
    """Return the object as-is if it is a string, or convert it to an integer repr.

    Parameters
    ----------
    obj : int, float, or str
        Value to validate and convert.

    Returns
    -------
    str
        The original string, or repr of the integer conversion.
    """
    if isinstance(obj, str):
        return obj
    obj = int(obj)
    return repr(obj)


def string(obj: Any, name: str | None = None) -> str:
    """Validate that an object is a string.

    Parameters
    ----------
    obj : any
        Object to validate.
    name : str, optional
        Parameter name used in error messages.

    Returns
    -------
    str
        The validated string.

    Raises
    ------
    ValueError
        If obj is not a string.
    """
    if isinstance(obj, str):
        return obj
    else:
        if name is not None:
            raise ValueError(f"The parameter '{name}' must be a string. Got:\n{obj}")
        else:
            raise ValueError(f"Expected a string. Got:\n{obj}")


def rng(obj: np.random.Generator | int | str | list) -> str:
    """Validate and convert a random seed or generator to a repr string.

    Parameters
    ----------
    obj : np.random.Generator, int, str, or list
        A NumPy random generator (which will be sampled to produce a seed),
        or a value directly usable as a seed.

    Returns
    -------
    str
        repr of the seed value.

    Raises
    ------
    ValueError
        If obj is not a Generator or a valid seed type.
    """
    if isinstance(obj, np.random.Generator):
        obj = obj.integers(0, 0xFFFFFFFF, dtype=np.uint32)
    elif not isinstance(obj, (int, str, list)):
        raise ValueError(f"Expected a random number generator or a valid seed. Got:\n{obj}")
    return repr(obj)


def real(
    number: int | float,
    min_value: float | None = None,
    max_value: float | None = None,
    name: str | None = None,
    b_return_repr: bool = True,
) -> str | float:
    """Validate that a number is a real value within optional bounds.

    Parameters
    ----------
    number : int or float
        Value to validate. Will be cast to float.
    min_value : float, optional
        Minimum allowed value (inclusive). Not checked if None.
    max_value : float, optional
        Maximum allowed value (inclusive). Not checked if None.
    name : str, optional
        Parameter name used in error messages.
    b_return_repr : bool, optional
        If True, return repr(number) as a string; otherwise return the float.

    Returns
    -------
    str or float
        repr of the validated float if b_return_repr is True, else the float.

    Raises
    ------
    ValueError
        If the number is below min_value or above max_value.
    """
    number = float(number)
    if min_value is not None and number < min_value:
        if name is not None:
            raise ValueError(
                f"The parameter {name} is {number}, which is less than its minimum value: {min_value}"
            )
        else:
            raise ValueError(f"Number: {number} is less than its minimum value: {min_value}")
    if max_value is not None and number > max_value:
        if name is not None:
            raise ValueError(
                f"The parameter {name} is {number}, which is larger than its maximum value: {max_value}"
            )
        else:
            raise ValueError(f"Number: {number} is larger than its maximum value: {max_value}")
    if b_return_repr:
        return repr(number)
    else:
        return number


def boolean(b: bool, name: str | None = None) -> bool:
    """Validate that a value is a boolean.

    Parameters
    ----------
    b : bool
        Value to validate.
    name : str, optional
        Parameter name used in error messages.

    Returns
    -------
    bool
        The validated boolean value.

    Raises
    ------
    ValueError
        If b is not a bool instance.
    """
    if not isinstance(b, bool):
        if name is not None:
            raise ValueError(f"The parameter '{name}' must be a boolean value. Got:\n{b}")
        else:
            raise ValueError(f"Expected a boolean. Got:\n{b}")
    return bool(b)


def dictionary(
    obj: dict,
    name: str | None = None,
    b_return_repr: bool = True,
) -> str | dict:
    """Validate that an object is a dictionary.

    Parameters
    ----------
    obj : dict
        Object to validate.
    name : str, optional
        Parameter name used in error messages.
    b_return_repr : bool, optional
        If True, return repr(obj) as a string; otherwise return the dict.

    Returns
    -------
    str or dict
        repr of the validated dict if b_return_repr is True, else the dict.

    Raises
    ------
    ValueError
        If obj is not a dict.
    """
    if not isinstance(obj, dict):
        if name is not None:
            raise ValueError(f"The parameter '{name}' is must be a dictionary. Got:\n{obj}")
        else:
            raise ValueError(f"Expected a dictionary. Got:\n{obj}")
    obj = dict(obj)
    if b_return_repr:
        return repr(obj)
    else:
        return obj


def validate_vector(
    a: npt.ArrayLike,
    name_of_a: str,
    max_size: int,
    b_size_of_a_must_be_eq_to_max_size: bool = False,
    name_of_variable_with_equivalent_size: str | None = None,
) -> np.ndarray:
    """Validate that an array-like object is a 1-D vector within size limits.

    Parameters
    ----------
    a : array-like
        Object to validate and convert to a NumPy 1-D array.
    name_of_a : str
        Variable name used in error messages.
    max_size : int
        Maximum allowed number of elements.
    b_size_of_a_must_be_eq_to_max_size : bool, optional
        If True, the size of a must equal max_size exactly.
    name_of_variable_with_equivalent_size : str, optional
        Name of the variable that dictates the required size, used in error messages.

    Returns
    -------
    np.ndarray
        Validated 1-D NumPy array.

    Raises
    ------
    ValueError
        If a cannot be cast to an array, is not 1-D, has the wrong size, or
        exceeds max_size.
    """
    try:
        a = np.asarray(a)
    except Exception:
        raise ValueError(
            f"The object {name_of_a} could not be broadcasted into vector form: np.asarray({name_of_a})."
        )

    if a.ndim != 1:
        raise ValueError(f"The object {name_of_a} is not a vector, it has {a.ndim} dimensions.")

    if b_size_of_a_must_be_eq_to_max_size and a.size != max_size:
        raise ValueError(
            f"The vector {name_of_a} has a size: {a.size}, but its size must be equal to"
            f" {name_of_variable_with_equivalent_size}: {max_size}"
        )

    if a.size > max_size:
        raise ValueError(
            f"The vector {name_of_a} exceeds the maximum dimensions. It has size {a.size},"
            f" while the maximum allowed size is {max_size}"
        )

    return a


def validate_matrix(
    A: npt.ArrayLike,
    name_of_A: str,
    max_dim0: int,
    max_dim1: int,
    b_A_must_be_square: bool = False,
    b_A_must_be_symmetric: bool = False,
    b_A_must_be_semidefinite_positive: bool = False,
    tolerance: float = 1e-7,
) -> np.ndarray:
    """Validate that an array-like object is a 2-D matrix satisfying given constraints.

    Parameters
    ----------
    A : array-like
        Object to validate and convert to a NumPy 2-D array.
    name_of_A : str
        Variable name used in error messages.
    max_dim0 : int
        Maximum allowed size along axis 0 (rows).
    max_dim1 : int
        Maximum allowed size along axis 1 (columns).
    b_A_must_be_square : bool, optional
        If True, A must have equal number of rows and columns.
    b_A_must_be_symmetric : bool, optional
        If True, A must satisfy np.allclose(A, A.T, atol=tolerance, rtol=tolerance).
    b_A_must_be_semidefinite_positive : bool, optional
        If True, all eigenvalues of A must be >= -tolerance.
    tolerance : float, optional
        Tolerance used for symmetry and semidefiniteness checks.

    Returns
    -------
    np.ndarray
        Validated 2-D NumPy array.

    Raises
    ------
    ValueError
        If A cannot be cast to an array, is not 2-D, exceeds max dimensions,
        is not square when required, is not symmetric when required, or is not
        semidefinite positive when required.
    """
    try:
        A = np.asarray(A)
    except Exception:
        raise ValueError(
            f"The object {name_of_A} could not be broadcasted into matrix form: np.asarray({name_of_A})."
        )

    if A.ndim != 2:
        raise ValueError(f"The object {name_of_A} is not a matrix, it has {A.ndim} dimensions.")

    if A.shape[0] > max_dim0 or A.shape[1] > max_dim1:
        raise ValueError(
            f"The matrix {name_of_A} exceeds the maximum dimensions. It has shape {A.shape}, while the maximum allowed shape is ({max_dim0}, {max_dim1})"
        )

    if b_A_must_be_square and A.shape[0] != A.shape[1]:
        raise ValueError(
            f"The matrix {name_of_A} is not a square matrix. Its dimensions are {A.shape}"
        )

    if b_A_must_be_symmetric and not np.allclose(A, A.T, atol=tolerance, rtol=tolerance):
        raise ValueError(
            f"The matrix {name_of_A} is not symmetric. If suitable, try replacing it by: A = (A + A.T)/2"
        )

    if b_A_must_be_semidefinite_positive:
        eigenvalues_of_A = np.linalg.eigvalsh(A)
        if np.any(eigenvalues_of_A < -tolerance):
            raise ValueError(
                f"The matrix {name_of_A} is not semidefinite positive. Its lowest eigenvalue is {eigenvalues_of_A[0]}"
            )

    return A
