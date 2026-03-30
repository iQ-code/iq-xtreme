import numpy as np


def integer(number, min_value, max_value, name=None, b_return_repr=True):
    number = int(number)
    if number < min_value or number > max_value:
        if name is not None:
            raise ValueError(f'The parameter {name} is {number}, which is out of range: ({min_value}, {max_value})')
        else:
            raise ValueError(f'Number: {number} is out of range: ({min_value}, {max_value})')
    if b_return_repr:
        return repr(number)
    else:
        return number


def integer_or_string(obj):
    if isinstance(obj, str):
        return obj
    obj = int(obj)
    return repr(obj)


def string(obj, name=None):
    if isinstance(obj, str):
        return obj
    else:
        if name is not None:
            raise ValueError(f"The parameter '{name}' must be a string. Got:\n{obj}")
        else:
            raise ValueError(f"Expected a string. Got:\n{obj}")


def rng(obj):
    if isinstance(obj, np.random.Generator):
        obj = obj.integers(0, 0xffffffff, dtype=np.uint32)
    elif not isinstance(obj, (int, str, list)):
        raise ValueError(f"Expected a random number generator or a valid seed. Got:\n{obj}")
    return repr(obj)


def real(number, min_value=None, max_value=None, name=None, b_return_repr=True):
    number = float(number)
    if min_value is not None and number < min_value:
        if name is not None:
            raise ValueError(f"The parameter {name} is {number}, which is less than its minimum value: {min_value}")
        else:
            raise ValueError(f"Number: {number} is less than its minimum value: {min_value}")
    if max_value is not None and number > max_value:
        if name is not None:
            raise ValueError(f"The parameter {name} is {number}, which is larger than its maximum value: {max_value}")
        else:
            raise ValueError(f"Number: {number} is larger than its maximum value: {max_value}")
    if b_return_repr:
        return repr(number)
    else:
        return number


def boolean(b, name=None):
    if not isinstance(b, bool):
        if name is not None:
            raise ValueError(f"The parameter '{name}' must be a boolean value. Got:\n{b}")
        else:
            raise ValueError(f"Expected a boolean. Got:\n{b}")
    return bool(b)


def dictionary(obj, name=None, b_return_repr=True):
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
    a, name_of_a, max_size,
    b_size_of_a_must_be_eq_to_max_size=False,
    name_of_variable_with_equivalent_size=None
):
    try:
        a = np.asarray(a)
    except Exception:
        raise ValueError(f"The object {name_of_a} could not be broadcasted into vector form: np.asarray({name_of_a}).")

    if a.ndim != 1:
        raise ValueError(f"The object {name_of_a} is not a vector, it has {a.ndim} dimensions.")

    if b_size_of_a_must_be_eq_to_max_size and a.size != max_size:
        raise ValueError(f"The vector {name_of_a} has a size: {a.size}, but its size must be equal to {name_of_variable_with_equivalent_size}: {max_size}")

    if a.size > max_size:
        raise ValueError(f"The vector {name_of_a} exceeds the maximum dimensions. It has size {a.size}, while the maximum allowed size is {max_size}")

    return a


def validate_matrix(
    A, name_of_A,
    max_dim0, max_dim1,
    b_A_must_be_square=False,
    b_A_must_be_symmetric=False,
    b_A_must_be_semidefinite_positive=False,
    tolerance=1e-7
):
    try:
        A = np.asarray(A)
    except Exception:
        raise ValueError(f"The object {name_of_A} could not be broadcasted into matrix form: np.asarray({name_of_A}).")

    if A.ndim != 2:
        raise ValueError(f"The object {name_of_A} is not a matrix, it has {A.ndim} dimensions.")

    if A.shape[0] > max_dim0 or A.shape[1] > max_dim1:
        raise ValueError(f"The matrix {name_of_A} exceeds the maximum dimensions. It has shape {A.shape}, while the maximum allowed shape is ({max_dim0}, {max_dim1})")

    if b_A_must_be_square and A.shape[0] != A.shape[1]:
        raise ValueError(f"The matrix {name_of_A} is not a square matrix. Its dimensions are {A.shape}")

    if b_A_must_be_symmetric and not np.allclose(A, A.T, atol=tolerance, rtol=tolerance):
        raise ValueError(f"The matrix {name_of_A} is not symmetric. If suitable, try replacing it by: A = (A + A.T)/2")

    if b_A_must_be_semidefinite_positive:
        eigenvalues_of_A = np.linalg.eigvalsh(A)
        if np.any(eigenvalues_of_A < -tolerance):
            raise ValueError(f"The matrix {name_of_A} is not semidefinite positive. Its lowest eigenvalue is {eigenvalues_of_A[0]}")

    return A
