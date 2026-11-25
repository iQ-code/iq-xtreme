import logging
import time
import types
from pathlib import Path

import requests
import tomllib

_base_url = "https://www.inspiration-q.com/api"
# SimpleNamespace is a bare object whose attributes can be freely set.
# Mutating attributes on an existing object never requires the `global` keyword.
_state = types.SimpleNamespace(auth={}, url_dict={})
_HTTP_TIMEOUT = 30  # seconds per HTTP request

logger = logging.getLogger(__name__)


def initialize_credentials(api_key):
    """Initialize the global authentication headers and URL dictionary."""
    set_auth(_build_auth(api_key))
    set_url_dict()
    return


def _build_auth(api_key):
    """Build the authentication header dictionary from an API key."""
    if api_key == "YOUR_API_KEY":
        iq_credentials_file = Path.home() / ".iq_config.toml"
        if iq_credentials_file.is_file():
            with open(iq_credentials_file, "rb") as f:
                iq_credentials = tomllib.load(f)
                api_key = iq_credentials["azure"]["api_key"]
    return {"Ocp-Apim-Subscription-Key": api_key}


def set_url_dict():
    """Populate the URL dictionary from the known entry points."""
    _state.url_dict = _specialize(_base_url, _known_entry_points)
    return


def set_auth(auth):
    """Set the authentication header dictionary."""
    _state.auth = auth
    return


class APIError(requests.HTTPError, ConnectionError):
    """Error answered by the Inspiration-Q API: a rejected request or a failed computation.

    It subclasses requests.HTTPError and ConnectionError in every Inspiration-Q SDK, so
    callers can catch API errors with either type.

    Attributes
    ----------
    description : str
        Reason given by the API.
    computation_id : str or None
        ID of the failed computation, if the API had created one.
    response : requests.Response or None
        HTTP response of a rejected request.
    """

    def __init__(self, message, description="", computation_id=None, response=None):
        super().__init__(message, response=response)
        self.description = description
        self.computation_id = computation_id


def post(function, *args, **kwargs):
    """Send a POST request to the Inspiration-Q API and return the response.

    Wraps _post with the global base URL, URL dictionary, and auth headers.

    Parameters
    ----------
    function : str
        API endpoint name (key in the URL dictionary).
    *args
        Positional arguments forwarded to _post after the function name.
    **kwargs
        Keyword arguments forwarded to _post (e.g., json payload).

    Returns
    -------
    dict
        Parsed JSON response from the API.

    Raises
    ------
    APIError
        If the API rejects the request, returns an exception field in the response body,
        or the computation fails.
    """
    r_json = _post(_base_url, _state.url_dict, _state.auth, function, *args, **kwargs)
    if "exception" in r_json:
        raise APIError(r_json["exception"], str(r_json["exception"]))
    if r_json.get("status") == "Failed":
        description = r_json.get("error_description") or "The API gave no error description."
        computation_id = r_json.get("computationId")
        raise APIError(
            f"Computation {computation_id} of the Inspiration-Q API function {function} failed:"
            f" {description}",
            description,
            computation_id,
        )
    return r_json


def _get(url, headers, **arguments):
    """Send a GET request and return the parsed JSON response.

    Parameters
    ----------
    url : str
        Full URL to send the request to.
    headers : dict
        Dictionary of HTTP headers (including auth).
    **arguments
        Additional keyword arguments passed to requests.get.

    Returns
    -------
    dict
        Parsed JSON response body.

    Raises
    ------
    APIError
        If the HTTP response indicates an error.
    """
    logger.debug(f"Sending GET request to {url} with headers {headers} and arguments {arguments}")
    r = requests.get(url=url, headers=headers, timeout=_HTTP_TIMEOUT, **arguments)
    logger.debug(f"Received GET response {r} with content: {r.content}")
    if not r.ok:
        reason = _error_reason(r)
        raise APIError(
            f"Error returned from Inspiration-Q API: {r}. {reason}".rstrip(), reason, response=r
        )
    return r.json()


def _error_reason(r):
    """Return the reason the API gave in an error answer.

    Parameters
    ----------
    r : requests.Response
        Unsuccessful HTTP response from the API.

    Returns
    -------
    str
        Title and detail of the RFC 9110 problem details object the API answers errors
        with, or the raw body if it is not one.
    """
    try:
        body = r.json()
    except ValueError:
        return r.text.strip()
    if not isinstance(body, dict):
        return str(body)
    # The API answers errors with an RFC 9110 problem details object.
    reason = ": ".join(str(body[k]) for k in ("title", "detail") if body.get(k))
    return reason or str(body)


def _post(base_url, url_dict, auth, function, waittime=1, **arguments):
    """Send a POST request and poll for the result until the computation completes.

    Submits the request to the appropriate endpoint and then polls the
    computation status via GET until a valid solution is found, the
    computation fails, or the timeout is exceeded.

    Parameters
    ----------
    base_url : str
        Base URL of the Inspiration-Q API.
    url_dict : dict
        Dictionary mapping function names to their full URLs.
    auth : dict
        Dictionary containing the authentication headers.
    function : str
        API endpoint name (key in url_dict).
    waittime : float
        Initial polling interval in seconds. Doubles up to 5s (default 1).
    **arguments
        Additional keyword arguments passed to requests.post (e.g., json).

    Returns
    -------
    dict
        Parsed JSON response body containing the computation result.

    Raises
    ------
    ValueError
        If the function name is not in the known URL dictionary.
    APIError
        If the API returns an HTTP error response.
    TimeoutError
        If the computation exceeds the maximum allowed time.
    """
    headers = auth

    if function not in url_dict:
        logger.debug(f"Known URL dict: {url_dict}")
        raise ValueError(f"Unknown API function {function}")

    url = url_dict[function]
    logger.debug(f"Sending POST request to {url} with headers {headers} and arguments {arguments}")
    r = requests.post(url=url, headers=headers, timeout=_HTTP_TIMEOUT, **arguments)
    logger.debug(f"Received POST response {r} with content: {r.content}")
    if not r.ok:
        reason = _error_reason(r)
        raise APIError(
            f"Error returned from Inspiration-Q API. function: {function}. response: {r}."
            f" {reason}".rstrip(),
            reason,
            response=r,
        )

    body = r.json()

    if not body.get("status", False):
        return body

    # response has this format (we add "-" to avoid Ruff warnings):
    # { -
    #     "computationId": "9970243c-44f7-4150-ab81-29c725feeabf", -
    #     "status": "Pending", -
    #     "computationStoreTime":"2024-03-14T17:06:15.452113+00:00"
    # } -
    # We do a GET to function/{computationId} to retrieve the actual response once calc finishes.

    url = base_url + "/" + function + "/" + body["computationId"]
    logger.debug(f"GET url for '{function}' is {url}")

    tottime = 0
    API_timeout = 12 * 3600
    valid_solution_identifiers = ["solution", "named_solution", "zscore"]
    while tottime < API_timeout:
        logger.debug(f"Waiting for {waittime}s for '{function}' to complete")

        time.sleep(waittime)
        tottime += waittime

        body = _get(url, headers)

        for solution_identifier in valid_solution_identifiers:
            if solution_identifier in body and body[solution_identifier]:
                logger.debug("Solution found in body: %s", body)
                return body

        if "status" in body and body["status"] == "Failed":
            return body

        waittime = min(2 * waittime, 5)

    raise TimeoutError(
        f"Exceeded maximum time to complete computation with method '{function}'."
        f" Maximum time: {API_timeout / 3600:2.1f}h"
    )


def _specialize(base_url, entry_points):
    """Build a dictionary mapping each entry point to its full URL.

    Parameters
    ----------
    base_url : str
        Base URL of the Inspiration-Q API.
    entry_points : iterable of str
        API endpoint path strings.

    Returns
    -------
    dict
        Mapping from endpoint name to its full URL.
    """
    return {k: base_url + "/" + k for k in entry_points}


_known_entry_points = [
    "v1/iq-xtreme/qubo",
    "v1/iq-xtreme/qudo",
    "v1/iq-xtreme/quco",
    "v1/iq-xtreme/ccqp",
    "v1/iq-xtreme/tsp",
]
