import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import tomllib

debug = False
_base_url = "https://www.inspiration-q.com/api"
_url_dict = {}
_auth = {}


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
            with open(iq_credentials_file, "r") as f:
                iq_credentials = tomllib.load(f)
                api_key = iq_credentials["azure"]["api_key"]
    return {"Ocp-Apim-Subscription-Key": api_key}


def set_url_dict():
    """Populate the global URL dictionary from the known entry points."""
    global _url_dict
    _url_dict = _specialize(_base_url, _known_entry_points)
    return


def set_auth(auth):
    """Set the global authentication header dictionary."""
    global _auth
    _auth = auth
    return


def post(*args, **kwdargs):
    """Send a POST request to the Inspiration-Q API and return the response.

    Wraps _post with the global base URL, URL dictionary, and auth headers.
    Raises an exception if the response contains an error.

    Parameters
    ----------
    *args
        Positional arguments forwarded to _post (first is the function name).
    **kwdargs
        Keyword arguments forwarded to _post (e.g., json payload).

    Returns
    -------
    dict
        Parsed JSON response from the API.

    Raises
    ------
    Exception
        If the API returns an exception field in the response body.
    """
    r_json = _post(_base_url, _url_dict, _auth, *args, **kwdargs)
    if "exception" in r_json:
        raise Exception(r_json["exception"])
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
    Exception
        If the HTTP response indicates an error.
    """
    if debug:
        _print_call("GET", url, headers, arguments)
    r = requests.get(url=url, headers=headers, **arguments)
    if debug:
        print(f"Received request Response {r} with content:\n{r.content}")
    if not r.ok:
        raise Exception(f"Error returned from Inspiration-Q API: {r}")
    return r.json()


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
    Exception
        If the function is unknown, the API returns an error,
        or the computation exceeds the maximum allowed time.
    """
    headers = auth

    if function not in url_dict:
        error_msg = f"Unknown API function {function}"
        if debug:
            print(url_dict)
        raise Exception(error_msg)

    url = url_dict[function]
    if debug:
        _print_call("POST", url, headers, arguments)
    r = requests.post(url=url, headers=headers, **arguments)
    if debug:
        print(f"Received request Response {r} with content:\n{r.content}")
    if not r.ok:
        error = f"Error returned from Inspiration-Q API. function: {function}. response: {r}"
        if debug:
            print(error)
        raise Exception(error)

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

    if debug:
        print(f"GET url for '{function}' is {url}")

    tottime = 0
    API_timeout = 12 * 3600
    valid_solution_identifiers = ["solution", "named_solution", "zscore"]
    while tottime < API_timeout:
        if debug:
            print(f"Waiting for {waittime}s for '{function}' to complete")

        time.sleep(waittime)
        tottime += waittime

        body = _get(url, headers)

        for solution_identifier in valid_solution_identifiers:
            if solution_identifier in body and body[solution_identifier]:
                if debug:
                    print(f"Found body:\n{body}")
                    print("solution found")
                return body

        if "status" in body and body["status"] == "Failed":
            return body

        waittime = min(2 * waittime, 5)

    if tottime >= API_timeout:
        raise Exception(
            f"Exceeded maximum time to complete computation with method '{function}'. "
            f" Maximum time: {API_timeout / 3600:2.1f}h"
        )

    raise Exception(f"Unable to complete computation with method '{function}'")


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


def _maybe_trim(string, length=69):
    """Trim a string for concise debug output when debug mode is 'short'.

    Parameters
    ----------
    string : str
        The string to potentially trim.
    length : int
        Maximum length before trimming is applied (default 69).

    Returns
    -------
    str
        The original string, or a trimmed version with ellipsis if too long.
    """
    if debug == "short" and len(string) > length:
        return string[:length] + "..." + string[-1]
    return string


def _print_call(type, url, headers, arguments):
    """Print a formatted debug message describing an outgoing HTTP request.

    Parameters
    ----------
    type : str
        HTTP method string (e.g., 'GET' or 'POST').
    url : str
        Full URL of the request.
    headers : dict
        Dictionary of HTTP headers being sent.
    arguments : dict
        Dictionary of additional request arguments.
    """
    print(
        f"Sending {type} request to {url}\nwith header: {headers}\n"
        f"and arguments: {_maybe_trim(str(arguments))}\n"
        f"at timestamp {datetime.now(timezone.utc)}",
        flush=True,
    )
    return


_known_entry_points = [
    "v1/iq-xtreme/qubo",
    "v1/iq-xtreme/qudo",
    "v1/iq-xtreme/quco",
    "v1/iq-xtreme/ccqp",
    "v1/iq-xtreme/tsp",
]
