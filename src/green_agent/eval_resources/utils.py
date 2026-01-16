import requests


def verify_fhir_server(fhir_api_base):
    """Verify connection to FHIR server."""
    res = send_get_request(f'{fhir_api_base}metadata')
    return res.get('status_code', 0) == 200


def send_get_request(url, params=None, headers=None):
    """Send a GET HTTP request to the given URL."""
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return {
            "status_code": response.status_code,
            "data": response.json() if response.headers.get('Content-Type') == 'application/json' else response.text
        }
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def send_post_request(url, payload, headers=None):
    """Send a POST HTTP request to the given URL with JSON payload."""
    try:
        default_headers = {"Content-Type": "application/json"}
        if headers:
            default_headers.update(headers)
        
        response = requests.post(url, json=payload, headers=default_headers)
        response.raise_for_status()
        
        try:
            data = response.json()
        except ValueError:
            data = response.text
            
        return {
            "status_code": response.status_code,
            "data": data
        }
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}