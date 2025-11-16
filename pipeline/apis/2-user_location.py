#!/usr/bin/env python3
import requests
import sys
from datetime import datetime, timedelta
"""
A script that prints the location of a specific user
"""


def get_user_location(api_url):
    """
    A method that prints the location of a specific user
    """

    try:
        response = requests.get(api_url)

        # Handle rate limiting (403 status code)
        if response.status_code == 403:
            reset_timestamp = int(response.headers.get('X-RateLimit-Reset', 0))
            current_timestamp = datetime.now().timestamp()
            minutes_left = (reset_timestamp - current_timestamp) / 60
            minutes_until_reset = max(0, minutes_left)
            return f"Reset in {int(minutes_until_reset)} min"

        # Handle user not found (404 status code)
        elif response.status_code == 404:
            return "Not found"

        # Handle other non-200 status codes
        elif response.status_code != 200:
            return "Not found"

        # Success case - extract location
        user_data = response.json()
        location = user_data.get('location')

        # Return location or 'Not found' if location is empty/None
        return location if location else "Not found"

    except requests.exceptions.RequestException:
        return "Not found"


if __name__ == '__main__':
    # Check if URL argument is provided
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <github_api_url>")
        sys.exit(1)

    api_url = sys.argv[1]
    result = get_user_location(api_url)
    print(result)
