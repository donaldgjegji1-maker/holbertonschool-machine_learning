#!/usr/bin/env python3
"""
A script that displays the first launch with these information
"""
import requests
from datetime import datetime


def get_first_upcoming_launch():
    """
    A mnethod that displays the first launch with these information
    """

    try:
        url = "https://api.spacexdata.com/v4/launches/upcoming"

        response = requests.get(url)
        response.raise_for_status()

        launches = response.json()

        if not launches:
            return "No upcoming launches found"

        # Sort launches by date_unix (ascending) to get the earliest first
        launches.sort(key=lambda x: x.get('date_unix', float('inf')))

        launch = launches[0]

        launch_name = launch.get('name', 'Unknown Mission')

        # Format date from ISO string to local time format
        date_utc = launch.get('date_utc')
        if date_utc:
            date_local = launch.get('date_local', date_utc)
        else:
            date_local = 'Unknown Date'

        rocket_id = launch.get('rocket')
        rocket_name = 'Unknown Rocket'
        if rocket_id:
            rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
            rocket_response = requests.get(rocket_url)
            if rocket_response.status_code == 200:
                rocket_data = rocket_response.json()
                rocket_name = rocket_data.get('name', 'Unknown Rocket')

        # Get launchpad information
        launchpad_id = launch.get('launchpad')
        launchpad_name = 'Unknown Launchpad'
        launchpad_locality = 'Unknown Locality'

        if launchpad_id:
            base_url = "https://api.spacexdata.com/v4/launchpads"
            launchpad_url = f"{base_url}/{launchpad_id}"
            launchpad_response = requests.get(launchpad_url)
            if launchpad_response.status_code == 200:
                launchpad_data = launchpad_response.json()
                launchpad_name = launchpad_data.get(
                    'name', 'Unknown Launchpad'
                    )
                launchpad_locality = launchpad_data.get(
                    'locality', 'Unknown Locality'
                    )

        # Format the output as specified
        output = f"{launch_name} ({date_local}) {rocket_name} - "
        output += f"{launchpad_name} ({launchpad_locality})"
        return output

    except requests.exceptions.RequestException as e:
        return f"Error fetching data: {e}"
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == '__main__':
    result = get_first_upcoming_launch()
    print(result)
