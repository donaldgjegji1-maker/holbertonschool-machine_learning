#!/usr/bin/env python3
"""
A script that displays the number of launches per rocket
"""
import requests
from collections import defaultdict


def get_launches_per_rocket():
    """
    A method that displays the number of launches per rocket
    """

    try:
        rockets_url = "https://api.spacexdata.com/v4/rockets"
        rockets_response = requests.get(rockets_url)
        rockets_response.raise_for_status()
        rockets_data = rockets_response.json()

        # Create rocket ID to name mapping
        rocket_id_to_name = {}
        for rocket in rockets_data:
            rocket_id_to_name[rocket['id']] = rocket['name']

        launches_url = "https://api.spacexdata.com/v4/launches"
        launches_response = requests.get(launches_url)
        launches_response.raise_for_status()
        launches = launches_response.json()

        # Count launches per rocket
        rocket_counts = defaultdict(int)
        for launch in launches:
            rocket_id = launch.get('rocket')
            if rocket_id and rocket_id in rocket_id_to_name:
                rocket_name = rocket_id_to_name[rocket_id]
                rocket_counts[rocket_name] += 1

        # Sort by count (descending) then by name (A to Z)
        sorted_rockets = sorted(
            rocket_counts.items(),
            key=lambda x: (-x[1], x[0])
        )

        return sorted_rockets

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


if __name__ == '__main__':
    results = get_launches_per_rocket()
    for rocket_name, count in results:
        print(f"{rocket_name}: {count}")
