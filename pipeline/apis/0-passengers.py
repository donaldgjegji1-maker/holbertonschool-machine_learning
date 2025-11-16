#!/usr/bin/env python3
"""
A script that returns the list of ships that
can hold a given number of passengers
"""
import requests


def availableShips(passengerCount):
    """
    A method that returns the list of ships that
    can hold a given number of passengers
    """

    ships = []
    url = "https://swapi.dev/api/starships/"

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data.get("results", []):
            passengers = ship.get("passengers", "0")

            # Clean the passengers field
            try:
                passengers_num = int(passengers.replace(",", ""))
            except Exception:
                continue  # Skip ships without numeric passenger capacity

            if passengers_num >= passengerCount:
                ships.append(ship["name"])

        url = data.get("next")

    return ships
