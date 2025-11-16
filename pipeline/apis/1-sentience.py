#!/usr/bin/env python3
import requests
"""
A script that returns a list of names of the home
planets of all sentient species.
"""


def sentientPlanets():
    """
    A script that returns a list of names of the home
    planets of all sentient species.
    """

    base_url = "https://swapi.dev/api/"
    species_url = f"{base_url}species/"
    planets_set = set()

    while species_url:
        response = requests.get(species_url)
        if response.status_code != 200:
            break

        data = response.json()

        for species in data['results']:
            # Check if species is sentient
            classification = species.get('classification', '').lower()
            designation = species.get('designation', '').lower()

            if 'sentient' in classification or 'sentient' in designation:
                homeworld_url = species.get('homeworld')
                if homeworld_url:
                    planet_response = requests.get(homeworld_url)
                    if planet_response.status_code == 200:
                        planet_data = planet_response.json()
                        planets_set.add(planet_data['name'])

        species_url = data.get('next')

    return sorted(list(planets_set))
