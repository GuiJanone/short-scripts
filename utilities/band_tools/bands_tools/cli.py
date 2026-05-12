import argparse

from .effective_mass import compute_effective_masses, print_effective_mass_results
from .parser import load_band_blocks, load_json_config, parse_red_bands, write_example_json
from .plotting import plot_bands


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot electronic band structure from JSON config")
    parser.add_argument("--json", help="JSON config file")
    parser.add_argument("--generate-example", action="store_true", help="Generate an example JSON file")
    args = parser.parse_args()

    if args.generate_example:
        write_example_json()
        return

    if not args.json:
        parser.error("--json is required unless --generate-example is used")

    params = load_json_config(args.json)
    blocks = load_band_blocks(params["bands_file"])
    red_bands = parse_red_bands(params.get("red_bands", []), len(blocks))

    plot_bands(blocks, params, red_bands)

    if "effective_mass" in params:
        effective_mass_config = params["effective_mass"]
        if not isinstance(effective_mass_config, dict):
            raise ValueError("effective_mass must be a JSON object")

        enabled = effective_mass_config.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("effective_mass.enabled must be true or false")

        if enabled:
            results = compute_effective_masses(
                blocks,
                effective_mass_config,
                efermi=float(params.get("Efermi", 0.0)),
            )
            print_effective_mass_results(results)
