import argparse
from .protocol import export_comparison


def main():
    parser = argparse.ArgumentParser(description="Extract all eight raw outputs from all five conditions")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    export_comparison(args.results)


if __name__ == "__main__":
    main()
