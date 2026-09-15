import argparse
from .protocol import write_json
from .upstream import official_fsr, verify_sources


def main():
    parser = argparse.ArgumentParser(description="Run the original, unmodified IF-SFT FSR function")
    parser.add_argument("jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify_sources()
    result = official_fsr(args.jsonl)
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
