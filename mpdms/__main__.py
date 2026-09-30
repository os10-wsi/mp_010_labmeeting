"""python -m mpdms {init,run} ..."""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("init", "run"):
        sys.exit("usage: python -m mpdms init <dms_folder>... | python -m mpdms run [configs/*.yaml] [--only a01,a02]")
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "init":
        from .init_dataset import main as m
    else:
        from .runner import main as m
    m(rest)


if __name__ == "__main__":
    main()
