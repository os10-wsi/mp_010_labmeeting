"""python -m mpdms {init,run} ..."""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("init", "run", "helix", "esm"):
        sys.exit("usage: python -m mpdms init <dms_folder>... | python -m mpdms run [configs/*.yaml] [--only a01,a02]"
                 " | python -m mpdms helix configs/A.yaml configs/B.yaml ..."
                 " | python -m mpdms esm configs/A.yaml ... [--models 1,2,3,4,5]")
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "init":
        from .init_dataset import main as m
    elif cmd == "esm":
        from .esm_score import main as m
    elif cmd == "helix":
        from .helix_pooled import main as m
    else:
        from .runner import main as m
    m(rest)


if __name__ == "__main__":
    main()
