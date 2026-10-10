"""python -m mpdms {init,run} ..."""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("init", "run", "helix", "esm", "validate", "bench", "family", "tmhmm", "report", "quickstart", "batch", "panel", "aggregate"):
        sys.exit("usage: python -m mpdms quickstart <fitness_file> --gff3 <file>   (start here)"
                 " | python -m mpdms batch <folder_of_dimsum_output>"
                 " | python -m mpdms panel configs/*.yaml --only a01,a15"
                 " | python -m mpdms aggregate configs/*.yaml"
                 " | python -m mpdms init <dms_folder>... | python -m mpdms run [configs/*.yaml] [--only a01,a02]"
                 " | python -m mpdms helix configs/A.yaml configs/B.yaml ..."
                 " | python -m mpdms esm configs/A.yaml ... [--models 1,2,3,4,5]"
                 " | python -m mpdms validate configs/*.yaml"
                 " | python -m mpdms bench configs/*.yaml [--split protein]"
                 " | python -m mpdms family configs/AQR1.yaml configs/QDR2.yaml"
                 " | python -m mpdms tmhmm configs/*.yaml"
                 " | python -m mpdms report configs/AQR1.yaml configs/QDR2.yaml")
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "quickstart":
        from .quickstart import main as m
    elif cmd == "batch":
        from .batch import main as m
    elif cmd == "panel":
        from .multipanel import main as m
    elif cmd == "aggregate":
        from .aggregate import main as m
    elif cmd == "init":
        from .init_dataset import main as m
    elif cmd == "report":
        from .report_cli import main as m
    elif cmd == "tmhmm":
        from .topology import main as m
    elif cmd == "family":
        from .family import main as m
    elif cmd == "validate":
        from .validate import main as m
    elif cmd == "bench":
        from .validate import bench_main as m
    elif cmd == "esm":
        from .esm_score import main as m
    elif cmd == "helix":
        from .helix_pooled import main as m
    else:
        from .runner import main as m
    m(rest)


if __name__ == "__main__":
    main()
