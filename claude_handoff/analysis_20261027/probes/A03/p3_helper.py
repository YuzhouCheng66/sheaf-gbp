"""p3_helper.py (A03): run p3_box_codec.house() on the LAST houses of the list (reverse order) in a second pool, to finish p3 sooner;
skips houses whose cache file already exists at the moment a worker starts them (results are deterministic, so an overlap with the
main run only wastes time)."""
import os, sys
import p3_box_codec as P


def one(h):
    if os.path.exists(os.path.join(P.CACHE, h + '.json')):
        return h
    return P.house(h)


if __name__ == '__main__':
    import multiprocessing as mp
    todo = [h for h in reversed(P.names()) if not os.path.exists(os.path.join(P.CACHE, h + '.json'))][:int(sys.argv[2])]
    with mp.Pool(int(sys.argv[1]), maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(one, todo):
            pass
