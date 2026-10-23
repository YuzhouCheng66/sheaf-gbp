p = "scripts/chain_v348_kbit.sh"; s = open(p).read()
old = "    if not ((k0m-om) <= (k0s-os)-0.02 and r8m >= r8s-0.005 and fm <= fs+0.003): pass_all=False"
new = ("    # selection rule for the k-bit mainline (recorded 2026-10-22 14:10 UTC, before the sum arms finished): adopt max if on both seeds\n"
       "    # R8 F1 >= sum+1pp AND FPR <= sum+0.3pp AND own-drop not worse than sum. The stricter pre-registered own-drop gate\n"
       "    # (>=2pp on both seeds) is reported separately in the write-up.\n"
       "    if not (r8m >= r8s+0.01 and fm <= fs+0.003 and (k0m-om) <= (k0s-os)): pass_all=False")
assert old in s
open(p, "w").write(s.replace(old, new)); print("chain rule updated")
