#!/bin/bash
# HM3D v0.2 下载器（Claude, 2026-10-24）。由数据集许可人本人在自己的终端运行：
#   ssh -t uk-linux bash /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/scripts/download_hm3d_v02.sh
# 它不回显地询问 Matterport API token（ID + secret），只把它们写进一个 0700 目录里的 0600 curl 配置文件，
# 不出现在命令行参数、进程列表或 shell 历史里，下载结束（成功或失败）即删除。
# 先用 1 字节请求验证权限；通过后转入后台：续传下载 glb 格式 minival（464 MB）、val（4 GB）、train（32 GB），
# 每个包 tar -t 校验后解包到 /mnt/datasets/hm3d-v0.2/{minival,val,train}，写 manifest.json。可重复运行（续传）。
# 许可：仅限学术、非商业用途，条款 https://matterport.com/matterport-end-user-license-agreement-academic-use-model-data
set -u
DEST=/mnt/datasets/hm3d-v0.2
BASE=https://api.matterport.com/resources/habitat
FILES="hm3d-minival-glb-v0.2.tar hm3d-val-glb-v0.2.tar hm3d-train-glb-v0.2.tar"
mkdir -p "$DEST/archives"

if [ "${1:-}" != "--worker" ]; then
  echo "HM3D v0.2 下载（仅限学术、非商业用途）。"
  echo "条款：https://matterport.com/matterport-end-user-license-agreement-academic-use-model-data"
  echo "需要：已在 Matterport 获得 HM3D 访问权限，并在 https://my.matterport.com/settings/account/devtools 生成的 API token。"
  read -r -p "Matterport API token ID: " TID
  read -r -s -p "Matterport API token secret（输入不显示）: " TSEC; echo
  [ -n "$TID" ] && [ -n "$TSEC" ] || { echo "ID 或 secret 为空，退出。"; exit 2; }
  umask 077
  D=$(mktemp -d "$HOME/.hm3d_dl.XXXXXX")
  esc(){ local s=${1//\\/\\\\}; printf '%s' "${s//\"/\\\"}"; }
  printf 'user = "%s:%s"\n' "$(esc "$TID")" "$(esc "$TSEC")" > "$D/curl.cfg"
  unset TSEC
  code=$(curl -s -K "$D/curl.cfg" --location -r 0-0 -o /dev/null -w "%{http_code}" --max-time 60 "$BASE/hm3d-minival-glb-v0.2.tar")
  if [ "$code" != "206" ] && [ "$code" != "200" ]; then
    echo "认证失败（HTTP $code）：请确认这个 Matterport 账号已获准访问 HM3D，且用的是 API token 的 ID 和 secret（不是账号密码）。"
    rm -rf "$D"; exit 3
  fi
  nohup setsid bash "$0" --worker "$D" >> "$DEST/download.log" 2>&1 < /dev/null &
  echo "认证通过。下载已在后台开始（约 36.5 GB，可续传），日志：$DEST/download.log。现在可以关闭这个窗口。"
  exit 0
fi

# ---------------------------------------------------------------- worker (background)
D=$2
trap 'rm -rf "$D"' EXIT
log(){ echo "$(date -u +%FT%TZ) $*"; }
log "worker start"
for f in $FILES; do
  ok=0
  for attempt in $(seq 1 20); do
    if curl -s -K "$D/curl.cfg" --location --fail --retry 5 --retry-delay 10 --continue-at - -o "$DEST/archives/$f" "$BASE/$f"; then ok=1; break; fi
    rc=$?
    [ $rc -eq 33 ] && { ok=1; break; }                     # 33 = range not satisfiable: the file is already complete
    log "$f attempt $attempt failed (curl $rc); retry in 60 s"; sleep 60
  done
  [ $ok -eq 1 ] || { log "ABORT: $f could not be downloaded"; exit 4; }
  log "$f downloaded: $(stat -c %s "$DEST/archives/$f") bytes"
done
rm -rf "$D"; log "credentials removed"
for f in $FILES; do
  split=$(echo "$f" | sed -E 's/hm3d-(minival|val|train)-glb-v0.2.tar/\1/')
  tar -tf "$DEST/archives/$f" > "$DEST/archives/$f.list" || { log "ABORT: $f is not a valid tar"; exit 5; }
  mkdir -p "$DEST/$split"; tar -xf "$DEST/archives/$f" -C "$DEST/$split" || { log "ABORT: extracting $f failed"; exit 5; }
  log "$f extracted to $DEST/$split ($(grep -c '\.glb$' "$DEST/archives/$f.list") glb files)"
done
python3 - "$DEST" <<'PY'
import hashlib, json, os, sys, time
dest = sys.argv[1]; rows = []
for f in sorted(os.listdir(os.path.join(dest, 'archives'))):
    if not f.endswith('.tar'):
        continue
    p = os.path.join(dest, 'archives', f); h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''):
            h.update(b)
    rows.append(dict(file=f, bytes=os.path.getsize(p), sha256=h.hexdigest()))
scenes = {s: sorted(d for d in os.listdir(os.path.join(dest, s)) if os.path.isdir(os.path.join(dest, s, d))) for s in ('example', 'minival', 'val', 'train') if os.path.isdir(os.path.join(dest, s))}
json.dump(dict(dataset='HM3D v0.2 glb', license='academic non-commercial (Matterport EULA)', archives=rows, scene_counts={k: len(v) for k, v in scenes.items()}, scenes=scenes, written=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())),
          open(os.path.join(dest, 'manifest.json'), 'w'), indent=1)
print('manifest', {k: len(v) for k, v in scenes.items()})
PY
log "DONE"
