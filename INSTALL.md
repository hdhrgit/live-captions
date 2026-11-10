# Installing captions-dvr

There are two pieces:

| Where | What | Port |
|---|---|---|
| Channels DVR box (dvr-host, Debian, systemd) | relay: pulls the streams, adds English subtitles (WebVTT) and captions (608/708), serves HLS + M3U + XMLTV | 8095 |
| Mac M4 (Apple Silicon) | speech service: Korean audio to English text with mlx-whisper large-v3 | 8096 |

Install the Mac first, so the relay has something to talk to. Both installers
can be re-run to upgrade.

## 1. Mac M4: speech service

Requirements: Apple Silicon, `python3` 3.9 or newer (Xcode CLT or Homebrew),
and git. Run as your normal user, not with sudo.

```sh
git clone <repository-url> ~/src/captions-dvr
cd ~/src/captions-dvr
./mac/install.sh
# optional overrides:
#   PORT=8096 MODEL=mlx-community/whisper-large-v3-mlx ./mac/install.sh
#   MODEL=mlx-community/whisper-medium-mlx   (faster, less accurate)
```

The installer does the following:

- creates a venv in `~/captions-asr/venv` with mlx-whisper and faster-whisper
  (faster-whisper is only used for Silero VAD)
- copies the server to `~/captions-asr/app`
- downloads and warms up the model (large-v3 is about 3 GB, so the first run
  takes a while)
- installs `~/Library/LaunchAgents/com.captions-dvr.asr.plist`. This runs
  `caffeinate -s -i python asr_server.py` with `KeepAlive`, listening on
  `0.0.0.0:$PORT`.
- finishes with `curl http://localhost:$PORT/health`

**Keep the Mac available:**

- The service runs under `caffeinate -s -i`, which prevents system sleep
  (on AC power) while the service is running. On a Mac mini used as a
  server, also set:
  ```sh
  sudo pmset -a sleep 0 disksleep 0 powernap 0 standby 0   # never system-sleep
  sudo pmset -a womp 1 autorestart 1                       # wake on LAN, restart after power loss
  ```
  Check with `pmset -g assertions` (expect PreventSystemSleep from caffeinate)
  and `pmset -g log | grep -E " (Sleep|DarkWake) "` (expect no new sleeps).
- On a laptop, keep it on power; with the lid closed it still sleeps unless an
  external display is connected.
- A LaunchAgent runs only while you're logged in. For a headless Mac, enable
  automatic login (System Settings > Users & Groups).
- The first time, macOS may ask whether Python may accept incoming network
  connections. Allow it, or the DVR box can't connect.

**Check it from another machine:**

```sh
curl http://<mac-ip>:8096/health
# {"ok": true, "backend": "MlxWhisper", "model": "mlx-community/whisper-large-v3-mlx"}
```

**Logs and control:**

```sh
tail -f ~/Library/Logs/captions-asr.log
launchctl kickstart -k gui/$(id -u)/com.captions-dvr.asr     # restart
launchctl bootout gui/$(id -u)/com.captions-dvr.asr          # stop and unload
```

## 2. Channels DVR box (dvr-host): relay

Requirements: Debian or Ubuntu with systemd. No ffmpeg is needed, because
PyAV bundles it. (ATSC 3.0 channels with AC-4 audio need an ffmpeg that can
decode AC-4, such as Channels DVR's own: set its path as "Audio ffmpeg" in
the web page. Debian's ffmpeg can't.)

As a Debian package (architecture-independent; installing downloads PyAV and
the other Python packages, so it needs internet): download
`captions-dvr_<version>_all.deb` from the GitHub release, then

```sh
sudo apt install ./captions-dvr_<version>_all.deb
```

or build it from a checkout with `deploy/build-deb.sh` (it lands in `dist/`).

Or from a checkout (`sudo apt install python3 python3-venv git`):

```sh
git clone <repository-url> ~/captions-dvr
cd ~/captions-dvr
sudo ./deploy/install.sh
```

Then edit `/etc/captions-dvr/captions-dvr.env`. These are the settings that
matter:

```sh
PORT=8095
PUBLIC_URL=http://dvr-host:8095      # what Channels uses; must match PORT
ASR_BACKEND=remote
ASR_URL=http://<mac-ip>:8096              # the Mac service from step 1
```

Sources and channels are picked in the web page at `http://dvr-host:8095/`
(stored in `/var/lib/captions-dvr/channels.json`). Apply changes to the env
file with:

```sh
sudo systemctl restart captions-dvr
journalctl -u captions-dvr -f
```

The service runs as user `captions-dvr` from `/opt/captions-dvr`, at `Nice=5`,
one process per channel being watched.
`sudo apt remove captions-dvr` keeps the settings and channels; `purge`
removes them too.
It uses about 10–15% of one J4125 core and 55 MB of RAM per channel being
watched, and nothing while idle.

### English guide (optional)

The relay can translate the programme guide (`epg.xml`) into English with
NLLB-200 on its own CPU (cached, so only new programmes cost time). Set
`EPG_TRANSLATE=1` in `/etc/captions-dvr/captions-dvr.env` and restart; the
first time it downloads the model (1.4 GB, CC-BY-NC) into
`/var/lib/captions-dvr/models` by itself. To fetch it beforehand:
`sudo captions-dvr-get-nllb`. Off by default; captions don't need it.

## 3. Channels DVR

In Settings > Sources > Add Source > Custom Channels:

- Name: `captions-dvr`
- Stream Format: **HLS**
- Source: URL, `http://dvr-host:8095/playlist.m3u`
- XMLTV Guide Data: `http://dvr-host:8095/epg.xml`
- Optionally set Prefer channel-number from M3U, so the channels keep the
  numbers you give them in the relay's web page.

Pick channels in the relay's web page (http://dvr-host:8095/, "Sources &
channels"); Channels picks them up on its next source refresh. In the Channels
player choose the **English subtitles** track. (CEA-608/708 CC embedding is
off by default; turn on "EIA CC" per channel if a client needs it. Channels
whose source has its own English captions pass those through as CC.) Subtitles start about 15 s after tuning, run
about 20–30 s behind the live source, and are kept in recordings.

Use only one captions-dvr source. (`/playlist-hls.m3u`, the older name, is
the same playlist.)

Or keep the channel list in Channels instead (stateless mode, see README):
Source URL `http://dvr-host:8095/api/m3u?url=<url-encoded M3U>`, Stream
Format **HLS**, and your own XMLTV guide. Add `&mt=seamless,nllb,whisper` to
get N.1, N.2 variants per backend.

## 4. Verify

```sh
curl http://<mac-ip>:8096/health                 # speech service up
curl http://dvr-host:8095/playlist.m3u  # enabled channels
curl -s http://dvr-host:8095/status         # per-channel state
```

While a channel is being watched, its entry in `/status` shows:

- `input: streaming`
- `lead_s` around 20–30
- an `asr_rtf` below 1 (the share of real time ASR needs)
- `captions` counting up and `recent` holding the latest lines
- in `dropped`, a growing `asr_error` count, which means the relay can't reach
  the Mac or the Mac is too slow (check `ASR_URL` and the Mac log)

If you don't have a player handy, a quick test from any machine with curl:
the newest subtitle segment of a channel (repeat until it has cues):

```sh
curl -s http://dvr-host:8095/hls/<id>/subs.m3u8 | tail -1 | xargs curl -s
```

## Docker (a Docker host)

`docker compose up -d --build` runs the relay with local CPU ASR on :8095.
See README.md.
