# OER RAG 鈥?鐢靛寲瀛︽枃鐚绱㈤棶绛?
鍩轰簬 Chroma + BGE-M3 宓屽叆 + OpenAI 鍏煎 API 鐨?OER/鐢靛偓鍖栨枃鐚?RAG 绯荤粺锛孲treamlit Web UI銆?
## 椤圭洰缁撴瀯

```
Nature鑽墿鍙戠幇/          鈫?宸ヤ綔鍖烘牴鐩綍锛堟湰鍦帮級
鈹溾攢鈹€ md/                  鈫?鏂囩尞 Markdown锛堢储寮曡緭鍏ワ紝涓嶅叆 Git锛?鈹溾攢鈹€ OER/OER_md/          鈫?鍘熷璇枡锛堝彲鍚屾鎴栭摼鎺ュ埌 md/锛?鈹斺攢鈹€ oer_rag/             鈫?鏈粨搴擄紙搴旂敤浠ｇ爜锛?    鈹溾攢鈹€ app.py           鈫?Streamlit 涓荤晫闈?    鈹溾攢鈹€ build_index.py   鈫?鏋勫缓 / 澧為噺鏇存柊 Chroma 绱㈠紩
    鈹溾攢鈹€ config.py        鈫?閰嶇疆涓庣幆澧冨彉閲?    鈹溾攢鈹€ chroma_db/       鈫?鍚戦噺搴擄紙涓嶅叆 Git锛屾湇鍔″櫒鎸佷箙鍖栧嵎锛?    鈹斺攢鈹€ output/          鈫?绱㈠紩鎶ュ憡銆佸璇濇棩蹇楋紙涓嶅叆 Git锛?```

> **璺緞绾﹀畾**锛歚config.py` 榛樿浠?`oer_rag` 鐨勪笂涓€绾х洰褰曡鍙?`md/`銆傚彲閫氳繃 `OER_RAG_MD_DIR` 瑕嗙洊锛圖ocker / 鏈嶅姟鍣ㄦ帹鑽愶級銆?
## 渚濊禆

| 缁勪欢 | 璇存槑 |
|------|------|
| Python | 3.10+锛堟帹鑽?3.11锛汥ocker 闀滃儚宸插浐瀹?3.11锛?|
| ChromaDB | 鏈湴鎸佷箙鍖栧悜閲忓簱 |
| 宓屽叆 | **cloud**锛堟帹鑽愶級锛欱GE-M3 via API锛?*local**锛歐indows + CUDA + `D:\bge-m3-local` |
| LLM | OpenAI 鍏煎 API锛堥粯璁?`OPENAI_BASE_URL=https://www.dmxapi.cn/v1`锛?|

## 鏈湴寮€鍙戯紙Windows锛?
```powershell
cd oer_rag
copy .env.example .env
# 缂栬緫 .env锛氬～鍏?OPENAI_API_KEY銆丱ER_RAG_ADMIN_KEY

# 纭繚 ../md 涓嬫湁 .md 鏂囦欢锛堟垨璁剧疆 OER_RAG_MD_DIR锛?pip install -r requirements.txt

# 棣栨鎴栬鏂欐洿鏂板悗閲嶅缓绱㈠紩
$env:EMBED_BACKEND = "cloud"
python build_index.py

# 鍚姩 UI
streamlit run app.py
# 鎴栧弻鍑?start_rag.bat / 涓婄骇鐩綍 涓€閿惎鍔≧AG.bat
```

娴忚鍣ㄦ墦寮€ http://localhost:8501

## 鐜鍙橀噺

澶嶅埗 `.env.example` 涓?`.env`銆傚叧閿」锛?
| 鍙橀噺 | 蹇呭～ | 璇存槑 |
|------|------|------|
| `OPENAI_API_KEY` | 鏄?| Chat + Cloud 宓屽叆 API 瀵嗛挜 |
| `OPENAI_BASE_URL` | 鍚?| API 浠ｇ悊鍦板潃 |
| `OER_RAG_ADMIN_KEY` | 寤鸿 | 绠＄悊鍛樺鍑哄璇?Excel锛涚暀绌哄垯绂佺敤 |
| `EMBED_BACKEND` | 鍚?| `cloud`锛堟湇鍔″櫒榛樿锛夋垨 `local`锛堜粎 Windows GPU锛?|
| `OER_RAG_MD_DIR` | 鍚?| 鏂囩尞鐩綍锛岄粯璁?`../md` |
| `OER_RAG_CHROMA_DIR` | 鍚?| 鍚戦噺搴撶洰褰曪紝榛樿 `./chroma_db` |

瀹屾暣鍒楄〃瑙?[.env.example](.env.example)銆?
## 涓嶅簲鎻愪氦鍒?GitHub 鐨勫唴瀹?
- `.env`锛堝惈 API Key銆佺鐞嗗憳瀵嗛挜锛?- `chroma_db/`锛堝悜閲忕储寮曪紝浣撶Н澶э級
- `output/`锛堝璇濇棩蹇椼€佺储寮曟姤鍛婏級
- 鏂囩尞璇枡 `md/` 鎴?`OER/OER_md/`锛堟暟鐧?MB锛屽崟鐙悓姝ワ級
- 鏈湴 Python 鐜锛坄D:\bge-m3-local\pyenv` 绛夛級

## 鏈嶅姟鍣ㄩ儴缃诧紙Docker锛屾帹鑽愶級

### 1. 鍑嗗鏈嶅姟鍣?
- Linux锛圲buntu 22.04+ 绛夛級锛屽畨瑁?Docker + Docker Compose v2
- 寮€鏀剧鍙?8501锛堢洿杩烇級鎴?80锛坣ginx 鍙嶄唬锛?
### 2. 鍏嬮殕浠ｇ爜

```bash
git clone https://github.com/runzhi-wang/Redox-LLM.git
cd Redox-LLM
cp .env.example .env
nano .env   # 濉叆鐢熶骇瀵嗛挜
```

### 3. 涓婁紶璇枡涓庣储寮?
**璇枡**锛堜簩閫変竴锛夛細

```bash
# A. rsync 鏈湴 md 鐩綍
rsync -avz /path/to/md/ user@server:/opt/redox-llm/data/md/

# B. 浣跨敤鐜版湁 OER_md锛堥渶涓庣储寮曟椂璺緞涓€鑷存垨閲嶅缓绱㈠紩锛?rsync -avz "/path/to/OER/OER_md/" user@server:/opt/redox-llm/data/md/
```

鍦?`.env` 鎴?shell 涓缃細

```bash
export MD_DATA_DIR=/opt/redox-llm/data/md
```

**鍚戦噺绱㈠紩**锛堜簩閫変竴锛夛細

```bash
# A. 澶嶅埗宸叉湁 chroma_db锛堟渶蹇笂绾匡級
rsync -avz ./chroma_db/ user@server:/var/lib/docker/volumes/...  # 鎴栭娆?up 鍚?docker cp

# B. 鍦ㄦ湇鍔″櫒閲嶅缓锛堥渶 API 棰濆害锛岀害 209+ 绡囨枃鐚級
chmod +x deploy/scripts/rebuild-index.sh
./deploy/scripts/rebuild-index.sh
```

### 4. 鍚姩

```bash
docker compose up -d --build
# 鍙€夛細nginx 鍙嶄唬 + 鍩虹璁よ瘉
docker compose --profile proxy up -d --build
```

璁块棶锛歚http://SERVER_IP:8501` 鎴?`http://your.domain.com`

### 5. 鏇存柊浠ｇ爜

```bash
chmod +x deploy/scripts/deploy.sh
./deploy/scripts/deploy.sh   # git pull + docker compose up -d --build
```

## 宓屽叆妯″瀷绛栫暐

| 鍦烘櫙 | 鎺ㄨ崘 |
|------|------|
| Linux 浜戞湇鍔″櫒 | `EMBED_BACKEND=cloud`锛屾棤闇€ GPU |
| Windows 寮€鍙戞満 + RTX | 鍙€?`local`锛岄渶瀹夎 `D:\bge-m3-local` |
| 绱㈠紩涓庢煡璇竴鑷存€?| **鍚屼竴 backend + 鍚屼竴 `OER_RAG_INDEX_VERSION`**锛涘垏鎹?backend 椤婚噸寤虹储寮?|

## 鍥㈤槦鍗忎綔

### 璁块棶鏂瑰紡

- **URL**锛氬洟闃熷叡浜湇鍔″櫒鍦板潃锛堝缓璁?nginx + HTTPS + 鍙€?HTTP Basic Auth锛?- **搴旂敤鍐呰璇?*锛氫粎绠＄悊鍛樺鍑哄姛鑳介渶 `OER_RAG_ADMIN_KEY`锛涙櫘閫氶棶绛旀棤鐧诲綍
- **API Key**锛氬缓璁湇鍔″櫒 `.env` 浣跨敤**鍥㈤槦鍏变韩 Key**锛涜嫢闇€鎸変汉璁¤垂鍙悗缁墿灞?
### 绱㈠紩缁存姢

璇枡鏂板 / 鍒嗗潡绛栫暐鍙樻洿锛坄OER_RAG_INDEX_VERSION`锛夊悗锛?
```bash
./deploy/scripts/rebuild-index.sh
docker compose restart oer-rag
```

### 瀵硅瘽鏃ュ織

- 杩愯鏃跺啓鍏?`output/chat_logs.jsonl`锛圖ocker 涓负鎸佷箙鍗?`chat_output`锛?- 绠＄悊鍛樺湪 UI銆岃缃?鈫?绠＄悊鍛樺瘑閽ャ€嶇櫥褰曞悗鍙鍑?Excel

## 鏃?Docker 閮ㄧ讲锛堝閫夛級

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export $(grep -v '^#' .env | xargs)
export EMBED_BACKEND=cloud
export OER_RAG_MD_DIR=/data/md
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

鍙敤 systemd 鎵樼锛涘弬鑰?`deploy/nginx/oer-rag.conf` 鍋氬弽浠ｃ€?
## CI

GitHub Actions锛坄.github/workflows/ci.yml`锛夊湪 push/PR 鏃惰繍琛?ruff 涓?import 鍐掔儫娴嬭瘯锛?*涓嶉儴缃层€佷笉涓婁紶瀵嗛挜**銆?
## 鏁呴殰鎺掓煡

| 鐜拌薄 | 澶勭悊 |
|------|------|
| 銆岀煡璇嗗簱鏈氨缁€?| 妫€鏌?`chroma_db` 鍗锋槸鍚︽寕杞斤紱杩愯 `rebuild-index.sh` |
| `OPENAI_API_KEY not set` | 纭 `.env` 瀛樺湪涓?`docker compose` 浣跨敤 `env_file` |
| 绱㈠紩鎵句笉鍒版枃鐚?| 纭 `MD_DATA_DIR` / `OER_RAG_MD_DIR` 鎸囧悜鍚?`.md` 鐨勭洰褰?|
| 宓屽叆瓒呮椂 | 璋冨ぇ `OER_RAG_EMBED_API_TIMEOUT_SEC` 鎴栭檷浣庡苟琛屽害 |

## 璁稿彲璇?
鍐呴儴鍥㈤槦浣跨敤锛涜鏂欑増鏉冨綊鍘熷嚭鐗堟柟銆?
