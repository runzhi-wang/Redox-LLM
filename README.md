# OER RAG — 电化学文献检索问答

基于 Chroma + BGE-M3 嵌入 + OpenAI 兼容 API 的 OER/电催化文献 RAG 系统，Streamlit Web UI。

## 项目结构

```
Nature药物发现/          ← 工作区根目录（本地）
├── md/                  ← 文献 Markdown（索引输入，不入 Git）
├── OER/OER_md/          ← 原始语料（可同步或链接到 md/）
└── oer_rag/             ← 本仓库（应用代码）
    ├── app.py           ← Streamlit 主界面
    ├── build_index.py   ← 构建 / 增量更新 Chroma 索引
    ├── config.py        ← 配置与环境变量
    ├── chroma_db/       ← 向量库（不入 Git，服务器持久化卷）
    └── output/          ← 索引报告、对话日志（不入 Git）
```

> **路径约定**：`config.py` 默认从 `oer_rag` 的上一级目录读取 `md/`。可通过 `OER_RAG_MD_DIR` 覆盖（Docker / 服务器推荐）。

## 依赖

| 组件 | 说明 |
|------|------|
| Python | 3.10+（推荐 3.11；Docker 镜像已固定 3.11） |
| ChromaDB | 本地持久化向量库 |
| 嵌入 | **cloud**（推荐）：BGE-M3 via API；**local**：Windows + CUDA + `D:\bge-m3-local` |
| LLM | OpenAI 兼容 API（默认 `OPENAI_BASE_URL=https://www.dmxapi.cn/v1`） |

## 本地开发（Windows）

```powershell
cd oer_rag
copy .env.example .env
# 编辑 .env：填入 OPENAI_API_KEY、OER_RAG_ADMIN_KEY

# 确保 ../md 下有 .md 文件（或设置 OER_RAG_MD_DIR）
pip install -r requirements.txt

# 首次或语料更新后重建索引
$env:EMBED_BACKEND = "cloud"
python build_index.py

# 启动 UI
streamlit run app.py
# 或双击 start_rag.bat / 上级目录 一键启动RAG.bat
```

浏览器打开 http://localhost:8501

## 环境变量

复制 `.env.example` 为 `.env`。关键项：

| 变量 | 必填 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | 是 | Chat + Cloud 嵌入 API 密钥 |
| `OPENAI_BASE_URL` | 否 | API 代理地址 |
| `OER_RAG_ADMIN_KEY` | 建议 | 管理员导出对话 Excel；留空则禁用 |
| `EMBED_BACKEND` | 否 | `cloud`（服务器默认）或 `local`（仅 Windows GPU） |
| `OER_RAG_MD_DIR` | 否 | 文献目录，默认 `../md` |
| `OER_RAG_CHROMA_DIR` | 否 | 向量库目录，默认 `./chroma_db` |

完整列表见 [.env.example](.env.example)。

## 不应提交到 GitHub 的内容

- `.env`（含 API Key、管理员密钥）
- `chroma_db/`（向量索引，体积大）
- `output/`（对话日志、索引报告）
- 文献语料 `md/` 或 `OER/OER_md/`（数百 MB，单独同步）
- 本地 Python 环境（`D:\bge-m3-local\pyenv` 等）

## 服务器部署（Docker，推荐）

### 1. 准备服务器

- Linux（Ubuntu 22.04+ 等），安装 Docker + Docker Compose v2
- 开放端口 8501（直连）或 80（nginx 反代）

### 2. 克隆代码

```bash
git clone https://github.com/runzhi-wang/Redox-LLM.git
cd Redox-LLM
cp .env.example .env
nano .env   # 填入生产密钥
```

### 3. 上传语料与索引

**语料**（二选一）：

```bash
# A. rsync 本地 md 目录
rsync -avz /path/to/md/ user@server:/opt/redox-llm/data/md/

# B. 使用现有 OER_md（需与索引时路径一致或重建索引）
rsync -avz "/path/to/OER/OER_md/" user@server:/opt/redox-llm/data/md/
```

在 `.env` 或 shell 中设置：

```bash
export MD_DATA_DIR=/opt/redox-llm/data/md
```

**向量索引**（二选一）：

```bash
# A. 复制已有 chroma_db（最快上线）
rsync -avz ./chroma_db/ user@server:/var/lib/docker/volumes/...  # 或首次 up 后 docker cp

# B. 在服务器重建（需 API 额度，约 400+ 篇文献）
chmod +x deploy/scripts/rebuild-index.sh
./deploy/scripts/rebuild-index.sh
```

### 4. 启动

```bash
docker compose up -d --build
# 可选：nginx 反代 + 基础认证
docker compose --profile proxy up -d --build
```

访问：`http://SERVER_IP:8501` 或 `http://your.domain.com`

### 5. 更新代码

```bash
chmod +x deploy/scripts/deploy.sh
./deploy/scripts/deploy.sh   # git pull + docker compose up -d --build
```

## 嵌入模型策略

| 场景 | 推荐 |
|------|------|
| Linux 云服务器 | `EMBED_BACKEND=cloud`，无需 GPU |
| Windows 开发机 + RTX | 可选 `local`，需安装 `D:\bge-m3-local` |
| 索引与查询一致性 | **同一 backend + 同一 `OER_RAG_INDEX_VERSION`**；切换 backend 须重建索引 |

## 团队协作

### 访问方式

- **URL**：团队共享服务器地址（建议 nginx + HTTPS + 可选 HTTP Basic Auth）
- **应用内认证**：仅管理员导出功能需 `OER_RAG_ADMIN_KEY`；普通问答无登录
- **API Key**：建议服务器 `.env` 使用**团队共享 Key**；若需按人计费可后续扩展

### 索引维护

语料新增 / 分块策略变更（`OER_RAG_INDEX_VERSION`）后：

```bash
./deploy/scripts/rebuild-index.sh
docker compose restart oer-rag
```

### 对话日志

- 运行时写入 `output/chat_logs.jsonl`（Docker 中为持久卷 `chat_output`）
- 管理员在 UI「设置 → 管理员密钥」登录后可导出 Excel

## 无 Docker 部署（备选）

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export $(grep -v '^#' .env | xargs)
export EMBED_BACKEND=cloud
export OER_RAG_MD_DIR=/data/md
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

可用 systemd 托管；参考 `deploy/nginx/oer-rag.conf` 做反代。

## CI

GitHub Actions（`.github/workflows/ci.yml`）在 push/PR 时运行 ruff 与 import 冒烟测试，**不部署、不上传密钥**。

## 故障排查

| 现象 | 处理 |
|------|------|
| 「知识库未就绪」 | 检查 `chroma_db` 卷是否挂载；运行 `rebuild-index.sh` |
| `OPENAI_API_KEY not set` | 确认 `.env` 存在且 `docker compose` 使用 `env_file` |
| 索引找不到文献 | 确认 `MD_DATA_DIR` / `OER_RAG_MD_DIR` 指向含 `.md` 的目录 |
| 嵌入超时 | 调大 `OER_RAG_EMBED_API_TIMEOUT_SEC` 或降低并行度 |

## 许可

内部团队使用；语料版权归原出版方。
