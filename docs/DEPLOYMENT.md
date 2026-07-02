# 部署流程

当前 P0 有两种部署方式：

- Vercel demo：`frontend/` 静态构建 + `api/index.py` Python 函数；
- 单机部署：FastAPI + SQLite + Nginx + systemd。

生产环境不建议继续使用 SQLite 和 Vercel `/tmp` 数据库作为长期存储。

## A. Vercel demo 部署

### 1. 导入项目

在 Vercel 中 Import Git Repository。

Application Preset 选择：

```text
Other
```

当前仓库根目录已经提供：

- `package.json`：根构建命令；
- `frontend/package.json`：Vue 前端；
- `api/index.py`：FastAPI Vercel 入口；
- `vercel.json`：构建输出与 rewrite。

### 2. Build 配置

保持仓库默认配置：

```text
Build Command: npm run build
Output Directory: frontend/dist
Install Command: 默认
```

`vercel.json` 已配置：

- `/api/*` 转发到 `api/index.py`；
- 其他路径返回前端 `index.html`。

### 3. 环境变量

Vercel Project Settings -> Environment Variables：

```bash
APP_SECRET=replace-with-a-random-long-string
CONSENT_VERSION=p0-placeholder
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_TIMEOUT_SECONDS=30
```

如果没有 `DEEPSEEK_API_KEY`，服务仍会运行，但前端状态栏会显示：

```text
fallback
```

配置 DeepSeek 后重新部署，状态栏应显示：

```text
deepseek
```

如果模型返回非法 JSON，系统会 fallback，并显示：

```text
deepseek / json-failed
```

### 4. Vercel 数据限制

Vercel demo 默认数据库路径是：

```text
/tmp/product_app.sqlite3
```

这只适合 demo。函数实例重启、扩缩容或冷启动后，数据可能丢失或不一致。生产应迁移到外部数据库。

## B. 单机服务器部署

## 1. 准备服务器

推荐：

- Ubuntu 22.04 / 24.04；
- Python 3.10+；
- 2 核 CPU、4GB 内存起步；
- 如只调用 DeepSeek API，不需要 GPU。

安装系统依赖：

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nginx git
```

## 2. 拉取代码

```bash
cd /opt
sudo git clone <your-repo-url> depression-system
sudo chown -R $USER:$USER /opt/depression-system
cd /opt/depression-system
```

## 3. 创建 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r src/product_app/requirements.txt
```

如果需要 RAG 或训练依赖，再分别安装对应 requirements。P0 产品服务不默认需要训练依赖。

## 4. 配置环境变量

创建 `/opt/depression-system/.env.product`：

```bash
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
PRODUCT_DB_PATH=/opt/depression-system/data/product_app.sqlite3
CONSENT_VERSION=p0-placeholder
APP_SECRET=replace-with-a-random-long-string
HOST=127.0.0.1
PORT=8000
```

如果暂时没有 DeepSeek key，服务仍能启动，但会使用 fallback 回复。

## 5. 本地启动验证

```bash
cd /opt/depression-system
source .venv/bin/activate
set -a
source .env.product
set +a
PYTHONPATH=src uvicorn product_app.main:app --host 127.0.0.1 --port 8000
```

访问：

```text
http://服务器IP:8000
```

如果要使用当前 Vue 前端，而不是旧的 FastAPI 静态页，建议同时构建 `frontend/` 并由 Nginx 托管静态文件，再把 `/api/` 代理到 FastAPI。

构建前端：

```bash
npm run build
```

前端产物：

```text
frontend/dist
```

## 6. systemd 服务

创建 `/etc/systemd/system/depression-product.service`：

```ini
[Unit]
Description=Depression Product P0
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/depression-system
EnvironmentFile=/opt/depression-system/.env.product
Environment=PYTHONPATH=/opt/depression-system/src
ExecStart=/opt/depression-system/.venv/bin/uvicorn product_app.main:app --host ${HOST} --port ${PORT}
Restart=always
RestartSec=3
User=www-data
Group=www-data

[Install]
WantedBy=multi-user.target
```

调整目录权限：

```bash
sudo chown -R www-data:www-data /opt/depression-system/data
```

启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable depression-product
sudo systemctl start depression-product
sudo systemctl status depression-product
```

查看日志：

```bash
sudo journalctl -u depression-product -f
```

## 7. Nginx 反向代理

推荐配置：前端静态文件由 Nginx 托管，API 代理到 FastAPI。

创建 `/etc/nginx/sites-available/depression-product`：

```nginx
server {
    listen 80;
    server_name your.domain.com;

    client_max_body_size 2m;

    location / {
        root /opt/depression-system/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

启用：

```bash
sudo ln -s /etc/nginx/sites-available/depression-product /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 8. HTTPS

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your.domain.com
```

## 9. 上线前检查

```bash
cd /opt/depression-system
source .venv/bin/activate
PYTHONPATH=src python scripts/p0_adversarial_audit.py
```

检查项：

- 注册、登录、同意流程；
- `/api/chat` 未同意时返回 403；
- high 风险关键词触发；
- DeepSeek JSON 校验；
- Nginx HTTPS 可访问；
- `data/product_app.sqlite3` 有备份策略。
