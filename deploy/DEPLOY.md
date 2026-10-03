# Deploy na VPS (compartilhada com o honda-rag)

Stack: **Caddy** (proxy compartilhado da VPS, já existe — repositório `honda-rag`, pasta `deploy/edge/`)
→ **frontend** (nginx, serve o build e faz proxy de `/api` pro backend) → **backend** (FastAPI + SQLite
+ PyTorch CPU-only).

Essa app entra na mesma VPS do honda-rag, sem porta própria: o Caddy compartilhado já tem o site dela em
`deploy/edge/sites/rockseg.caddy` (repo honda-rag). Se o honda-rag ainda não estiver na VPS, veja o
`deploy/DEPLOY.md` dele primeiro — o proxy e a rede `edge` precisam existir antes do passo 5 aqui.

> **Atenção de recursos:** a VPS é um KVM 1 (4 GB de RAM) e já roda o honda-rag (app + Postgres + Ollama,
> ~3,4 GB somados). Com o backend deste app (torch CPU-only) a memória fica apertada — acompanhe
> `docker stats`/`free -h` depois do primeiro deploy e considere subir pro KVM 2 (8 GB) se faltar.

## 1. DNS

Crie um registro **A** do subdomínio apontando para o mesmo IP da VPS do honda-rag:
```bash
dig +short rockseg.seudominio.com    # deve mostrar o IP da VPS
```

## 2. Código no servidor

```bash
sudo mkdir -p /opt/project-mestrado && sudo chown $USER /opt/project-mestrado
git clone https://github.com/hereisjohnny2/project-mestrado.git /opt/project-mestrado && cd /opt/project-mestrado
```
Clone por **HTTPS** (o repositório é público): o deploy automático faz `git fetch` no servidor e assim não
precisa de chave do GitHub lá.

## 3. Configuração

```bash
cd /opt/project-mestrado
cp .env.prod.example .env.prod && chmod 600 .env.prod
nano .env.prod          # ROCKSEG_DOMAIN, ROCKSEG_SECRET_KEY (valor longo e aleatório)
```

## 4. Site no proxy compartilhado

No repositório **honda-rag** (não neste), em `/opt/honda-rag/deploy/edge/.env.edge`, confirme que
`ROCKSEG_DOMAIN` está com o mesmo valor do passo 3. O arquivo de site (`sites/rockseg.caddy`) já vem do
repositório — só falta a variável e um reload:
```bash
cd /opt/honda-rag/deploy/edge
nano .env.edge                                                        # ROCKSEG_DOMAIN=rockseg.seudominio.com
docker compose --env-file .env.edge exec -T caddy caddy reload --config /etc/caddy/Caddyfile
```

## 5. Subir

```bash
cd /opt/project-mestrado
docker compose --env-file .env.prod -f docker-compose.vps.yml up -d --build
```
A primeira build do backend baixa o torch CPU-only (~200 MB) e pode demorar alguns minutos.

## 6. Criar o primeiro usuário

Não há cadastro pela interface — só por aqui, direto no servidor (senha pedida sem eco):
```bash
docker compose --env-file .env.prod -f docker-compose.vps.yml exec backend python -m app.create_user pessoa@example.com --name "Pessoa"
```

## 7. Verificar

```bash
docker compose --env-file .env.prod -f docker-compose.vps.yml ps
curl -I https://rockseg.seudominio.com    # deve dar 200
```

## 8. Rotina

| Tarefa | Comando |
|---|---|
| Atualizar o código | automático a cada push na `main` (seção 11); à mão: `./deploy/deploy.sh <sha>` |
| Voltar uma versão | *Actions > Deploy > Run workflow* com o SHA anterior, ou `./deploy/deploy.sh <sha>` |
| Novo usuário | repita o passo 6 |

## 9. Deploy automático (GitHub Actions)

A cada push na `main` (exceto mudanças só em `.md` e `legacy/`), o workflow
[`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml):

1. constrói as imagens `backend` e `frontend` no runner do GitHub e publica em
   `ghcr.io/hereisjohnny2/project-mestrado-{backend,frontend}:<sha>` (a VPS não compila nada);
2. entra na VPS por SSH e roda [`deploy/deploy.sh`](deploy.sh) do próprio commit: baixa as imagens, faz
   `git checkout` do mesmo SHA e `up -d` com as imagens marcadas como `rockseg-{backend,frontend}:current`;
3. espera o backend ficar *healthy* (até 3 min). Se não ficar, volta para as imagens anteriores e o job falha.

O que ele **não** faz: criar usuários (passo 6) nem mexer no `.env.prod`, que continua só no servidor.
Faça o **primeiro deploy à mão** (passos 1–7) e só depois ligue o automático.

### Configuração (uma vez) — reaproveitando as credenciais do honda-rag

Esta VPS já tem um usuário de deploy e uma chave SSH do CI cadastrados (ver `deploy/DEPLOY.md` do
honda-rag, seção 11). O GitHub não compartilha *secrets* entre repositórios, então eles precisam ser
cadastrados de novo aqui, com o **mesmo valor**:

1. No GitHub deste repositório, em *Settings > Environments*, crie o ambiente **`production`** e nele
   cadastre:

   | Tipo | Nome | Valor |
   |---|---|---|
   | Secret | `VPS_SSH_KEY` | o mesmo conteúdo de `deploy_key` já usado no honda-rag |
   | Secret | `VPS_KNOWN_HOSTS` | o mesmo conteúdo de `known_hosts` já usado no honda-rag |
   | Variable | `VPS_HOST` | o mesmo IP (ou nome) da VPS |
   | Variable | `VPS_USER` | o mesmo usuário de deploy |
   | Variable | `VPS_PORT` | opcional, padrão `22` |
   | Variable | `VPS_APP_DIR` | opcional, padrão `/opt/project-mestrado` |

   Opcional: em *Required reviewers*, adicione você mesmo para cada deploy esperar um clique; em
   *Deployment branches*, restrinja a `main`.
2. A imagem no GHCR sai privada; o CI entrega à VPS um token que só vale durante o job. Para rodar
   `./deploy/deploy.sh <sha>` à mão no servidor, torne os pacotes públicos (*Packages >
   project-mestrado-backend/frontend > Package settings*) ou faça `docker login ghcr.io` com um token
   `read:packages`.

**Não edite arquivos versionados no servidor:** o `deploy.sh` para com erro se houver alterações locais.
Configuração do servidor fica no `.env.prod`, que o Git ignora.

## Segurança (resumo)

- Nenhuma porta é publicada por este compose; quem publica 80/443 é só o Caddy compartilhado
  (`deploy/edge/` no repo honda-rag).
- Sem cadastro público: contas só são criadas no servidor (passo 6). `ROCKSEG_COOKIE_SECURE=true` exige
  HTTPS para o cookie de sessão, o que já é o caso atrás do Caddy.
- A chave SSH do CI é a mesma do honda-rag; se vazar, remova a linha dela do
  `~/.ssh/authorized_keys` da VPS (afeta os dois repositórios).
- `.env.prod` fica só no servidor (`chmod 600`) e está no `.gitignore`.
