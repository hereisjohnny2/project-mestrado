#!/usr/bin/env bash
# Atualiza o servidor para um commit: código (git) + imagens backend/frontend do CI (GHCR).
#
# Chamado pelo GitHub Actions (.github/workflows/deploy.yml), que manda este arquivo por SSH no stdin:
#   ssh vps "bash -s -- <sha> <imagem-base>" < deploy/deploy.sh
# (assim roda sempre a versão do script do commit que está sendo publicado, não a que está no servidor).
# Também dá para rodar à mão no servidor (rollback, por exemplo):
#   ./deploy/deploy.sh <sha> [ghcr.io/hereisjohnny2/project-mestrado]
#
# Variáveis opcionais: APP_DIR (padrão /opt/project-mestrado); GHCR_USER/GHCR_TOKEN para baixar as
# imagens se o pacote no GHCR for privado (o CI passa o GITHUB_TOKEN do job, que expira quando o job termina).
set -euo pipefail

main() {
  local sha="${1:?uso: deploy.sh <sha> [imagem-base]}"
  local image_base="${2:-ghcr.io/hereisjohnny2/project-mestrado}"

  cd "${APP_DIR:-/opt/project-mestrado}"
  local dc="docker compose --env-file .env.prod -f docker-compose.vps.yml"

  [ -f .env.prod ]         || die "falta .env.prod (passo de configuração do deploy/DEPLOY.md)"
  # Só conteúdo conta: um `chmod +x` feito no servidor não é alteração (e não pode travar o checkout).
  git config core.fileMode false
  if ! git diff --quiet HEAD --; then
    git status --short --untracked-files=no >&2
    die "há alterações locais nos arquivos acima; o servidor deve espelhar o repositório (desfaça com: git checkout -- <arquivo>)"
  fi

  echo "==> imagens ${image_base}-{backend,frontend}:${sha}"
  if [ -n "${GHCR_TOKEN:-}" ]; then
    docker login ghcr.io -u "${GHCR_USER:-github-actions}" --password-stdin <<<"$GHCR_TOKEN" >/dev/null
  fi
  local pulled=0
  docker pull -q "${image_base}-backend:${sha}" && docker pull -q "${image_base}-frontend:${sha}" && pulled=1
  [ -n "${GHCR_TOKEN:-}" ] && docker logout ghcr.io >/dev/null
  [ "$pulled" = 1 ] || die "não consegui baixar as imagens de ${sha}"

  echo "==> código em ${sha}"
  git fetch -q origin "$sha"
  git checkout -q -B main "$sha"

  echo "==> subindo"
  local previous_backend previous_frontend
  previous_backend="$(docker image inspect -f '{{.Id}}' rockseg-backend:current 2>/dev/null || true)"
  previous_frontend="$(docker image inspect -f '{{.Id}}' rockseg-frontend:current 2>/dev/null || true)"
  docker tag "${image_base}-backend:${sha}" rockseg-backend:current
  docker tag "${image_base}-frontend:${sha}" rockseg-frontend:current
  docker rmi -f "${image_base}-backend:${sha}" "${image_base}-frontend:${sha}" >/dev/null
  $dc up -d --remove-orphans

  wait_healthy "$dc" backend 180 || {
    $dc logs --tail 80 backend
    if [ -n "$previous_backend" ]; then
      echo "==> backend não ficou saudável: voltando para as imagens anteriores (o código fica em ${sha})" >&2
      docker tag "$previous_backend" rockseg-backend:current
      [ -n "$previous_frontend" ] && docker tag "$previous_frontend" rockseg-frontend:current
      $dc up -d
    fi
    die "deploy de ${sha} falhou"
  }

  docker image prune -f >/dev/null     # remove as imagens :current anteriores, que ficaram sem etiqueta
  echo "==> ok: ${sha} no ar"
}

wait_healthy() {   # <comando compose> <serviço> <segundos>
  local dc="$1" svc="$2" deadline=$((SECONDS + $3)) id status
  id="$($dc ps -q "$svc")"
  while [ "$SECONDS" -lt "$deadline" ]; do
    status="$(docker inspect -f '{{.State.Health.Status}}' "$id" 2>/dev/null || echo missing)"
    [ "$status" = healthy ] && return 0
    sleep 5
  done
  echo "backend ainda '${status}' após $3 s" >&2
  return 1
}

die() { echo "ERRO: $*" >&2; exit 1; }

# Tudo dentro de main e sem stdin: quando o script chega por "bash -s", nenhum comando pode ler o resto dele.
main "$@" </dev/null
