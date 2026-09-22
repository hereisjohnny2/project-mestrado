# Rock Segmentation — versão web

Versão web do projeto de mestrado: anotação de regiões de interesse (poro ×
sólido), treino de uma rede neural sobre essas anotações e aplicação em
lote sobre outros bancos de imagens.

O plano completo de desenvolvimento está em [`docs/PLANO-WEB.md`](docs/PLANO-WEB.md).

## Estrutura

```
.
├── backend/    FastAPI + SQLite + PyTorch (núcleo de ML portado de legacy/)
├── frontend/   React + TypeScript + Vite
├── legacy/     as duas aplicações originais do mestrado (congeladas)
└── docs/       plano de desenvolvimento
```

## `legacy/`

O código original do mestrado, preservado sem alterações:

- `legacy/rock-nn/` — CLI Python/PyTorch de treino e aplicação do modelo
  (ver `legacy/README.md` para instruções de uso).
- `legacy/tools/` — scripts shell para rodar treino/aplicação em lote.

O app de anotação em C++/Qt que gerava os arquivos `.dat` vive em seu
próprio repositório: [`rock-image-annotation`](https://github.com/hereisjohnny2/rock-image-annotation).

Esse código é a referência de comportamento: `backend/app/ml/` é um porte
dele, e um teste de paridade (`backend/tests/test_parity.py`) garante que o
treino e a inferência produzem resultados numericamente idênticos aos do
CLI original — ver plano §3.3.

## Rodando com Docker

```shell
docker compose up --build
```

- Backend (FastAPI): http://localhost:8000 — `/health` para checar o status.
- Frontend: http://localhost:5173

## Desenvolvimento local

### Backend

```shell
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

> `requirements.txt` pede a build CPU do PyTorch (`torch==2.3.1+cpu`) via
> `download.pytorch.org`. Se sua rede/proxy bloquear esse host, instale a
> build padrão do PyPI em vez dela — funciona igual em CPU, só ocupa mais
> espaço em disco por trazer bibliotecas CUDA não usadas:
> `pip install torch==2.3.1 -r <(grep -v torch requirements.txt)`.

Rodar os testes (inclui o teste de paridade contra `legacy/rock-nn`):

```shell
cd backend
pytest -v
```

### Frontend

```shell
cd frontend
npm install
npm run dev
```

## Estado atual

Fase 0 do plano: fundação do backend, porte do núcleo de ML com teste de
paridade, e esqueleto do frontend. As telas de anotação, treino e
segmentação ainda não existem — ver `docs/PLANO-WEB.md` para as próximas
fases.
