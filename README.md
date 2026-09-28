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

Desenvolvimento (código do backend montado por bind-mount):

```shell
docker compose up --build
```

- Backend (FastAPI): http://localhost:8000 — `/health` para checar o status.
- Frontend: http://localhost:5173

Produção local (reinício automático, só o frontend exposto, porta em
`ROCKSEG_PORT`, padrão 8080; dados no volume `storage`):

```shell
docker compose -f docker-compose.prod.yml up -d --build
```

### Dados de exemplo

Cria um projeto com imagens sintéticas, anotações, dataset e um modelo já
treinado (as imagens são geradas, não são seções delgadas reais):

```shell
docker compose exec backend python -m app.seed
docker compose -f docker-compose.prod.yml exec backend python -m app.seed
# ou, com o backend local rodando: cd backend && python -m app.seed
```

## Migrando o que já existe do mestrado

Na tela **Dataset e treino** de um projeto:

- **Importar .dat** — arquivo `R<TAB>G<TAB>B<TAB>Rótulo` gerado pelo app Qt (`Poro`
  vira poro; qualquer outro rótulo, sólido). Linhas malformadas são rejeitadas
  com o número da linha.
- **Importar .pt** — `state_dict` salvo pelo `trainer.py` legado. O modelo é
  avaliado no dataset selecionado (o formato antigo não guarda métricas e o
  split original é desconhecido, então os números incluem pixels de treino)
  e ganha `.json` e export TorchScript como qualquer outro. Modelos já
  TorchScript (`-scripted.pt`) não são aceitos.
- **Comparar modelos** — métricas, curvas de perda e porosidade/máscaras de duas
  segmentações lado a lado.

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

Rodar os testes (inclui o teste de paridade contra `legacy/rock-nn`, que
por sua vez precisa do `matplotlib` — só para importar `legacy/rock-nn`,
não é dependência do backend novo):

```shell
cd backend
pip install -r requirements-test.txt
pytest -v
```

### Frontend

```shell
cd frontend
npm install
npm run dev
```

## Estado atual

Fases 0–4 do plano concluídas: núcleo de ML com teste de paridade, anotação,
dataset e treino, segmentação em lote, importação de artefatos legados,
comparação de modelos, export TorchScript, compose de produção e seed. Ver
`docs/PLANO-WEB.md` (§7 lista melhorias deixadas para depois).
