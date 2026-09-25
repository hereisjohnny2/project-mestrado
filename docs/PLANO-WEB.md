# Plano — Rock Segmentation Web

Versão web unificada das duas aplicações do mestrado: anotação de regiões de
interesse + treino de rede neural + segmentação em lote (poro × sólido).

---

## 1. Diagnóstico do que existe hoje

### 1.1 As duas aplicações

**App de anotação** — [`rock-image-annotation`](https://github.com/hereisjohnny2/rock-image-annotation) (C++ / Qt 6)

- `ImageDisplayWidget` carrega a imagem e deixa pintar com pincel sobre ela.
  Cada pixel pintado vai para um `QHash<QPoint, {QRgb, layerName}>` — a *layer*
  ativa é o rótulo (`Poro`, `Solido`, ...).
- `collectDataFromImage()` despeja esse hash numa `PixelDataTable`
  (PosX, PosY, R, G, B, Label) e limpa o buffer.
- `saveTableData()` grava apenas as colunas 2–5:
  `R \t G \t B \t Label \n` → arquivo `.dat`.

**App de rede neural** — `project-mestrado` (Python / PyTorch)

| Arquivo | Papel |
|---|---|
| `rock-nn/rock_model.py` | MLP `3 → 4 → 4 → 4 → 2`, ReLU nas 3 primeiras, `log_softmax` na saída |
| `rock-nn/utils/dataset.py` | lê o `.dat`, `Poro → 1`, resto → `0`; split 80/20; `DataLoader` batch 16 |
| `rock-nn/utils/network.py` | `nll_loss`, loop de épocas; teste por `argmax` |
| `rock-nn/trainer.py` | Adam `lr=0.0025`, salva `<dataset>-nn-model.pt` (só o `state_dict`) |
| `rock-nn/utils/image.py` | `binarize()` achata a imagem em `(N,3)` float **0–255 cru**, `argmax`, volta para `(H,W)`; porosidade e PNG binário |
| `rock-nn/tester.py` | carrega modelo, aplica, grava `porosity.txt` e `<nome>-bin.png` |
| `tools/*` | shell scripts para rodar em lote |

### 1.2 O pipeline atual, ponta a ponta

```
imagem ──[Qt: pinta camadas]──> coleta pixels ──> arquivo .dat (R\tG\tB\tLabel)
      └──[Python: -t dataset.dat -e N]──> <dataset>-nn-model.pt
              └──[Python: --save -i img -m model -o out]──> img-bin.png + porosity.txt
```

### 1.3 Pontos de atrito (o que a versão web resolve)

| # | Problema | Onde |
|---|---|---|
| 1 | Duas aplicações, dois ambientes (Qt/C++ e Python), transporte manual de `.dat` e `.pt` | — |
| 2 | O `.dat` é o único contrato: perde-se *qual imagem* e *quais regiões* geraram o dataset; não dá para reeditar a anotação | `saveTableData()` |
| 3 | O modelo salvo é só `state_dict` — sem métricas, sem hiperparâmetros, sem data, sem vínculo com o dataset | `trainer.py:29` |
| 4 | `calculate_porosity()` é um laço Python duplo sobre todos os pixels — ~1M de iterações interpretadas numa imagem 1000×1000 | `utils/image.py:32` |
| 5 | `binarize()` joga a imagem inteira numa única tensor de uma vez — estoura memória em imagem grande (e em GPU) | `utils/image.py:8` |
| 6 | README diz 70/30, o código faz 80/20 (`split_dataset(ratio=0.8)`) | `utils/dataset.py:18` |
| 7 | Sem *seed* → treinos não reproduzíveis | `trainer.py` |
| 8 | Acurácia global é a única métrica — com classes desbalanceadas (poro costuma ser minoria) ela esconde o comportamento do modelo | `utils/network.py:16` |
| 9 | Lote só via shell script, sem progresso, sem retomada | `tools/*` |
| 10 | Zoom quebrado no app de anotação (TODO aberto no README) | `rock-image-annotation` |

> Itens 4–8 são de **infraestrutura**, não do modelo: nenhum deles muda a
> matemática do treino. Ver §3.

---

## 2. Visão do produto

Uma única aplicação web, dois módulos, um fluxo contínuo:

**Módulo Treino**
`Projeto → upload de imagens → anotar poro/sólido com pincel e polígono →
gerar dataset → treinar → ver métricas → publicar modelo versionado`

**Módulo Segmentação**
`Escolher modelo → subir um lote de imagens → rodar → galeria com original ×
binarizada × overlay + tabela de porosidade → exportar CSV/ZIP`

Premissas (confirmadas em §9):
- uso local via `docker compose`, **um usuário**, sem login na v1;
- imagens de até ~4000×4000, lotes de até algumas centenas;
- GPU opcional — tudo tem que rodar em CPU.

---

## 3. Núcleo de ML — o que se mantém e o que muda

### 3.1 Congelado (paridade bit a bit com o código atual)

- `RockNetModel`: `3→4→4→4→2`, ReLU ×3, `log_softmax(dim=1)`.
- Perda `nll_loss`, otimizador `Adam(lr=0.0025)`, batch 16.
- Split 80% treino / 20% teste, aleatório.
- Rótulos: `Poro → 1`, `Solido → 0`.
- Entrada: RGB **cru, 0–255, sem normalização** (o treino foi feito assim; mudar
  isso invalidaria os modelos já treinados).
- Inferência: `argmax` por pixel; porosidade = fração de pixels com rótulo 1.
- Duas classes apenas: **poro** e **sólido**.

Esses valores viram *defaults* num `TrainingConfig`, não constantes soltas —
ficam configuráveis na tela, mas o botão "treinar" sem mexer em nada reproduz
exatamente o comportamento de hoje.

### 3.2 Alterado (só infraestrutura)

| Mudança | Antes | Depois |
|---|---|---|
| Porosidade | laço Python duplo | `float(arr.mean())` — vetorizado |
| Inferência | imagem inteira numa tensor | blocos de ~1M pixels, `torch.inference_mode()` |
| Reprodutibilidade | sem seed | seed no `TrainingConfig`, gravada no artefato |
| Métricas | acurácia global | + precisão/recall/F1 por classe, matriz de confusão, IoU, curva de perda por época |
| Artefato | `model.pt` | `model.pt` + `model.json` (hiperparâmetros, hash do dataset, métricas, data, versão) + export TorchScript (`model_serializer.py`) |
| Lote | shell script | job com fila, progresso e resultado persistido |

### 3.3 Critério de aceite: teste de paridade

Antes de qualquer tela, um teste automatizado que prova que o núcleo portado é
o mesmo código:

1. Pega um `.dat` e uma imagem conhecidos.
2. Treina pelo CLI atual e pelo backend novo com a **mesma seed**.
3. Compara os `state_dict` — devem bater dentro de `atol=1e-6`.
4. Aplica os dois modelos à mesma imagem: máscara binária idêntica e porosidade
   igual à do `calculate_porosity()` antigo até `1e-9`.

Esse teste fica no CI. É ele que garante o "por hora não precisa mudar como o
modelo é treinado".

---

## 4. Arquitetura

### 4.1 Stack

```
rock-segmentation-web/
├── backend/          FastAPI + SQLAlchemy + PyTorch
│   ├── app/
│   │   ├── api/          rotas
│   │   ├── core/         config, storage, jobs
│   │   ├── db/           models SQLAlchemy + migrações
│   │   ├── ml/           ← núcleo portado de rock-nn (congelado, §3.1)
│   │   │   ├── model.py        RockNetModel (idêntico)
│   │   │   ├── dataset.py      .dat + máscaras → tensores
│   │   │   ├── training.py     loop de treino + métricas
│   │   │   ├── inference.py    binarização em blocos + porosidade
│   │   │   └── artifacts.py    .pt + model.json + TorchScript
│   │   └── services/     annotation, dataset, training, segmentation
│   └── tests/            inclui o teste de paridade (§3.3)
├── frontend/         React + TypeScript + Vite
│   └── src/
│       ├── features/annotation/   canvas, pincel, camadas
│       ├── features/training/     config, progresso, métricas
│       └── features/segmentation/ lote, galeria, porosidade
├── legacy/           rock-nn atual, preservado para o teste de paridade
└── docker-compose.yml
```

**Por que FastAPI:** o núcleo é PyTorch, então o backend tem que ser Python de
qualquer jeito — reaproveita o código existente sem reescrever nada. Dá jobs em
background e streaming de progresso (SSE) de graça.

**Por que React + canvas:** a anotação é a parte pesada da UI (pincel, zoom,
undo, camadas). Canvas 2D nativo resolve, sem dependência de biblioteca de
anotação pesada.

### 4.2 Modelo de dados (SQLite, um arquivo, backup trivial)

```
Project    id, name, created_at
Image      id, project_id, filename, path, width, height, sha256
Annotation id, image_id, class (poro|solido), mask_path, updated_at
Dataset    id, project_id, path(.dat), n_pixels, n_poro, n_solido, sha256, created_at
Model      id, dataset_id, name, version, pt_path, json_path, metrics, config, created_at
Run        id, model_id, status, created_at, finished_at
Result     id, run_id, image_id, porosity, time_ms, bin_path, overlay_path
```

Arquivos em disco, fora do banco:

```
storage/{project}/images/    originais (+ conversão para RGB/PNG, como convert_image_to_rgb_format)
storage/{project}/masks/     máscaras de anotação, 1 PNG por imagem
storage/{project}/datasets/  .dat gerados
storage/{project}/models/    model.pt + model.json + model-scripted.pt
storage/{project}/runs/      saídas da segmentação
```

### 4.3 Formato de anotação — a mudança estrutural

Hoje a anotação vive como lista de pontos e morre no `.dat`. Na versão web ela
vira **uma máscara PNG por imagem**, paleta indexada:

```
0 = não anotado    1 = poro    2 = sólido
```

Ganhos: a anotação é reabrível e reeditável; o dataset é regenerável a qualquer
momento de forma determinística; undo/redo é natural; e a máscara é um artefato
de pesquisa por si só.

O `.dat` continua sendo gerado (mesma tabulação `R\tG\tB\tLabel`), tanto para o
treino quanto para compatibilidade com o CLI atual — o que mantém o teste de
paridade possível.

### 4.4 API

```
POST   /projects                          criar projeto
GET    /projects/{id}
POST   /projects/{id}/images              upload (multi-arquivo, converte p/ RGB)
GET    /images/{id}                       metadados + URL da imagem

GET    /images/{id}/mask                  máscara PNG atual
PUT    /images/{id}/mask                  salvar máscara (PNG indexado)

POST   /projects/{id}/datasets            gerar .dat a partir das máscaras
GET    /datasets/{id}                     estatísticas (n pixels por classe, balanço)
GET    /datasets/{id}/download            baixar o .dat (compat. CLI)

POST   /training/jobs                     {dataset_id, epochs, lr, batch, split, seed}
GET    /training/jobs/{id}/stream         SSE: época, perda, progresso
GET    /training/jobs/{id}                estado final + métricas
GET    /models                            modelos publicados
GET    /models/{id}/download              .pt / .json / TorchScript

POST   /segmentation/runs                 {model_id, image_ids[]}
GET    /segmentation/runs/{id}/stream     SSE: progresso por imagem
GET    /segmentation/runs/{id}/results    porosidade por imagem + tempos
GET    /segmentation/runs/{id}/export     CSV (equivale ao porosity.txt) ou ZIP
```

---

## 5. Telas

### 5.1 Anotação (o coração do módulo de treino)

Três `<canvas>` sobrepostos: imagem base, máscara (opacidade ajustável) e
cursor/preview.

- **Ferramentas:** pincel (tamanho 1–100), borracha, retângulo, polígono,
  balde por tolerância de cor (opcional, acelera muito a marcação).
- **Classes:** Poro / Sólido, atalhos `1` e `2`, cores configuráveis.
- **Zoom e pan** via matriz de transformação — corrige o TODO aberto do app Qt.
  `Espaço + arrastar` para pan, scroll para zoom, `0` para encaixar na tela.
- **Undo/redo** por snapshot da máscara (`Ctrl+Z` / `Ctrl+Shift+Z`).
- **Contador ao vivo:** pixels marcados por classe, com aviso de desbalanceamento.
- Salvamento automático da máscara (debounce ~2s) + indicador de estado.

### 5.2 Dataset e Treino

Lista de imagens anotadas com contagem por classe → "Gerar dataset" →
painel com histograma RGB por classe (mostra visualmente se as classes são
separáveis no espaço de cor — é exatamente o que o MLP enxerga).

Formulário de treino com os *defaults* de §3.1 travados num botão
"padrão da dissertação". Durante o treino: curva de perda ao vivo via SSE.
Ao final: acurácia, matriz de confusão, métricas por classe, botão "publicar
modelo" (dá nome e versão).

### 5.3 Segmentação

Escolher modelo (com as métricas dele à vista) → arrastar um lote de imagens →
rodar. Barra de progresso por imagem.

Resultado: galeria com *slider* comparando original × binarizada, overlay da
máscara predita sobre o original, tabela de porosidade ordenável e
exportação CSV (`nome, porosidade, tempo_ms` — mesmo conteúdo do `porosity.txt`)
ou ZIP com todos os PNGs binarizados.

---

## 6. Fases de entrega

Cada fase termina rodando e testável — nada de "só funciona no final".

### Fase 0 — Fundação e paridade *(a mais importante)*
- Esqueleto FastAPI + SQLite + camada de storage.
- Porte de `rock-nn` para `backend/app/ml/`, com as correções de §3.2.
- `legacy/` congelado com o código atual.
- **Teste de paridade de §3.3 passando no CI.**
- Docker compose subindo backend + frontend vazio.

*Valido assim:* `pytest` verde, incluindo o teste de paridade.

### Fase 1 — Anotação
- Projetos, upload de imagens (com conversão RGB), listagem.
- Canvas com pincel, borracha, classes, zoom/pan, undo/redo.
- Persistência da máscara PNG, autosave.

*Valido assim:* anoto uma imagem, recarrego a página, a anotação está intacta.

### Fase 2 — Dataset e Treino
- Geração do `.dat` a partir das máscaras + estatísticas e histogramas.
- Job de treino com SSE, métricas completas, publicação do modelo versionado.

*Valido assim:* dataset gerado na web + treino na web produz acurácia
equivalente à de um treino com o CLI antigo sobre o mesmo material.

### Fase 3 — Segmentação em lote
- Upload de lote, run assíncrono, progresso, resultados persistidos.
- Galeria comparativa, tabela de porosidade, exportação CSV/ZIP.

*Valido assim:* porosidade da web bate com a do `tester.py` para a mesma
imagem e o mesmo modelo.

### Fase 4 — Acabamento
- Importar `.dat` e `.pt` legados (aproveita tudo que já foi produzido no mestrado).
- Export TorchScript.
- README, docker-compose de produção, script de seed com imagens de exemplo.
- Comparação entre modelos lado a lado (útil para a continuidade da pesquisa).

---

## 7. Melhorias deixadas para depois (fora do escopo desta versão)

Registradas aqui para não se perderem, mas **não** entram agora — o pedido é
manter o modelo como está:

- Features além de RGB (vizinhança 3×3, textura, gradiente) — provavelmente o
  maior ganho de acurácia por esforço.
- CNN pequena (U-Net) em vez do MLP por pixel, que ignora contexto espacial.
- Pós-processamento morfológico (abertura/fechamento) para limpar ruído *salt
  and pepper* da classificação por pixel.
- Mais de duas classes (ex.: separar minerais na fase sólida).
- Active learning: o app sugere os pixels de maior incerteza para anotar.

---

## 8. Riscos

| Risco | Mitigação |
|---|---|
| Imagens muito grandes travam o canvas | renderizar em resolução reduzida, pintar na resolução nativa via matriz de transformação |
| Treino longo prende o worker | job em processo separado, com cancelamento |
| Dataset desbalanceado passa despercebido | aviso na tela com a proporção por classe + métricas por classe |
| Porte do ML introduz divergência silenciosa | teste de paridade no CI (§3.3) |
| Modelos antigos deixarem de carregar | `legacy/` preservado + importador na Fase 4 |

---

## 9. Decisões tomadas

| Questão | Decisão |
|---|---|
| **Repositório** | Este repo vira o monorepo da versão web. `rock-nn/` e `tools/` migram para `legacy/`, preservados para o teste de paridade (§3.3). |
| **Anotação** | Pincel, borracha, retângulo e polígono na v1. Balde por tolerância de cor fica para depois (risco de induzir viés na marcação). |
| **Usuários** | Monousuário, sem autenticação. O modelo de dados já tem `Project`, então login pode ser acrescentado depois sem refatorar. |
| **Deploy** | `docker compose up` local, CPU. O código mantém a detecção de GPU que já existe hoje (`torch.cuda.is_available()`), então um servidor com GPU funciona sem mudança. |
