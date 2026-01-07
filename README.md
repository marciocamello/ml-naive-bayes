# Projeto: Classificação — Naive Bayes (Obesidade)

Este repositório contém um projeto prático de classificação para predição de obesidade usando Naive Bayes. O objetivo é documentar o fluxo completo: EDA, pré-processamento, seleção de features, treinamento, avaliação, persistência do modelo e uma API para inferência.

**Conteúdo**

- `modelo-obesidade.ipynb` — notebook com EDA, pré-processamento, seleção de features, treino (GaussianNB), tuning (Optuna) e salvamento do modelo.
- `api_modelo_obesidade.py` — pequena API em Flask que carrega `modelo_obesidade.pkl` e expõe o endpoint `/predict` (POST) para inferência em tempo real.
- `Pipfile` — gerenciamento de dependências (Pipenv).
- `request_example.http` — exemplo de requisição para testar a API localmente.
- `datasets/dataset_obesidade.csv` — dataset usado para treino/EDA.
- `datasets/dictionary_obesidade.txt` — dicionário de dados com descrição das colunas.

**Resumo do fluxo implementado**

1. Carregamento e EDA do dataset (`modelo-obesidade.ipynb`).
2. Bucketing de idade e engenharia de features (criação de `Faixa_Etaria`).
3. Seleção de features com `SelectKBest` (chi2).
4. Treinamento de baseline com `GaussianNB` e avaliação (classification report, matriz de confusão, recall macro).
5. Otimização de número de features com `Optuna` (estudo para `k`).
6. Salvamento do modelo em `modelo_obesidade.pkl` via `joblib`.
7. API Flask (`api_modelo_obesidade.py`) para inferência via POST `/predict`.

**Tecnologias**

- Python 3.11+
- pandas, numpy
- scikit-learn (GaussianNB, SelectKBest)
- joblib
- optuna (tuning)
- flask, pydantic, flask-pydantic (API)
- plotly/matplotlib/sweetviz (visualizações e EDA)

**Estrutura do projeto**

```
ml-naive-bayes/
├── api_modelo_obesidade.py
├── modelo-obesidade.ipynb
├── Pipfile
├── request_example.http
├── datasets/
│   ├── dataset_obesidade.csv
│   └── dictionary_obesidade.txt
└── modelo_obesidade.pkl (gerado pelo notebook)
```

**Como reproduzir o ambiente**

Recomendado: use Pipenv (há um `Pipfile` no repositório). No Windows (PowerShell):

```powershell
pip install pipenv
pipenv install --dev
pipenv shell
```

Se preferir usar `pip`, exporte dependências do `Pipfile` ou instale manualmente as libs listadas em `Pipfile`.

**Como executar (notebook)**

1. Abra o ambiente (p.ex. `pipenv shell`).
2. Execute `jupyter notebook` e abra `modelo-obesidade.ipynb`.
3. Execute as células na ordem: EDA → pré-processamento → treino → salvar modelo.

O notebook salva o modelo com:

```python
joblib.dump(model_kbest, 'modelo_obesidade.pkl')
```

**Como executar (API local)**

1. Certifique-se de que `modelo_obesidade.pkl` existe na raiz do projeto (gerado pelo notebook).
2. Rode a API Flask:

```bash
python api_modelo_obesidade.py
```

A API irá rodar por padrão em `http://localhost:5000` e expõe o endpoint `POST /predict`.

Exemplo de requisição (arquivo de exemplo incluído): veja [request_example.http](request_example.http).

Exemplo com `curl`:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '@request_example.http'
```

(No Windows, use o conteúdo JSON do [request_example.http](request_example.http) como body.)

**Inferência em batch**

Se preferir gerar predições em batch a partir de um arquivo CSV com as mesmas colunas/transformações do treinamento, crie um script simples que:

- carregue o CSV (ex.: `datasets/dataset_obesidade.csv`),
- aplique as mesmas transformações do notebook (bucketing de idade, seleção de features, etc.),
- carregue `modelo_obesidade.pkl` com `joblib.load(...)`,
- gere um arquivo de saída com as predições.

Exemplo mínimo (no README do repositório remoto havia um snippet semelhante).

**Dicionário de dados**

Veja o arquivo `datasets/dictionary_obesidade.txt` para descrição das colunas e valores possíveis.

**Próximos passos sugeridos (posso implementar)**

- Gerar `requirements.txt` a partir do `Pipfile`.
- Criar `inferencia_batch.py` para inferência em batch.
- Substituir Flask por FastAPI e adicionar `uvicorn` para produção, com endpoint e documentação automática (`/docs`).

Se quiser, eu implemento qualquer uma das opções acima.

**Contribuições**

1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/nome-da-feature`)
3. Commit e push
4. Abra um Pull Request

**Licença**

Se desejar, posso adicionar `LICENSE` (sugestão: MIT).
