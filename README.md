# Project Mestrado - Scripts Python

## Sobre 

Neste repositório são encontrados os scripts em `python` e `shell` para executar o treinamento e aplicação de uma rede neural sobre amostras de rochas digitais. O trabalho completo pode ser encontrado em seu próprio [repositório](https://github.com/hereisjohnny2/TeseMestrado).

Durante o processo de treinamento um arquivo em formato de texto, contendo os valores de Vermelho, Verde, Azul e *label* de vários pixeis de um determinada imagem, é utilizado para alimentar a rede neural, linha a linha. Esse *dataset* é dividido em dados de treino e de teste, onde 30% dos dados são categorizados como teste e 70% como treino.

Na entrada da rede neural são utilizados 3 neurônios, cada um representando um canal de cor, e na sua saída, apenas dois, um representando o valore obtido para categoria **Poro** e outro para a categoria **Solido**. O maior desses valores é comparado com a *label* do *dataset*, o erro obtido é propagado pela rede por meio do algoritmo de *Backpropagation* e o processo se repete para dos os dados treinamento por *n* épocas. 

Ao final do treinamento, é realizado o processo de teste, onde os valores do *dataset* de teste são aplicados à rede neural a fim de se verificar sua acurácia.

O resultado deste processo como um todo gera um arquivo `.pt`, que contém os valores do pesos e biases da rede após o treinamento. Esses valores então são carregados para que as imagens sejam aplicadas à ele, gerando como resultado um arquivo com os valores de **porosidade** e a imagem binarizada.

No texto do trabalho é possível encontrar mais informações sobre os resultados obtidos e a teoria por trás das inteligência artificial aplicada a rocha digital.

## Tecnologias

- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pillow](https://pillow.readthedocs.io/en/stable/)

## Rodando o projeto

Após clonar esse repositório, crie um ambiente virtual para executar o python e baixar suas dependências:

```shell
$ python -m venv venv
```

Em seguida, ative o ambiente virtual:

```shell
$ source venv/bin/activate 
```

Instale então as dependências do projeto:

```shell
$ pip install -r requirements.txt
```

Para treinar um novo modelo a partir de um *dataset* em arquivo de texto com um determinado número de épocas, digite no terminal:

```shell
$ python rock-nn/main.py -t <caminho-para-dataset> -e <numero-de-epocas>
```

Para aplicar o modelo treinado sobre uma imagem entre o o seguinte comando no terminal:

```shell
python rock-nn/main.py --save -i <caminho-para-imagem> -m <caminho-para-modelo> -o <diretorio-de-saida>
```

Algumas imagens podem não possuir os canais de cores em `RGB` e isso pode atrapalhar ou confundir a aplicação do modelo. Para isso execute o comando abaixo para convert a imagem diretamente para `.png` com os canais em `RGB`:

```shell
python rock-nn/main.py -i <caminho-para-imagem>
```