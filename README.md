# SIS-DCM - Sistema Inteligente de Suporte ao Diagnóstico de Câncer de Mama

**Relatório Técnico**: https://colab.research.google.com/drive/1gDEtNGraajF_hn4HGHEClsjkFzgaEi5q?usp=sharing

---

Este repositório contém o código-fonte, a estrutura de dados e as instruções de configuração para o modelo preditivo de auxílio ao diagnóstico médico de câncer de mama desenvolvido para o Tech Challenge (Fase 1).

## 📋 Estrutura do Projeto

- `app.py`: Pipeline completo de Machine Learning (Carregamento, Tratamento, Treinamento dos modelos e Avaliação de Performance).
- `Dockerfile`: Configuração de containerização para execução em qualquer ambiente.
- `requirements.txt`: Lista de dependências e bibliotecas Python necessárias.

## 🛠️ Tecnologias Utilizadas

- Python 3.9+
- Pandas e Numpy
- Scikit-Learn (Modelagem e Escalonamento)
- Docker

## 🚀 Como Executar o Projeto

### Opção 1: Executando Localmente com Python

1. Certifique-se de ter o Python instalado.
2. Instale as dependências requisitadas:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o script principal:
   ```bash
   python app.py
   ```

### Opção 2: Executando com Docker

Para garantir que o projeto funcione em qualquer máquina sem a necessidade de configurar ambientes locais:

1. Construa a imagem Docker:
   ```bash
   docker build -t sis-dcm-app .
   ```
2. Execute o container para ver o resultado do treinamento:
   ```bash
   docker run -it --name sis-dcm-container sis-dcm-app
   ```

## 📊 Resultados do Modelo

O projeto implementa e avalia dois modelos preditivos:

- **Regressão Logística**: Foco em alta velocidade de execução linear.
- **Random Forest**: Oferece melhor acurácia global e robustez lidando com limites não-lineares.
