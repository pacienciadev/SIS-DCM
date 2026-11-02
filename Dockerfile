# Utilizar uma imagem oficial do Python como base
FROM python:3.9-slim

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependências
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código de aplicação
COPY app.py .

# Definir o comando padrão de execução
CMD ["python", "app.py"]
