# Relatório com Versão Python e Versões dos Pacotes

# Execute este script para verificar as versões na sua máquina. 


import sys
import joblib
import pandas
import sklearn
import flask

packages = [joblib, pandas, sklearn, flask]

print('\nVersões da Linguagem Python e dos Pacotes Usados Neste Capítulo:\n')

# Extrai a string completa da versão
version_string = sys.version

# Dividir a string pela primeira ocorrência de espaço e pegar o primeiro elemento
version_number = version_string.split()[0]

print("Versão da Linguagem Python:", version_number)
for package in packages:
    print(f"Versão do {package.__name__}:", package.__version__)
