#MPORTANTE: Antes de começar, intale os pacotes contidos em requirements.tct
#
#python -m pip install -r requirements.txt

import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# IMPORTANTE: Instruções para Repositório de Dados 
# 1. Criar m docmento no Google Sheet
# 2. INsira dados com estes cabeçalhos: faltas, nota_1, nota_2, resultado
# 3. Adicione de 10 a 15 linhas conforme o exemplo abaixo:
#   8,2.5,4.5, reprovado
#   2,5.0,7.0,aprovado
#   ...(adicione até 15 linhas)
# 4. Ir em "Arquivo" -> "Compartilhar" -> "Publicar na web"
# 5. Selecionar "Valores separados por vírgula (.csv)" e clicar em "Publicar"
# 6. CCOPIAR O LINK GERADO E COLA ABAIXO:
URL_DADOS = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRvDTFPoL9mTAZ6LvP_jk0zN22rHj60_i3KBSELd-p4TgBGu_jnlhV_ZKTxgDzwN6KMpfoTU4ir3biL/pub?output=csv"

ARQUIVO_MODELO = "modelo_desempenho.pkl"

def treinar_modelo():
    print(f"Baixando dados de {URL_DADOS} ...")
    data = pd.read_csv(URL_DADOS)
    print("--- Dados Carregados ---")
    print(data.head())

    print("--- Preparando dados para o treino ---")
    features = ["faltas", "nota_1", "nota_2"]  
    target = "resultado"
    #X recebe as características(features) e Y recebe o
    #rótulo (chamado de alvo ou target, nesse caso)
    X = data[features]
    Y = data[target]

    #Instancia "modelo" com 3 vizinhos (n_neighbors=3)
    modelo = KNeighborsClassifier(n_neighbors=3)
    #Para treinar o modelo usa-se o método .fit() com as entradas X e Y 
    modelo.fit(X, Y)
    print(f"--- Modelo Treinado! Classes: {modelo.classes_}---")

    #Salvando o modelo treinado no arquivo cujo nome está armazenado na
    #constante ARQUIVO_MODELO
    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"--- Modelo salvo com sucesso em '{ARQUIVO_MODELO}' ---")

if __name__ == "__main__":
    treinar_modelo()