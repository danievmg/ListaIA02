#Daniel Cristiano

# importando as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# carregando o dataset
df = pd.read_csv('dataset_emprestimo_aprovacao.csv')

# visualizando os dados
print("Primeiras linhas:")
print(df.head())
print()

print("Informacoes do dataset:")
df.info()
print()

# checando se tem valores nulos
print("Valores nulos por coluna:")
print(df.isnull().sum())
print()

# vendo a distribuicao da variavel alvo
print("Distribuicao da variavel alvo (emprestimo_aprovado):")
print(df['emprestimo_aprovado'].value_counts())
print()

# analise exploratoria - boxplot pra ver como cada variavel se relaciona com a aprovacao
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

df.boxplot(column='renda_mensal', by='emprestimo_aprovado', ax=axes[0])
axes[0].set_title('Renda Mensal x Aprovacao')
axes[0].set_xlabel('Aprovado (0=Nao, 1=Sim)')

df.boxplot(column='score_credito', by='emprestimo_aprovado', ax=axes[1])
axes[1].set_title('Score de Credito x Aprovacao')
axes[1].set_xlabel('Aprovado (0=Nao, 1=Sim)')

df.boxplot(column='dividas_ativas', by='emprestimo_aprovado', ax=axes[2])
axes[2].set_title('Dividas Ativas x Aprovacao')
axes[2].set_xlabel('Aprovado (0=Nao, 1=Sim)')

plt.suptitle('')
plt.tight_layout()
plt.savefig('analise_exploratoria.png')
plt.show()

# separando as variaveis independentes (X) da variavel alvo (y)
X = df[['renda_mensal', 'score_credito', 'dividas_ativas']]
y = df['emprestimo_aprovado']

# dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dados de treino: {len(X_train)}")
print(f"Dados de teste: {len(X_test)}")
print()

# treinando o modelo de regressao logistica
# usei max_iter=1000 pra garantir que o modelo vai convergir
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train, y_train)

print("Modelo treinado!")
print()

# fazendo as previsoes
y_pred = modelo.predict(X_test)

# avaliando o modelo
print("=== Resultados ===")
print(f"Acuracia: {accuracy_score(y_test, y_pred):.4f}")
print()

# matriz de confusao
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusao:")
print(cm)
print()

# plotando a matriz de confusao de forma mais visual
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negado', 'Aprovado'],
            yticklabels=['Negado', 'Aprovado'])
plt.title('Matriz de Confusao - Regressao Logistica')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('matriz_confusao_logistica.png')
plt.show()

# precision, recall e f1
print("Relatorio de Classificacao:")
print(classification_report(y_test, y_pred, target_names=['Negado', 'Aprovado']))

# interpretando os coeficientes
print("=== Interpretacao dos Coeficientes ===")
coeficientes = pd.Series(modelo.coef_[0], index=X.columns)
print(coeficientes)
print()

# plotando os coeficientes
coeficientes.sort_values().plot(kind='barh', color='steelblue', figsize=(7, 4))
plt.title('Coeficientes do Modelo')
plt.axvline(x=0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig('coeficientes.png')
plt.show()

print("""
Interpretacao:
- score_credito tem coeficiente positivo, ou seja, quanto maior o score
  maior a chance do emprestimo ser aprovado
- dividas_ativas tem coeficiente negativo, ou seja, quem tem mais dividas
  tem menos chance de aprovacao
- renda_mensal tambem influencia positivamente na aprovacao
""")
