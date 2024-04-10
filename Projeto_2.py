import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.model_selection import train_test_split

st.set_page_config(page_icon = ':moneybag:', page_title = 'Previsão de Renda', layout = 'wide')

st.markdown('# Perfil dos clientes da instituição XY :moneybag:')

'A instituição financeira XY deseja uma análise profunda do perfil dos seus clientes atuais e um bom modelo de previsão de renda para novos clientes que ingressarem na instituição.'

'Para termos esses resultados, vamos fazer análises dos dados que temos disponíveis no banco de dados e entender o perfil dos clientes.'

'Nosso objetivo principal nesse projeto será entender melhor quem é o consumidor final da instituição financeira e, por consequência, atender melhor os clientes baseado em seu perfil e renda.'

st.markdown('## Análise dos dados')

'Analisando as variáveis úteis do DataFrame:'

st.markdown('''| Variável                | Descrição                                           | Tipo         |
|:-----------------------:|:-----------------------:| ------------:|
| sexo                    |  Gênero do cliente    | object       |
| posse_de_veiculo        |  Se possui ou não veículo    | bool         |
| posse_de_imovel         |  Se possui ou não imóvel    | bool         |
| tipo_renda              |  Qual a fonte da sua renda    | object       |
| educacao                |  Nível de escolaridade    | object       |
| estado_civil            |  Estado civil do cliente    | object       |
| idade                   |  Idade do cliente    | int64        |
| tempo_emprego           |  Tempo no emprego atual    | float64      |
| qt_pessoas_residencia   |  Quantas pessoas moram com o cliente    | float64      |
| renda                   |  Sua renda    | float64      |''', unsafe_allow_html=True)

st.markdown("<style> h1 {text-align: center;} </style>", unsafe_allow_html=True)
st.markdown("<style> h2 {text-align: center;} </style>", unsafe_allow_html=True)
st.markdown("<style> h5 {text-align: center;} </style>", unsafe_allow_html=True)

renda = pd.read_csv('./input/previsao_de_renda.csv')

renda.drop_duplicates(inplace = True)

renda.drop(columns = ['Unnamed: 0', 'data_ref', 'id_cliente', 'tipo_residencia', 'qtd_filhos'], inplace = True) 

renda['qt_pessoas_residencia'] = renda['qt_pessoas_residencia'].astype(int)

renda.fillna(0, inplace = True)

'Agora, vamos ver na prática como está o nosso dataframe já com todas as preparações feitas para a nossa análise:'

st.write(renda.head(1))

'Criando gráficos para análise univariada:'

var_dict = {'sexo' : 'Sexo', 'posse_de_veiculo' : 'Possui Veículo', 'posse_de_imovel' : 'Possui Imóvel',
            'tipo_renda' : 'Tipo de Renda', 'educacao' : 'Educação', 'estado_civil' : 'Estado Civil',
            'idade' : 'Idade', 'tempo_emprego' : 'Tempo no emprego',
            'qt_pessoas_residencia' : 'Quantidade de Pessoas na Residência', 'renda' : 'Renda'
            }

inv_var_dict = {'Sexo': 'sexo', 'Possui Veículo': 'posse_de_veiculo', 'Possui Imóvel': 'posse_de_imovel', 
                'Tipo de Renda': 'tipo_renda', 'Educação': 'educacao', 'Estado Civil': 'estado_civil', 
                'Idade': 'idade', 'Tempo no emprego': 'tempo_emprego',
                'Quantidade de Pessoas na Residência': 'qt_pessoas_residencia'}

cat = renda.select_dtypes(include = ['object', 'bool']).columns.tolist()

categoricos = st.selectbox('Selecione uma das variáveis categóricas:', (var_dict.get(chave) for chave in cat))
    
fig, ax = plt.subplots(figsize = (4,2))
sns.countplot(data = renda, x = inv_var_dict[categoricos], palette = 'rocket')
ax.set_title(f'Quantidade de cliente por: {categoricos}', fontdict={'fontsize':5})
ax.set_xlabel(f'{categoricos}', fontdict={'fontsize':4})
ax.set_ylabel('Contagem', fontdict={'fontsize':4})
ax.tick_params(labelsize = 4)
fig.tight_layout()
sns.despine(bottom = True, left = False)
st.pyplot(fig, use_container_width = False)

num = renda.select_dtypes(include = ['number']).columns.drop('renda').tolist()

numericas = st.selectbox('Selecione uma das variáveis numéricas:', (var_dict.get(chave) for chave in num))

fig, ax = plt.subplots(figsize = (4,2))
sns.histplot(data = renda, x = inv_var_dict[numericas], palette = 'rocket', color = 'rebeccapurple',
             element = 'step', bins = 50, discrete = True)  
ax.set_title(f'Quantidade de cliente por: {numericas}', fontdict={'fontsize':5})
ax.set_xlabel(f'{numericas}', fontdict={'fontsize':4})
ax.set_ylabel('Contagem', fontdict={'fontsize':4})
ax.tick_params(labelsize = 4)
fig.tight_layout()
sns.despine(bottom = True, left = False)
st.pyplot(fig, use_container_width = False)

'Criando gráficos para análise bivariada:'

var1 = st.selectbox('Selecione uma das variáveis:', inv_var_dict)

var2 = st.selectbox('Selecione a outra variável:', inv_var_dict)

if var1 == var2:
    st.write('Por favor, selecione variáveis diferentes')
elif inv_var_dict[var1] in cat and inv_var_dict[var2] in cat:
    fig, ax = plt.subplots(figsize = (4,2))
    sns.countplot(data = renda, x = inv_var_dict[var1], hue = inv_var_dict[var2], palette = 'rocket')
    ax.set_title(f'{var1} x {var2}', fontdict={'fontsize':5})
    ax.set_xlabel(f'{var1}', fontdict={'fontsize':4})
    ax.set_ylabel('Contagem', fontdict={'fontsize':4})
    ax.legend(fontsize = 5)
    ax.tick_params(labelsize = 4)
    fig.tight_layout()
    sns.despine(bottom = True, left = False)
    st.pyplot(fig, use_container_width = False)
elif inv_var_dict[var1] in num and inv_var_dict[var2] in num:
    fig, ax = plt.subplots(figsize = (4,2))
    sns.scatterplot(data = renda, x = inv_var_dict[var1], y = inv_var_dict[var2], color = 'purple')
    ax.set_title(f'{var1} x {var2}', fontdict={'fontsize':5})
    ax.set_xlabel(f'{var1}', fontdict={'fontsize':4})
    ax.set_ylabel(f'{var2}', fontdict={'fontsize':4})
    ax.tick_params(labelsize = 4)
    fig.tight_layout()
    sns.despine(bottom = True, left = False)
    st.pyplot(fig, use_container_width = False)
elif inv_var_dict[var1] in num and inv_var_dict[var2] in cat:
    fig, ax = plt.subplots(figsize = (4,2))
    sns.violinplot(data = renda, x = inv_var_dict[var1], y = inv_var_dict[var2], 
                   palette = 'rocket', linewidth = 0.25)
    ax.set_title(f'{var1} x {var2}', fontdict={'fontsize':5})
    ax.set_xlabel(f'{var1}', fontdict={'fontsize':4})
    ax.set_ylabel(f'{var2}', fontdict={'fontsize':4})
    ax.tick_params(labelsize = 4)
    fig.tight_layout()
    sns.despine(bottom = True, left = False)
    st.pyplot(fig, use_container_width = False)
elif inv_var_dict[var1] in cat and inv_var_dict[var2] in num:    
    fig, ax = plt.subplots(figsize = (4,2))
    sns.violinplot(data = renda, x = inv_var_dict[var2], y = inv_var_dict[var1], 
                   palette = 'rocket', linewidth = 0.25)
    ax.set_title(f'{var1} x {var2}', fontdict={'fontsize':5})
    ax.set_xlabel(f'{var1}', fontdict={'fontsize':4})
    ax.set_ylabel(f'{var2}', fontdict={'fontsize':4})
    ax.tick_params(labelsize = 4)
    fig.tight_layout()
    sns.despine(bottom = True, left = False)
    st.pyplot(fig, use_container_width = False)
else:
    st.write('erro')

st.markdown(' ## Principal perfil de cliente')

renda['idade'] = pd.cut(renda['idade'], bins = [0, 18, 30, 50, 80], labels = ['Menor de Idade: 0 a 18 anos', 'Jovem Adulto: 19 a 30 anos', 
                                                                              'Adulto: 31 - 49 anos', 'Adulto +: Mais de 50 anos'])

renda['renda'] = pd.cut(renda['renda'], bins = [0, 5000, 8000, 15000, 300000000], labels = ['Até $5.000', 
                                                                                            'De $5.000 a $8.000', 
                                                                                            'De $8.000 a $15.000', 'Mais de $15.000'])

renda['qt_pessoas_residencia'] = pd.cut(renda['qt_pessoas_residencia'], bins = [0, 1, 3, 5, 100], labels = ['Mora Sozinho(a)', 
                                                                                                            'Mora com 2 ou 3 pessoas', 
                                                                                                            'Mora com 4 ou 5 pessoas', 
                                                                                                            'Mora com mais de 6 pessoas'])

tabela = []


for coluna in renda.columns.drop(['tempo_emprego']):
    mais_frequente = renda[coluna].mode()[0]
    porcentagem = renda[coluna].value_counts(normalize = True)[0]*100
    linha = [f'{var_dict[coluna]}', f'{mais_frequente}', f'{porcentagem:.2f}%']
    tabela.append(linha)

tabela = pd.DataFrame(tabela)

tabela.columns = ['Coluna', 'Valor mais frequente', 'Porcentagem de frequência']

st.table(tabela)

'Entendemos assim que o perfil mais comum é:'

'##### Mulher com ensino Médio completo, assalariada ganhando até $5.000.'
'##### Casada, morando em casa com 2 ou 3 pessoas possuindo imóvel mas não tem carro.'