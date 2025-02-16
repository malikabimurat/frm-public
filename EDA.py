#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
dataset_path = "Dataset.csv"
variables_description_path = "Variables_description.csv"

# Читаем файлы с указанием кодировки
try:
    df = pd.read_csv(dataset_path, encoding="Windows-1251")
    variables_description = pd.read_csv(variables_description_path)
except Exception as e:
    print("Ошибка загрузки данных:", e)


# In[4]:


# Просмотр общей информации о данных
df.info()
df.head()



# In[5]:


# 1. Анализ пропущенных значений
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("Топ-10 переменных с наибольшим количеством пропусков:")
print(missing_values.head(10))

# Визуализация пропусков
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.values, y=missing_values.index, palette="viridis")
plt.xlabel("Количество пропущенных значений")
plt.ylabel("Переменные")
plt.title("Пропущенные значения в данных")
plt.show()


# In[6]:


# 2. Поиск выбросов через IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Определение границ выбросов
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Подсчет выбросов
outliers = ((df < lower_bound) | (df > upper_bound)).sum()
print("Топ-10 переменных с наибольшим количеством выбросов:")
print(outliers.sort_values(ascending=False).head(10))


# In[7]:


# 3. Расчет Information Value (IV) и Weight of Evidence (WoE)
target = 'GB_flag'  # Целевая переменная

# Функция расчета IV и WoE
def calculate_iv_woe(df, feature, target):
    df = df[[feature, target]].copy()
    df = df.dropna()
    
    try:
        df["bin"] = pd.qcut(df[feature], q=10, duplicates='drop')  # Децилирование
    except Exception:
        return None, None  # Если не удалось разбить на бины
    
    grouped = df.groupby("bin")[target].agg(["count", "sum"])
    grouped.columns = ["total", "fraud"]
    grouped["non_fraud"] = grouped["total"] - grouped["fraud"]
    
    grouped["perc_fraud"] = grouped["fraud"] / grouped["fraud"].sum()
    grouped["perc_non_fraud"] = grouped["non_fraud"] / grouped["non_fraud"].sum()
    
    grouped["WoE"] = np.log(grouped["perc_fraud"] / grouped["perc_non_fraud"]).replace({np.inf: 0, -np.inf: 0})
    grouped["IV"] = (grouped["perc_fraud"] - grouped["perc_non_fraud"]) * grouped["WoE"]
    
    return grouped["IV"].sum(), grouped[["WoE"]]

# Вычисление IV и WoE для всех числовых переменных
iv_values = {}
woe_values = {}

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove(target)  # Исключаем целевую переменную

for col in num_cols:
    iv, woe = calculate_iv_woe(df, col, target)
    if iv is not None:
        iv_values[col] = iv
        woe_values[col] = woe

# Топ-10 переменных по IV
iv_sorted = sorted(iv_values.items(), key=lambda x: x[1], reverse=True)
print("Топ-10 переменных с наибольшим IV:")
print(iv_sorted[:10])


# In[8]:


# Преобразуем нужные столбцы в числовой формат, заменяя ошибки на NaN
cols_to_convert = ['overdueamount', 'outstandingamount', 'instalmentamount', 
                   'NUM_CONTRACTS_FOR', 'NUM_CONTRACTS_STARTED_L6M']

for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Ошибки (str) заменяются на NaN

# Генерация новых признаков
df['debt_ratio'] = df['overdueamount'] / (df['outstandingamount'] + 1) # Доля просроченной задолженности.Показывает, насколько большая часть кредита уже просрочена.
#Если этот показатель высокий, вероятность мошенничества может быть выше.
df['installment_ratio'] = df['instalmentamount'] / (df['outstandingamount'] + 1) # Отношение платежа к задолженности. Определяет, насколько ежемесячный платеж соотносится с общей суммой задолженности. Высокие значения могут указывать на заемщиков с низкой кредитной нагрузкой.
df['new_feature_1'] = df['NUM_CONTRACTS_FOR'] * df['NUM_CONTRACTS_STARTED_L6M'] # Пример новой фичи. Если человек оформляет много новых кредитов за последние 6 месяцев, это может быть подозрительным сигналом (например, обналичивание кредитов перед исчезновением).

print("Созданы новые признаки")


# In[9]:


# Сохранение подготовленного датасета
df.to_csv("Processed_dataset.csv", index=False)
print("Файл с обработанными данными сохранен")


# In[10]:


pd.read_csv("Processed_dataset.csv")


# In[ ]:




