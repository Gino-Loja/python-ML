import pandas as pd

# Lee el dataset

# Divide el DataFrame en grupos de 2016 filas
df = pd.read_csv('C:/Users/ginol/Downloads/datos_sensores.csv')  # Reemplaza 'tu_archivo.csv' con el nombre de tu archivo
  # Reemplaza 'tu_archivo.csv' con el nombre de tu archivo

# Divide el DataFrame en grupos de 2016 filas
grupos = [df.iloc[i:i+2016] for i in range(0, len(df), 2016)]

# Calcula el promedio de cada columna para cada grupo
promedios_por_grupo = [grupo.mean() for grupo in grupos]

# Crea un nuevo DataFrame con los promedios
df_promedios = pd.DataFrame(promedios_por_grupo)

# Guarda el DataFrame con los promedios en un solo archivo CSV
df_promedios.to_csv('promedios_totales.csv', index=False)

print("Proceso completado. Se ha creado un archivo con los promedios totales.")



