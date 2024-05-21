import pandas as pd

def llenar_nulos_con_cero(archivo_entrada, archivo_salida):
    # Cargar el archivo CSV
    df = pd.read_csv(archivo_entrada)

    # Rellenar valores nulos con 0
    df = df.fillna(0)

    # Guardar el DataFrame actualizado en un nuevo archivo CSV
    df.to_csv(archivo_salida, index=False)

if __name__ == "__main__":
    # Reemplaza 'archivo_entrada.csv' con el nombre de tu archivo de entrada
    # Reemplaza 'archivo_salida.csv' con el nombre que deseas para el archivo de salida
    llenar_nulos_con_cero('./data/sonarkube-final.csv', 'SONARKUBE_FINAL.csv')
