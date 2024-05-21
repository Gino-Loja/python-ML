
while True:
        try:
            rows = int(input("Ingrese el número de filas (1-20): "))
            cols = int(input("Ingrese el número de columnas (1-20): "))
            if 1 <= rows <= 20 and 1 <= cols <= 20:
                break
            else:
                print("Por favor, ingrese números entre 1 y 20.")
        except ValueError:
            print("Por favor, ingrese un número entero.")

matrix = [[col + row * cols + 1 for col in range(cols)] for row in range(rows)]

def print_matrix(matrix):
    print("     ", end=' ')
  
    for i in range(len(matrix[0])):
        print("{:3d}".format(i), end=' ')
        
    print()
    print("----"*(len(matrix[0])+2))
    
    for i, row in enumerate(matrix):
        print("{:3d}".format(i),"|", end=' ')
        for num in row:
            print("{:3d}".format(num), end=' ')
        print()


print_matrix(matrix)


