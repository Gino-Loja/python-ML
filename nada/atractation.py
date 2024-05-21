import time

# Definición de la clase Visitante


class Visitante:
    def __init__(self, nombre, edad, altura):
        self.nombre = nombre
        self.edad = edad
        self.altura = altura
        # Lista para registrar las atracciones visitadas por el visitante
        self.atracciones_visitadas = []

# Definición de la clase Atraccion


class Atraccion:
    def __init__(self, nombre, capacidad_maxima, tiempo_espera, edad_minima, altura_minima):
        self.nombre = nombre
        self.capacidad_maxima = capacidad_maxima
        self.tiempo_espera = tiempo_espera
        self.edad_minima = edad_minima
        self.altura_minima = altura_minima
        # Lista que almacena los visitantes en la atracción en un momento dado
        self.lista_visitantes = []
        self.lleno = False  # Indica si la atracción está llena
        self.tiempo_inicio = 0  # Tiempo de inicio de la atracción
        self.tiempo = 0  # Tiempo transcurrido desde que la atracción está llena

    def validacion_visitante_atraccion(self, visitante: Visitante):
        # Verifica si el visitante cumple con los requisitos de edad y altura para ingresar a la atracción
        if visitante.edad >= self.edad_minima and visitante.altura >= self.altura_minima:
            # Registra la atracción en las atracciones visitadas por el visitante
            if not (self.nombre in visitante.atracciones_visitadas):
                visitante.atracciones_visitadas.append(self.nombre)
            # Agrega al visitante a la lista de visitantes de la atracción
            self.lista_visitantes.append(visitante)
            return True
        else:
            return False

    def verificar_disponibilidad(self):
        # Verifica si la atracción está llena
        if len(self.lista_visitantes) == self.capacidad_maxima:
            self.lleno = True
            # Registra el tiempo de inicio si es la primera vez que la atracción está llena
            if self.tiempo_inicio == 0:
                self.tiempo_inicio = int(time.time()) % 60
            return self.lleno
        return self.lleno

    def salida_visitantes(self):
        # Vacía la lista de visitantes de la atracción
        self.lista_visitantes = []

    def cupos_disponibles(self):

        return self.capacidad_maxima - len(self.lista_visitantes)

    def tiempo_duracion(self):
        # Calcula el tiempo transcurrido si la atracción está llena
        if self.lleno:
            self.tiempo = (int(time.time()) % 60) - self.tiempo_inicio
            # Si el tiempo transcurrido supera el tiempo de espera, reinicia la atracción
            if self.tiempo >= self.tiempo_espera:
                self.tiempo_inicio = 0
                self.salida_visitantes()
                self.lleno = False
        else:
            self.tiempo = 0

# Definición de la clase Reserva


class Reserva:
    def __init__(self, lista_atracciones: list[Atraccion]):
        self.atracciones = lista_atracciones
        self.visitantes = []  # Lista que almacena los visitantes que han ingresado al parque

    def agregar_visitante(self, visitante: Visitante):
        # Agrega un visitante a la lista de visitantes
        self.visitantes.append(visitante)

    def realizar_reserva(self, indice: int):
        # Verifica si un visitante cumple con los requisitos para reservar en una atracción específica
        cumple = self.atracciones[indice].validacion_visitante_atraccion(
            self.visitantes[len(self.visitantes) - 1])
        return cumple

    def paso_tiempo(self):
        # Simula el paso del tiempo para cambiar el estado de las atracciones
        for i in self.atracciones:
            i.tiempo_duracion()


# Creación de instancias de Atraccion
# El tiempo es en segundos por obvias razonas 
                        #nombre       # c_maxima # tiempo # edad #altura
montana_rusa = Atraccion("Montaña Rusa", 3, 5, 15, 150)
rueda_chicago = Atraccion("Rueda de Chicago", 8, 10, 18, 150)
casa_embrujada = Atraccion("Casa Embrujada", 10, 10, 18, 150)

# Creación de la instancia de Reserva con las atracciones disponibles
reserva = Reserva([montana_rusa, rueda_chicago, casa_embrujada])

# Menú principal del programa
while True:
    print("1. Ingresar Visitante")
    print("2. Realizar Reserva")
    print("3. Disponibilidad De Reservación")
    print("4. Mostrar Datos")
    print("5. Salir")

    opcion = int(input("Selecciona una opción: "))

    if opcion == 1:
        # Ingreso de datos de un nuevo visitante
        nombre = input("Ingrese su nombre: ")
       
        # Validación de la edad
        while True:
            try:
                edad = int(input("Ingrese su edad: "))
                if edad > 0 and edad < 100:
                    break
                else:
                    print("La edad debe estar entre 0 y 100. Inténtalo de nuevo.")
            except ValueError:
                print("Por favor, ingrese un número entero para la edad.")

        # Validación de la altura
        while True:
            try:
                altura = int(input("Ingrese su altura: "))
                if  altura > 0 and  altura < 210:
                    break
                else:
                    print("La altura debe estar entre 0 y 210. Inténtalo de nuevo.")
            except ValueError:
                print("Por favor, ingrese un número entero para la altura.")

        reserva.agregar_visitante(Visitante(nombre, edad, altura))
        print()

    elif opcion == 2:
        while True:
            print("Atracciones Disponibles")
            for i in range(len(reserva.atracciones)):
                # Muestra las atracciones disponibles para reservar
                if not reserva.atracciones[i].verificar_disponibilidad():
                    print(i + 1, ". ",
                          reserva.atracciones[i].nombre,
                          ", cupos disponibles: ", reserva.atracciones[i].cupos_disponibles())

            opcion_atraccion = int(input("Selecciona una atracción: "))

            if 1 <= opcion_atraccion <= len(reserva.atracciones):
                # Realiza la reserva y muestra el resultado
                validacion = reserva.realizar_reserva(opcion_atraccion - 1)
                if validacion:
                    print("Se ha registrado satisfactoriamente")
                else:
                    print(
                        "Lamentablemente usted no cumple con los requisitos mínimos para disfrutar de esta atracción")
                break
            else:
                print("Opción no válida. Inténtalo de nuevo.")

    elif opcion == 3:
        # Muestra la disponibilidad de reservación y el tiempo transcurrido en cada atracción
        print("Numero visitantes en cada atracción")
        for i in reserva.atracciones:
            print(i.nombre, ", capacidad: ",
                  i.capacidad_maxima, ", estado: ",
                  ", lleno" if i.verificar_disponibilidad() else "Disponible",
                  ", Tiempo transcurrido: ", i.tiempo)

    elif opcion == 4:
        # Muestra el número de atracciones y el historial de visitantes
        print("Numero de atracciones")
        for i in reserva.atracciones:
            print(i.nombre)
        print("Historial de Visitantes")
        for i in reserva.visitantes:
            print(i.nombre, "A visitado: ", i.atracciones_visitadas)

    elif opcion == 5:
        # Sale del programa
        print("Saliendo del programa. ¡Hasta luego!")
        break

    else:
        print("Opción no válida. Inténtalo de nuevo.")

    # Simula el paso del tiempo para cambiar el estado de las atracciones
    reserva.paso_tiempo()