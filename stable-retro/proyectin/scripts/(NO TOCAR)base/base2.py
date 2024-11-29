import retro
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Forzar el uso de un backend que no necesita GUI (sin Qt)

# Mapeo de botones basado en la tabla proporcionada
botones = [
    "B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"
]

def describir_accion(action):
    """Convierte el vector de acción binario en una lista de botones presionados."""
    descripcion = []
    for i, estado in enumerate(action):
        if estado == 1:
            descripcion.append(botones[i])
    return descripcion

def main():
    # Especificar el juego
    game = "GradiusIII-Snes"
    env = retro.make(game)
    
    # Configuración del experimento
    max_tries = 5  # Número de intentos permitidos
    intentos_colores = ['b', 'g', 'r', 'c', 'm']  # Colores para los intentos

    done = False
    global_step = 0  # Contador de pasos global
    intentos = []  # Lista para almacenar datos por intento
    current_intento = []  # Datos del intento actual (steps y scores)

    # Inicializar variables del intento actual
    intento_actual = 1
    step_count = 0
    max_score = 0

    for intento in range(max_tries):
        print(f"--- Intento {intento + 1} ---")
        obs = env.reset()
        done = False
        step_count = 0  # Contador de pasos local
        scores = []
        steps = []

        while not done:
            action = np.zeros(12, dtype=int)  # Inicia el vector de acción con todos los botones sin presionar
            
            # Control de tiempo para mover arriba y abajo
            if step_count < 100:  # Moverse hacia arriba durante los primeros 100 pasos
                action[4] = 1  # Mover hacia arriba (UP)
            elif step_count < 200:  # Moverse hacia abajo durante los siguientes 100 pasos
                action[5] = 1  # Mover hacia abajo (DOWN)
            else:  # Después de los 200 pasos, reiniciar el ciclo
                step_count = 0

            # Disparar constantemente durante todo el tiempo
            action[0] = 1  # Disparar (B)
            
            # Ejecutar la acción en el entorno
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1  # Aumenta el contador de pasos local
            global_step += 1  # Aumenta el contador de pasos global

            # Obtener puntaje del juego
            score = info.get('score', 0)  # Puntaje actual
            lives = info.get('lives', 'No disponible')
            max_score = max(max_score, score)

            # Registrar datos del intento actual
            scores.append(score)
            steps.append(global_step)

            # Imprimir el estado actual
            print(f"Intento: {intento + 1} - Paso global: {global_step} - Paso local: {step_count} - Puntaje máximo: {max_score} - Puntaje: {score} - Vidas: {lives}")

            # Verificar si se perdió una vida o el juego terminó
            if truncated or done or info.get('lives', 0) == 0:
                break

        # Almacenar datos del intento actual
        intentos.append((steps, scores))

    # Graficar puntaje vs pasos para cada intento con un color distinto
    plt.figure(figsize=(10, 6))
    for i, (steps, scores) in enumerate(intentos):
        plt.plot(steps, scores, label=f"Intento {i + 1}", color=intentos_colores[i % len(intentos_colores)])
    
    # Configurar el gráfico
    plt.xlabel('Pasos')
    plt.ylabel('Puntaje')
    plt.title('Puntaje vs Pasos por Intento')
    plt.legend()
    plt.grid(True)

    # Guardar el gráfico como archivo
    plt.savefig('score_vs_steps_multicolor.png')  # Guarda el gráfico como un archivo PNG
    print("Gráfico guardado como 'score_vs_steps_multicolor.png'.")

    env.close()  # Cerrar el entorno cuando haya terminado

if __name__ == "__main__":
    main()
