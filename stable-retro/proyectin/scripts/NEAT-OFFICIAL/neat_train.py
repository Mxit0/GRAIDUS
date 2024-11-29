import retro
import neat
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Crear el entorno
def create_environment():
    """Crear y configurar el entorno de GradiusIII-Snes."""
    env = retro.make(game="GradiusIII-Snes")
    return env

# Función para evaluar una red neuronal en el entorno
def eval_genomes(genomes, config):
    """Evaluar la población de redes neuronales."""
    for genome_id, genome in genomes:
        genome.fitness = 0  # Inicializar la puntuación de cada genoma

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = create_environment()

        done = False
        obs = env.reset()
        total_reward = 0

        while not done:
            action = np.argmax(net.activate(obs))  # Tomar acción usando la red neuronal
            obs, reward, done, info = env.step(action)  # Realizar la acción en el entorno
            total_reward += reward

        genome.fitness = total_reward  # Asignar la recompensa total como la aptitud del genoma
        env.close()

# Función para guardar los datos de entrenamiento en un archivo CSV
def save_training_data(times, scores, filename="training_data_neat.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (minutes)", "Score"])
        writer.writerows(zip(times, scores))
    print(f"Datos guardados en {filename}")

# Callback para recolectar datos durante el entrenamiento
class EvalCallback:
    def __init__(self, start_time, verbose=0, log_interval=1000):
        self.max_score = -np.inf  # Puntaje máximo inicial
        self.current_score = 0  # Puntaje actual
        self.log_interval = log_interval  # Intervalo para imprimir logs
        self.start_time = start_time  # Tiempo de inicio del entrenamiento
        self.step_count = 0  # Contador de pasos
        self.times_in_minutes = []  # Almacenar tiempos en minutos
        self.scores = []  # Almacenar puntajes

    def collect_data(self, score):
        """Este método se llama después de cada paso de entrenamiento.
        Guarda el puntaje máximo y actual en cada paso, y registra el tiempo en minutos.
        """
        self.step_count += 1

        # Actualizar el puntaje máximo
        if score > self.max_score:
            self.max_score = score

        # Registrar puntaje y tiempo
        elapsed_time_minutes = (time.time() - self.start_time) / 60
        self.times_in_minutes.append(elapsed_time_minutes)
        self.scores.append(score)

        # Imprimir logs cada log_interval pasos
        if self.step_count % self.log_interval == 0:
            elapsed_time = timedelta(seconds=int(time.time() - self.start_time))
            print(f"Tiempo transcurrido de entrenamiento: {elapsed_time} - "
                  f"Puntaje actual: {score} - "
                  f"Puntaje Máximo: {self.max_score}")

# Función principal para entrenar NEAT
def train_neat():
    """Entrenar el modelo NEAT en GradiusIII-Snes."""
    # Cargar la configuración del NEAT
    config_path = "config_neat.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Crear una población inicial
    p = neat.Population(config)

    # Crear el callback para almacenar los datos durante el entrenamiento
    start_time = time.time()
    eval_callback = EvalCallback(start_time=start_time, log_interval=5000)

    # Definir el número de generaciones para el entrenamiento
    generations = 1000  # Ajusta según tus necesidades
    print(f"Iniciando el entrenamiento con {generations} generaciones...")

    # Iniciar el entrenamiento
    p.run(eval_genomes, generations)

    # Unificar los datos de todos los callbacks
    unified_times = eval_callback.times_in_minutes
    unified_scores = eval_callback.scores

    # Guardar los resultados en un archivo CSV
    save_training_data(unified_times, unified_scores, filename="training_data_neat.csv")

    # Graficar el puntaje vs tiempo de entrenamiento (en minutos)
    combined_data = sorted(zip(unified_times, unified_scores))
    sorted_times, sorted_scores = zip(*combined_data)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_times, sorted_scores, label="Score", color="blue")
    plt.xlabel('Tiempo de entrenamiento (minutos)')
    plt.ylabel('Puntaje')
    plt.title('Puntaje vs Tiempo de Entrenamiento - NEAT')
    plt.grid(True)
    plt.legend()
    plt.savefig('score_vs_time_neat.png')  # Guardamos la gráfica
    print("Gráfico guardado como 'score_vs_time_neat.png'.")

    # Calcular el tiempo total de ejecución
    total_time_minutes = (time.time() - start_time) / 60
    end_datetime = datetime.now()  # Hora de fin en formato legible

    # Guardar logs en un archivo
    log_filename = "neat_train_log.txt"
    with open(log_filename, "w") as log_file:
        log_file.write(f"La hora de inicio del entrenamiento fue: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Fin del entrenamiento: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"El entrenamiento tomó {total_time_minutes:.2f} minutos en total.\n")

    # También imprimir en consola
    print(f"La hora de inicio del entrenamiento fue: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fin del entrenamiento: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"El entrenamiento tomó {total_time_minutes:.2f} minutos en total.")

if __name__ == "__main__":
    train_neat()
