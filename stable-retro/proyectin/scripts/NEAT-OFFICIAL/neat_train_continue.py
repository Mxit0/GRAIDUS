import retro
import numpy as np
import time
import csv
import os
from datetime import datetime, timedelta
import neat

# Crear el entorno
def make_env():
    """Crea y configura el entorno de GradiusIII-Snes."""
    env = retro.make(game="GradiusIII-Snes")
    return env

# Callback para recolectar datos durante el entrenamiento
class EvalCallback:
    def __init__(self, start_time, verbose=0, log_interval=1000, csv_filename="training_data_continue.csv"):
        self.max_score = -np.inf
        self.current_score = 0
        self.log_interval = log_interval
        self.start_time = start_time
        self.step_count = 0
        self.times_in_minutes = []
        self.scores = []
        self.csv_filename = csv_filename

    def on_step(self, score):
        """Este método se llama después de cada paso de entrenamiento."""
        self.step_count += 1
        self.current_score = score

        if self.current_score > self.max_score:
            self.max_score = self.current_score

        # Registrar puntaje y tiempo
        elapsed_time_minutes = (time.time() - self.start_time) / 60
        self.times_in_minutes.append(elapsed_time_minutes)
        self.scores.append(self.current_score)

        # Imprimir logs cada log_interval pasos
        if self.step_count % self.log_interval == 0:
            elapsed_time = timedelta(seconds=int(time.time() - self.start_time))
            print(f"Tiempo transcurrido de entrenamiento: {elapsed_time} - Puntaje actual: {self.current_score} - Puntaje Máximo: {self.max_score}")

        # Guardar datos en el archivo CSV
        self.save_to_csv()

    def save_to_csv(self):
        """Guardar los datos de tiempos y puntajes en un archivo CSV."""
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.times_in_minutes[-1], self.scores[-1]])

# Función para crear el modelo de NEAT
def create_neat_model(config_file):
    """Crea el modelo de NEAT usando el archivo de configuración."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    return p

# Función para la evaluación y entrenamiento de la red neuronal
def evaluate_genome(genome, config, env):
    """Evaluar una red neuronal de NEAT en el entorno."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_reward = 0
    state = env.reset()
    done = False

    while not done:
        output = net.activate(state)  # Obtener la salida de la red
        action = np.argmax(output)  # Tomar la acción con la mayor probabilidad
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state

    return total_reward

# Leer la configuración del archivo de NEAT
def read_config_file():
    """Leer el archivo de configuración de NEAT"""
    config_file = "config-neat"  # Asegúrate de tener este archivo de configuración
    return config_file

# Función principal para continuar el entrenamiento
def main():
    # Leer la información del archivo de log del entrenamiento anterior
    start_datetime_original, end_datetime_original, total_time_original = read_training_log()

    # Registrar el tiempo de inicio del nuevo entrenamiento
    start_time = time.time()
    start_datetime_continue = datetime.now()

    print(f"Hora de inicio del entrenamiento de continuación: {start_datetime_continue.strftime('%Y-%m-%d %H:%M:%S')}")

    # Crear el entorno de entrenamiento
    env = make_env()

    # Crear el modelo de NEAT
    config_file = read_config_file()
    neat_model = create_neat_model(config_file)

    # Crear el callback para recolectar datos durante el entrenamiento
    eval_callback = EvalCallback(start_time=start_time, verbose=1, log_interval=5000, csv_filename="training_data_continue.csv")

    # Entrenar el modelo de NEAT
    for generation in range(100):  # Este es solo un número de generaciones, ajusta según sea necesario
        print(f"Iniciando generación {generation}")
        
        # Evaluar cada genoma de la población de NEAT
        for genome_id, genome in neat_model.population.items():
            reward = evaluate_genome(genome, neat_model.config, env)
            eval_callback.on_step(reward)

        # Guardar el modelo de NEAT y los logs
        neat_model.save("neat_gradius_model_continue")

    # Guardar los logs de la continuación
    end_datetime_continue = datetime.now()  # Hora de finalización de la continuación
    total_time_continue = (time.time() - start_time) / 60  # Tiempo de la continuación

    print(f"Hora de fin de la continuación: {end_datetime_continue.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tiempo de la continuación: {total_time_continue:.2f} minutos")

    # Unificar puntajes y tiempos
    print(f"Puntaje máximo previo: {eval_callback.max_score}")
    print(f"Puntaje máximo actual: {eval_callback.max_score}")

    env.close()

if __name__ == "__main__":
    main()
