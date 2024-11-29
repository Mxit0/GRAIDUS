import retro
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
import csv

matplotlib.use('Agg')  # Para usar un backend sin necesidad de GUI

# Crear el entorno
def make_env():
    """Crea y configura el entorno de GradiusIII-Snes."""
    env = retro.make(game="GradiusIII-Snes")
    env = WarpFrame(env)  # Preprocesamiento de las imágenes
    env = ClipRewardEnv(env)  # Recorta las recompensas
    return env

# Callback para recolectar datos durante el entrenamiento
class EvalCallback(BaseCallback):
    def __init__(self, start_time, verbose=0, log_interval=1000, csv_filename="training_data_continue.csv"):
        super(EvalCallback, self).__init__(verbose)
        self.max_score = -np.inf  # Puntaje máximo inicial
        self.current_score = 0  # Puntaje actual
        self.log_interval = log_interval  # Intervalo para imprimir logs
        self.start_time = start_time  # Tiempo de inicio del entrenamiento
        self.step_count = 0  # Contador de pasos
        self.times_in_minutes = []  # Almacenar tiempos en minutos
        self.scores = []  # Almacenar puntajes
        self.csv_filename = csv_filename  # Nombre del archivo CSV donde se guardarán los datos

    def _on_step(self) -> bool:
        """
        Este método se llama después de cada paso de entrenamiento.
        Guarda el puntaje máximo y actual en cada paso, y registra el tiempo en minutos.
        """
        self.step_count += 1

        # Obtenemos el puntaje de la última observación
        info = self.locals.get("infos", [])[0]  # Información de la última observación
        self.current_score = info.get('score', 0)  # Puntaje actual

        # Actualizar el puntaje máximo
        if self.current_score > self.max_score:
            self.max_score = self.current_score

        # Registrar puntaje y tiempo
        elapsed_time_minutes = (time.time() - self.start_time) / 60
        self.times_in_minutes.append(elapsed_time_minutes)
        self.scores.append(self.current_score)

        # Imprimir logs cada log_interval pasos
        if self.step_count % self.log_interval == 0:
            elapsed_time = timedelta(seconds=int(time.time() - self.start_time))
            print(f"Tiempo transcurrido de entrenamiento: {elapsed_time} - "
                  f"Puntaje actual: {self.current_score} - "
                  f"Puntaje Máximo: {self.max_score}")

        # Guardar datos en el archivo CSV
        self.save_to_csv()

        return True  # Retornar True para continuar el entrenamiento

    def save_to_csv(self):
        """Guardar los datos de tiempos y puntajes en un archivo CSV."""
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.times_in_minutes[-1], self.scores[-1]])

def read_training_log():
    """Leer el archivo de log para obtener los datos del entrenamiento previo"""
    log_filename = "ppoTrainLog_continue.txt"
    if os.path.exists(log_filename):
        with open(log_filename, "r") as log_file:
            lines = log_file.readlines()
            start_datetime_original = datetime.strptime(lines[0].strip().split(": ")[1], "%Y-%m-%d %H:%M:%S")
            end_datetime_original = datetime.strptime(lines[1].strip().split(": ")[1], "%Y-%m-%d %H:%M:%S")
            total_time_original = float(lines[2].strip().split(": ")[1].replace(" minutos en total.", ""))
        return start_datetime_original, end_datetime_original, total_time_original
    else:
        print("No se encontró el archivo de log anterior.")
        return None, None, None

def main():
    # Leer la información del archivo de log del entrenamiento anterior
    start_datetime_original, end_datetime_original, total_time_original = read_training_log()

    # Registrar el tiempo de inicio del nuevo entrenamiento
    start_time = time.time()
    start_datetime_continue = datetime.now()

    # Cargar el modelo pre-entrenado
    model = PPO.load("ppo_gradius_model")

    print(f"Hora de inicio original: {start_datetime_original}")
    print(f"Hora de fin original: {end_datetime_original}")
    print(f"Tiempo total de entrenamiento previo: {total_time_original:.2f} minutos")

    print(f"Hora de inicio del entrenamiento de continuación: {start_datetime_continue.strftime('%Y-%m-%d %H:%M:%S')}")

    # Crear el entorno de entrenamiento
    env_num = 6
    env = SubprocVecEnv([lambda idx=i: make_env() for i in range(env_num)])

    # Crear el callback para recolectar datos durante el entrenamiento
    eval_callback = EvalCallback(start_time=start_time, verbose=1, log_interval=5000, csv_filename="training_data_continue.csv")

    # Continuar el entrenamiento del modelo con el callback
    model.learn(total_timesteps=2712571, log_interval=1, callback=eval_callback)

    # Guardar el modelo entrenado
    model.save("ppo_gradius_model_continue")
    print("Modelo continuado y guardado como 'ppo_gradius_model_continue'.")

    # Guardar los logs del nuevo entrenamiento
    end_datetime_continue = datetime.now()  # Hora de finalización de la continuación
    total_time_continue = (time.time() - start_time) / 60  # Tiempo de la continuación

    log_filename = "ppoTrainLog_continue.txt"
    with open(log_filename, "a") as log_file:  # Agregar al final del archivo de log
        log_file.write(f"Hora de fin de la continuación: {end_datetime_continue.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Tiempo de entrenamiento de la continuación: {total_time_continue:.2f} minutos\n")

    # Mostrar los resultados
    print(f"Hora de fin de la continuación: {end_datetime_continue.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tiempo de la continuación: {total_time_continue:.2f} minutos")

    # Unificar puntajes y tiempos
    max_score_previous = eval_callback.max_score
    print(f"Puntaje máximo previo: {max_score_previous}")
    print(f"Puntaje máximo actual: {eval_callback.max_score}")

    # Guardar los logs de ambos entrenamientos
    with open(log_filename, "a") as log_file:
        log_file.write(f"Puntaje máximo previo: {max_score_previous}\n")
        log_file.write(f"Puntaje máximo actual: {eval_callback.max_score}\n")

    env.close()

if __name__ == "__main__":
    main()
