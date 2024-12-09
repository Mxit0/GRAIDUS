import retro
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import csv
import os
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

matplotlib.use('Agg')  # Para usar un backend sin necesidad de GUI


# Crear el entorno
def make_env():
    """Crea y configura el entorno de GradiusIII-Snes."""
    env = retro.make(game="GradiusIII-Snes")
    env = WarpFrame(env)  # Preprocesamiento de las imágenes
    env = ClipRewardEnv(env)  # Recorta las recompensas
    return env


# Leer datos del archivo de log
import re
from datetime import datetime

def read_training_log():
    """Leer el archivo de log para obtener los datos del entrenamiento previo y acumulado."""
    log_filename = "ppoTrainLog.txt"
    if os.path.exists(log_filename):
        with open(log_filename, "r") as log_file:
            lines = log_file.readlines()

        # Imprimir las líneas leídas para depurar
        print("Contenido del archivo de log:")
        for line in lines:
            print(f"'{line.strip()}'")

        try:
            # Asegurarse de que las líneas mínimas existen
            if len(lines) < 3:
                print("El archivo de log no tiene suficientes datos.")
                return None, None, None

            # Procesar bloque principal (inicio, fin, tiempo total)
            print("Procesando bloque principal...")
            
            # Usamos regex para extraer las fechas y el tiempo total
            start_line = lines[0].strip()
            end_line = lines[1].strip()
            total_time_line = lines[2].strip()

            # Extraer la fecha de inicio con regex
            start_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", start_line)
            if start_match:
                start_datetime_original = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S")
            else:
                print("Error: No se encontró la fecha de inicio.")
                return None, None, None

            # Extraer la fecha de fin con regex
            end_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", end_line)
            if end_match:
                end_datetime_original = datetime.strptime(end_match.group(1), "%Y-%m-%d %H:%M:%S")
            else:
                print("Error: No se encontró la fecha de fin.")
                return None, None, None

            # Extraer el tiempo total con regex
            total_time_match = re.search(r"(\d+\.\d+)", total_time_line)
            if total_time_match:
                total_time_original = float(total_time_match.group(1))
            else:
                print("Error: No se encontró el tiempo total.")
                return None, None, None

            # Imprimir los valores obtenidos del bloque principal
            print(f"Inicio: {start_datetime_original}")
            print(f"Fin: {end_datetime_original}")
            print(f"Duración total: {total_time_original}")

            # Procesar continuaciones
            print("Procesando continuaciones...")
            for line in lines[3:]:
                if line.startswith("Continuación:"):
                    print(f"Procesando continuación: '{line.strip()}'")
                    parts = line.split(", ")
                    cont_start = datetime.strptime(parts[0].split("inicio ")[1], "%Y-%m-%d %H:%M:%S")
                    cont_end = datetime.strptime(parts[1].split("fin ")[1], "%Y-%m-%d %H:%M:%S")
                    cont_duration = float(parts[2].split("duración ")[1].replace(" minutos.", ""))

                    # Actualizar los valores acumulados
                    end_datetime_original = cont_end  # La última continuación define el final más reciente
                    total_time_original += cont_duration  # Sumar duración acumulada

            return start_datetime_original, end_datetime_original, total_time_original

        except Exception as e:
            print(f"Error al procesar el archivo de log: {e}")
            return None, None, None
    else:
        print("No se encontró el archivo de log anterior.")
        return None, None, None





# Guardar datos de entrenamiento en un CSV
def save_training_data(times, scores, filename="training_data_continue.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (minutes)", "Score"])
        writer.writerows(zip(times, scores))
    print(f"Datos guardados en {filename}")


# Callback para recolectar datos durante el entrenamiento
class EvalCallback(BaseCallback):
    def __init__(self, start_time, verbose=0, log_interval=1000):
        super(EvalCallback, self).__init__(verbose)
        self.max_score = -np.inf
        self.current_score = 0
        self.log_interval = log_interval
        self.start_time = start_time
        self.step_count = 0
        self.times_in_minutes = []
        self.scores = []

    def _on_step(self) -> bool:
        self.step_count += 1
        info = self.locals.get("infos", [])[0]
        self.current_score = info.get('score', 0)
        if self.current_score > self.max_score:
            self.max_score = self.current_score
        elapsed_time_minutes = (time.time() - self.start_time) / 60
        self.times_in_minutes.append(elapsed_time_minutes)
        self.scores.append(self.current_score)
        if self.step_count % self.log_interval == 0:
            elapsed_time = timedelta(seconds=int(time.time() - self.start_time))
            print(f"Tiempo transcurrido: {elapsed_time} - Puntaje actual: {self.current_score} - Máximo: {self.max_score}")
        return True


def main():
    # Leer el archivo de log
    start_datetime_original, end_datetime_original, total_time_original = read_training_log()
    if start_datetime_original:
        print(f"Inicio entrenamiento anterior: {start_datetime_original.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tiempo de entrenamiento anterior: {total_time_original:.2f} minutos.")
    else:
        print("No se encontraro log previo de entrenamiento.")

    # Registrar el tiempo de inicio
    start_time = time.time()
    start_datetime = datetime.now()

    print(f"Inicio entrenamiento actual: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    env_num = 4
    env = SubprocVecEnv([lambda idx=i: make_env() for i in range(env_num)])

    # Cargar el modelo previamente entrenado
    model_path = "ppo_gradius_model.zip"
    if os.path.exists(model_path):
        try:
            model = PPO.load(model_path, env=env)
            print(f"Modelo '{model_path}' cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return
    else:
        print(f"No se encontró el modelo '{model_path}'. Asegúrate de que exista.")
        return

    # Crear callbacks para recolectar datos
    eval_callbacks = [EvalCallback(start_time=start_time, verbose=1, log_interval=5000) for _ in range(env_num)]

    # Continuar el entrenamiento
    model.learn(total_timesteps=1000, log_interval=1, callback=eval_callbacks)

    # Guardar el modelo actualizado
    model.save("ppo_gradius_model_continued.zip")
    print("Modelo actualizado guardado como 'ppo_gradius_model_continued.zip'.")

    # Consolidar datos de entrenamiento
    unified_times = []
    unified_scores = []
    for callback in eval_callbacks:
        unified_times.extend(callback.times_in_minutes)
        unified_scores.extend(callback.scores)

    # Guardar los datos en un archivo CSV
    save_training_data(unified_times, unified_scores, filename="training_data_continue.csv")

    # Graficar los datos
    combined_data = sorted(zip(unified_times, unified_scores))
    sorted_times, sorted_scores = zip(*combined_data)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_times, sorted_scores, label="Score", color="green")
    plt.xlabel('Tiempo de entrenamiento (minutos)')
    plt.ylabel('Puntaje')
    plt.title('Continuación del Entrenamiento: Puntaje vs Tiempo')
    plt.grid(True)
    plt.legend()
    plt.savefig('score_vs_time_continue.png')
    print("Gráfico guardado como 'score_vs_time_continue.png'.")

    # Calcular el tiempo total de esta sesión
    total_time_minutes_session = (time.time() - start_time) / 60
    end_datetime = datetime.now()

    # Tiempo total acumulado
    total_time_accumulated = total_time_original + total_time_minutes_session if total_time_original else total_time_minutes_session

    print(f"Tiempo de entrenamiento de esta sesión: {total_time_minutes_session:.2f} minutos.")
    print(f"Tiempo de entrenamiento total (anterior + actual): {total_time_accumulated:.2f} minutos.")

    # Guardar nuevos logs
    log_filename = "ppoTrainLog_continue.txt"
    with open(log_filename, "a") as log_file:
        log_file.write(f"Continuación: inicio {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}, "
                       f"fin {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}, "
                       f"duración {total_time_minutes_session:.2f} minutos.\n")

    env.close()


if __name__ == "__main__":
    main()
