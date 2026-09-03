import cv2
import time
import logging
from modules.mpu.vision import CameraStreamer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Локальное тестирование модуля на ПК/роботе ---
streamer: CameraStreamer | None = None
try:
    # Инициализируем и сразу запускаем (start сам дождется первого кадра)
    streamer = CameraStreamer(camera_index=0, width=640, height=480, fps=30).run()
    
    print("Нажмите 'q' в окне видео для выхода.")
    
    # Эмуляция главного цикла (Main Loop) программы
    while True:
        # Делаем copy=True, чтобы безопасно накладывать графику
        flag, frame = streamer.read(copy=True)
        
        if flag and frame is not None:
            # Эмуляция задержки отработки нейросети (например, YOLO)
            time.sleep(0.05) 
            
            cv2.imshow("Robot Camera Stream Test", frame)
        
        # Ждем 1 мс. Если нажата 'q' - прерываем цикл
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except Exception as e:
    logging.error(f"Ошибка при выполнении: {e}")
finally:
    if streamer:
        streamer.stop()
    cv2.destroyAllWindows()