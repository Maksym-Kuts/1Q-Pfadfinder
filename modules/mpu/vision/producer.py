import cv2
from threading import Thread, Lock, Event
import time
import logging
import numpy as np
from typing import Tuple

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class CameraStreamer:

    """ Потокобезопасный класс для захвата видеопотока с камеры.
        Модуль для компьютерного зрения (CV).
        Обеспечивает автоматическое восстановление соединения с сохранением настроек (разрешение, частота кадров)
        и защиту памяти с помощью `threading.Lock`. """

    def __init__(
            
            self, 
            camera_index: int = 0, 
            width: int = 640, 
            height: int = 480, 
            fps: int = 30

        ) -> None:

        # Инициализация полей
        self.camera_index: int = camera_index
        self.width:        int = width
        self.height:       int = height
        self.fps:          int = fps

        self.current_frame: np.ndarray | None = None # Текущий кадр
        
        self.thread:  Thread           | None = None # Текущий поток
        self.capture: cv2.VideoCapture | None = None # Захват камеры

        self.lock:        Lock = Lock()              # Замок для контроля памяти
        self.is_active:   bool = False               # Флаг: активна ли камера
        self.frame_ready: Event = Event()            # Сигнал готовности кадра
        
        self._init_camera()                          # Инициализируем камеру при создании объекта


    def _init_camera(self)   -> bool:

        """ Инициализация камеры. 
            Перенесена в отдельный метод, чтобы при восстановлении соединения снова применялось требуемое разрешение. """
        
        logging.info(f"Инициализация камеры {self.camera_index}...")
        
        self._clear_capture()                               # Обнуление захватов
            
        self.capture = cv2.VideoCapture(self.camera_index)  # Захват камеры

        # Запрещаем драйверу Linux (V4L2) накапливать кадры в буфере.
        # Гарантирует, что мы всегда получаем картинку "здесь и сейчас", без задержек.
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # ДЛЯ LINUX/ARM64: Запрашиваем MJPG аппаратно.
        # Чтобы камера не отдала сырой YUYV, который забьет USB-шину и обрушит FPS.
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Настройка кадра
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)  # Ширина
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height) # Высота
        self.capture.set(cv2.CAP_PROP_FPS,          self.fps)    # FPS
        
        # Выключаем автофокус (если поддерживается)
        self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        if not self._initialised():
            logging.error(f"Не удалось открыть USB-камеру {self.camera_index}")
            return False
                
        return True

    def _clear_capture(self) -> None:

        """ Защита от утечки файловых дескрипторов (зомби-устройств). """

        if self.capture is not None and self.capture.isOpened():
            self.capture.release()

    def _initialised(self)   -> bool:

        """ Проверка инициализации камеры (True/False). """

        if self.capture is None or not self.capture.isOpened():
            return False
        return True

    def _update_cycle(self)  -> None:

        """ Фоновый процесс (Worker Thread) для считывания кадров с камеры. """

        failed_reconnection: int = 0 # Попытки переподключиться
        
        while self.is_active:

            # Читаем свежий кадр, если камера инициализирована. Иначе False и None
            flag, frame = self.capture.read() if self._initialised() else (False, None)

            if flag:                     # Если свежий кадр получен
                failed_reconnection = 0  # Сбрасываем счетчик при успешном кадре
                with self.lock:          # Блокируем доступ для обновления кадра
                    self.current_frame = frame
                
                # Сообщаем главному потоку, что первый кадр получен (срабатывает только один раз)
                if not self.frame_ready.is_set():
                    self.frame_ready.set()

            else:                        # Логика отказоустойчивости и переподключения
                failed_reconnection += 1 # Увеличиваем счётчик попыток
                if failed_reconnection <= 10:
                    logging.warning(f"Потеряна связь с камерой. Попытка переподключения {failed_reconnection}/10...")
                    time.sleep(1.0)          # Даем USB-шине время на сброс
                    if not self.is_active:   # Если во время сна поток был остановлен
                        break
                    self._init_camera()      # Пробуем переинициализировать
                else:
                    logging.critical("АППАРАТНЫЙ СБОЙ: Камера окончательно потеряна. Остановка видеопотока.")
                    self.is_active = False
                    self._clear_capture()
                    break


    def run(self)                      -> 'CameraStreamer':

        """ Запускает фоновый поток захвата видео и блокирует главный поток 
            до тех пор, пока не будет успешно получен первый кадр (или не истечет время ожидания). """
        
        if self.is_active:
            return self
        
        self.is_active = True
        # daemon=True гарантирует, что зависший поток не помешает ОС убить процесс
        self.thread = Thread(target=self._update_cycle, daemon=True)
        self.thread.start()
        
        logging.info("Ожидание получения первого кадра...")
        # Ждем захвата первого кадра, чтобы не отдать None в основной цикл
        if not self.frame_ready.wait(timeout=12.0):
            self.stop()
            raise TimeoutError("Камера не отдала кадр за 12 секунд. Проверьте подключение/USB.")
            
        logging.info("Видеопоток успешно запущен и готов к работе.")
        return self

    def stop(self)                     -> None:

        """ Безопасно останавливает поток и освобождает аппаратуру. """

        self.is_active = False
        
        if self.thread is not None and self.thread.is_alive(): # Ожидание завершения фоновых потоков
            self.thread.join()
        
        self._clear_capture()
            
        logging.info("Камера аппаратно отключена.")

    def read(self, copy: bool = False) -> Tuple[bool, np.ndarray | None]:

        """ Возвращает статус и самый свежий кадр из буфера.
        
        :param copy: Установите True, если планируете изменять кадр (например, рисовать bbox).
        :return: Кортеж (успех, кадр) """

        with self.lock:  # Блокируем доступ на момент чтения/копирования
            if self.current_frame is not None:
                # Возвращаем копию, если запрошено, чтобы не испортить исходный буфер
                frame_out = self.current_frame.copy() if copy else self.current_frame
                return True, frame_out
            return False, None