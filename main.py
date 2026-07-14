import os
import numpy as np
from PIL import Image as PILImage
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.clock import Clock

# Пути к медиафайлам на Android
DOWNLOADS_DIR = "/storage/emulated/0/Download"
PICTURES_DIR = "/storage/emulated/0/Pictures"
DCIM_DIR = "/storage/emulated/0/DCIM"

if not os.path.exists(DOWNLOADS_DIR):
    DOWNLOADS_DIR = "./"
    PICTURES_DIR = "./"
    DCIM_DIR = "./"

# --- ПОЛНОЦЕННЫЙ НЕЙРОСЕТЕВОЙ ЯДЕРНЫЙ ДВИЖОК С ПОДДЕРЖКОЙ ДАТАСЕТА ---

class HyperGAMPoseNetwork:
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=128*128*3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 1. He-Initialization (He-Etiquette) для базовых весов
        self.W1_base = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1_base = np.zeros((1, hidden_dim))
        self.W2_base = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2_base = np.zeros((1, output_dim))
        
        # Batch Normalization
        self.gamma = np.ones((1, hidden_dim))
        self.beta = np.zeros((1, hidden_dim))
        
        # 2. LoRA-Позёр (ранг r=4)
        self.r = 4
        self.lora_A = np.random.randn(hidden_dim, self.r) * np.sqrt(2.0 / hidden_dim)
        self.lora_B = np.zeros((self.r, output_dim))
        
        # 3. Adam-Оптимизатор
        self.m_lA, self.v_lA = np.zeros_like(self.lora_A), np.zeros_like(self.lora_A)
        self.m_lB, self.v_lB = np.zeros_like(self.lora_B), np.zeros_like(self.lora_B)
        self.beta1, self.beta2 = 0.9, 0.999
        self.eps = 1e-8
        self.t = 0
        
        self.loss_history = []
        self.dataset = []  # Хранилище для картинок из галереи
        
        # Опорные точки латентного пространства
        self.seed_A = np.random.randn(1, self.input_dim)
        self.seed_B = np.random.randn(1, self.input_dim)

    def forward(self, x, pose_modifiers=None, use_dropout=True, dropout_rate=0.5):
        if pose_modifiers is not None:
            pose_shift = np.tile(pose_modifiers, (self.r, 1))[:self.r, :self.output_dim]
            W_effective = self.W2_base + np.dot(self.lora_A, self.lora_B + pose_shift)
        else:
            W_effective = self.W2_base + np.dot(self.lora_A, self.lora_B)

        # Слой 1
        self.z1 = np.dot(x, self.W1_base) + self.b1_base
        
        # Batch Normalization
        mean = np.mean(self.z1, axis=0, keepdims=True)
        var = np.var(self.z1, axis=0, keepdims=True)
        self.z1_hat = (self.z1 - mean) / np.sqrt(var + self.eps)
        self.bn_out = self.gamma * self.z1_hat + self.beta
        
        # Leaky ReLU
        self.a1 = np.where(self.bn_out > 0, self.bn_out, self.bn_out * 0.2)
        
        # Dropout
        if use_dropout:
            self.dropout_mask = (np.random.rand(*self.a1.shape) >= dropout_rate).astype(np.float32) / (1.0 - dropout_rate)
            self.a1 = self.a1 * self.dropout_mask

        # Слой 2
        out = np.dot(self.a1, W_effective) + self.b2_base
        return np.tanh(out)

    def train_step(self, x, y_target, base_lr=0.001, label_smoothing=0.1, dropout_rate=0.5, max_steps=200):
        self.t += 1
        
        # Cosine Annealing Learning Rate Decay
        lr_min = base_lr * 0.01
        lr = lr_min + 0.5 * (base_lr - lr_min) * (1.0 + np.cos(np.pi * min(self.t, max_steps) / max_steps))
        
        out = self.forward(x, use_dropout=True, dropout_rate=dropout_rate)
        
        # Label Smoothing (сглаживание меток)
        smoothed_target = y_target * (1.0 - label_smoothing) + 0.5 * label_smoothing
        
        loss = np.mean((out - smoothed_target) ** 2)
        self.loss_history.append(float(loss))
        
        loss_grad = out - smoothed_target
        d_out = loss_grad * (1.0 - out ** 2)
        
        grad_lora_B = np.dot(self.lora_A.T, np.dot(self.a1.T, d_out))
        grad_lora_A = np.dot(self.a1.T, np.dot(d_out, self.lora_B.T))
        
        # Gradient Clipping (стабилизация градиентов)
        np.clip(grad_lora_A, -1.0, 1.0, out=grad_lora_A)
        np.clip(grad_lora_B, -1.0, 1.0, out=grad_lora_B)
        
        # Adam обновление
        self.m_lA = self.beta1 * self.m_lA + (1 - self.beta1) * grad_lora_A
        self.v_lA = self.beta2 * self.v_lA + (1 - self.beta2) * (grad_lora_A ** 2)
        m_hat_A = self.m_lA / (1.0 - self.beta1 ** self.t)
        v_hat_A = self.v_lA / (1.0 - self.beta2 ** self.t)
        self.lora_A -= lr * m_hat_A / (np.sqrt(v_hat_A) + self.eps)
        
        self.m_lB = self.beta1 * self.m_lB + (1 - self.beta1) * grad_lora_B
        self.v_lB = self.beta2 * self.v_lB + (1 - self.beta2) * (grad_lora_B ** 2)
        m_hat_B = self.m_lB / (1.0 - self.beta1 ** self.t)
        v_hat_B = self.v_lB / (1.0 - self.beta2 ** self.t)
        self.lora_B -= lr * m_hat_B / (np.sqrt(v_hat_B) + self.eps)
        
        return loss, lr

    # --- ЗАГРУЗКА ИЗОБРАЖЕНИЙ ИЗ ГАЛЕРЕИ ---
    def load_gallery_dataset(self):
        self.dataset = []
        target_dirs = [PICTURES_DIR, DCIM_DIR, DOWNLOADS_DIR]
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        for directory in target_dirs:
            if not os.path.exists(directory):
                continue
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        full_path = os.path.join(root, file)
                        try:
                            # Загрузка через Pillow
                            img = PILImage.open(full_path).convert('RGB')
                            # Масштабируем до 128x128
                            img = img.resize((128, 128), PILImage.Resampling.LANCZOS)
                            img_data = np.array(img, dtype=np.float32)
                            
                            # Нормализация в диапазон [-1, 1]
                            img_flat = (img_data / 127.5) - 1.0
                            img_flat = img_flat.flatten()
                            
                            self.dataset.append(img_flat)
                            
                            # Аугментация: горизонтальный флип (отражение)
                            flipped_img = np.fliplr(img_data)
                            flipped_flat = ((flipped_img / 127.5) - 1.0).flatten()
                            self.dataset.append(flipped_flat)
                            
                        except Exception:
                            continue  # Пропускаем поврежденные файлы
                            
        return len(self.dataset) // 2  # Возвращаем количество уникальных картинок

    def save_weights(self, filename):
        path = os.path.join(DOWNLOADS_DIR, filename)
        np.savez(
            path, 
            W1=self.W1_base, b1=self.b1_base, 
            W2=self.W2_base, b2=self.b2_base,
            lora_A=self.lora_A, lora_B=self.lora_B,
            gamma=self.gamma, beta=self.beta
        )
        with open(path.replace('.npz', '_loss.txt'), 'w') as f:
            f.write("\n".join(map(str, self.loss_history)))
        
    def load_weights(self, filename):
        path = os.path.join(DOWNLOADS_DIR, filename)
        if os.path.exists(path):
            data = np.load(path)
            self.W1_base = data['W1']
            self.b1_base = data['b1']
            self.W2_base = data['W2']
            self.b2_base = data['b2']
            self.lora_A = data['lora_A']
            self.lora_B = data['lora_B']
            self.gamma = data['gamma']
            self.beta = data['beta']
            return True
        return False


# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ KIVY ---

class GANPoseApp(App):
    def build(self):
        self.gan = HyperGAMPoseNetwork()
        self.is_training = False
        
        base_model_file = "base_gan.npz"
        if os.path.exists(os.path.join(DOWNLOADS_DIR, base_model_file)):
            self.gan.load_weights(base_model_file)

        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=6)
        
        # Изображение
        self.img_widget = Image(size_hint_y=0.35)
        main_layout.add_widget(self.img_widget)
        
        # Монитор
        self.status_label = Label(
            text="Loss: 0.0000 | Найдено картинок: 0 | Ожидание...", 
            size_hint_y=0.06, 
            color=[1, 0.4, 0.7, 1]
        )
        main_layout.add_widget(self.status_label)
        
        # --- Слайдеры управления позой ---
        sliders_layout = BoxLayout(orientation='vertical', size_hint_y=0.18, spacing=4)
        
        self.slider_tension = Slider(min=-0.5, max=0.5, value=0, step=0.01)
        self.slider_tension_lbl = Label(text="Связывание / Натяжение: 0.00", size_hint_y=0.3)
        self.slider_tension.bind(value=self.on_slider_change)
        sliders_layout.add_widget(self.slider_tension_lbl)
        sliders_layout.add_widget(self.slider_tension)
        
        self.slider_angle = Slider(min=-0.5, max=0.5, value=0, step=0.01)
        self.slider_angle_lbl = Label(text="Наклон тела GF: 0.00", size_hint_y=0.3)
        self.slider_angle.bind(value=self.on_slider_change)
        sliders_layout.add_widget(self.slider_angle_lbl)
        sliders_layout.add_widget(self.slider_angle)
        
        main_layout.add_widget(sliders_layout)

        # --- Слайдер интерполяции и цветовой гаммы ---
        interp_layout = BoxLayout(orientation='vertical', size_hint_y=0.18, spacing=4)
        
        self.slider_interp = Slider(min=0.0, max=1.0, value=0.0, step=0.02)
        self.slider_interp_lbl = Label(text="Анимация позы (Latent Blend): 0.00", size_hint_y=0.3)
        self.slider_interp.bind(value=self.on_interp_change)
        interp_layout.add_widget(self.slider_interp_lbl)
        interp_layout.add_widget(self.slider_interp)
        
        self.slider_palette = Slider(min=0.0, max=1.0, value=0.4, step=0.05)
        self.slider_palette_lbl = Label(text="Блокировка палитры GF: 40%", size_hint_y=0.3)
        self.slider_palette.bind(value=self.on_slider_change)
        interp_layout.add_widget(self.slider_palette_lbl)
        interp_layout.add_widget(self.slider_palette)
        
        main_layout.add_widget(interp_layout)
        
        # Спиннер выбора моделей
        self.pose_spinner = Spinner(
            text='Выбрать позу (LoRA) из Download',
            values=self.get_downloaded_poses(),
            size_hint_y=0.06
        )
        self.pose_spinner.bind(text=self.on_pose_selected)
        main_layout.add_widget(self.pose_spinner)
        
        # Панель управления (Кнопки)
        btn_layout = BoxLayout(size_hint_y=0.17, spacing=8)
        
        btn_gallery = Button(text="Загрузить Галерею")
        btn_gallery.bind(on_press=self.load_dataset_button)
        btn_layout.add_widget(btn_gallery)
        
        btn_gen = Button(text="Новые Семена")
        btn_gen.bind(on_press=self.randomize_seeds)
        btn_layout.add_widget(btn_gen)
        
        self.btn_train = Button(text="Старт Обучения")
        self.btn_train.bind(on_press=self.toggle_training)
        btn_layout.add_widget(self.btn_train)
        
        main_layout.add_widget(btn_layout)
        
        self.generate_art(None)
        return main_layout

    def on_slider_change(self, instance, value):
        self.slider_tension_lbl.text = f"Связывание / Натяжение: {self.slider_tension.value:.2f}"
        self.slider_angle_lbl.text = f"Наклон тела GF: {self.slider_angle.value:.2f}"
        self.slider_palette_lbl.text = f"Блокировка палитры GF: {int(self.slider_palette.value * 100)}%"
        self.generate_art(None)

    def on_interp_change(self, instance, value):
        self.slider_interp_lbl.text = f"Анимация позы (Latent Blend): {self.slider_interp.value:.2f}"
        self.generate_art(None)

    def randomize_seeds(self, instance):
        self.gan.seed_A = np.random.randn(1, self.gan.input_dim)
        self.gan.seed_B = np.random.randn(1, self.gan.input_dim)
        self.generate_art(None)
        self.update_status("Сгенерированы новые опорные кадры!")

    def load_dataset_button(self, instance):
        self.status_label.text = "Сканирование галереи..."
        num_images = self.gan.load_gallery_dataset()
        if num_images > 0:
            self.status_label.text = f"Успех! Загружено {num_images} картинок GF (+ дубликаты-отражения)."
        else:
            self.status_label.text = "Картинки не найдены. Положи рисунки GF в папку Download или Pictures!"

    def get_downloaded_poses(self):
        try:
            files = [f for f in os.listdir(DOWNLOADS_DIR) if f.endswith('.npz') and f != "base_gan.npz"]
            return files if files else ["Нет файлов в Download"]
        except Exception:
            return ["Разрешите доступ к памяти"]

    def on_pose_selected(self, spinner, text):
        if text not in ["Нет файлов в Download", "Разрешите доступ к памяти"]:
            if self.gan.load_weights(text):
                self.generate_art(None)
                self.update_status(f"Поза {text} загружена!")

    def update_status(self, custom_msg=""):
        last_loss = f"{self.gan.loss_history[-1]:.4f}" if self.gan.loss_history else "0.0000"
        self.status_label.text = f"Loss: {last_loss} | Датасет: {len(self.gan.dataset)//2} изобр. | {custom_msg}"

    def apply_gf_palette(self, rgb_array, strength):
        if strength == 0:
            return rgb_array
        palette = np.array([
            [190, 20, 40],    # Красный (каноничное платье)
            [85, 40, 25],     # Коричневый (волосы)
            [245, 200, 175],  # Кожа
            [20, 15, 25]      # Темный фон
        ]) / 255.0

        pixels = rgb_array.reshape(-1, 3)
        distances = np.linalg.norm(pixels[:, None, :] - palette[None, :, :], axis=2)
        closest_color_idx = np.argmin(distances, axis=1)
        target_colors = palette[closest_color_idx]
        blended = (1.0 - strength) * pixels + strength * target_colors
        return blended.reshape(rgb_array.shape)

    def generate_art(self, instance):
        alpha = self.slider_interp.value
        interpolated_latent = (1.0 - alpha) * self.gan.seed_A + alpha * self.gan.seed_B
        
        pose_modifiers = np.zeros(self.gan.output_dim)
        pose_modifiers[:self.gan.output_dim // 2] = self.slider_tension.value
        pose_modifiers[self.gan.output_dim // 2:] = self.slider_angle.value
        
        raw_pixels = self.gan.forward(interpolated_latent, pose_modifiers=pose_modifiers, use_dropout=False)
        
        pixels_normalized = (raw_pixels - raw_pixels.min()) / (raw_pixels.max() - raw_pixels.min() + 1e-8)
        pixels_colored = self.apply_gf_palette(pixels_normalized, self.slider_palette.value)
        pixels_final = (pixels_colored * 255).astype(np.uint8)
        
        texture = Texture.create(size=(128, 128), colorfmt='rgb')
        texture.blit_buffer(pixels_final.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.img_widget.texture = texture

    def toggle_training(self, instance):
        if self.is_training:
            self.is_training = False
            self.btn_train.text = "Старт Обучения"
            Clock.unschedule(self.train_loop_step)
            self.update_status("Обучение остановлено!")
        else:
            self.is_training = True
            self.btn_train.text = "Остановить"
            Clock.schedule_interval(self.train_loop_step, 0.08)

    def train_loop_step(self, dt):
        if not self.is_training:
            return False
            
        noise = np.random.randn(1, self.gan.input_dim)
        
        # Выбираем цель для обучения: реальное изображение или случайный шум
        if len(self.gan.dataset) > 0:
            # Берём случайный кадр из загруженного датасета твоих рисунков
            idx = np.random.randint(0, len(self.gan.dataset))
            target_pose = self.gan.dataset[idx].reshape(1, -1)
        else:
            # Резервный режим (если датасет пуст)
            target_pose = np.random.rand(1, self.gan.output_dim)
        
        current_loss, current_lr = self.gan.train_step(
            noise, 
            target_pose, 
            base_lr=0.001, 
            dropout_rate=0.5,
            max_steps=200
        )
        
        # Сохранение весов
        active_pose = self.pose_spinner.text
        if active_pose in ["Выбрать позу (LoRA) из Download", "Нет файлов в Download", "Разрешите доступ к памяти"]:
            active_pose = "gallery_trained_lora.npz"
            
        self.gan.save_weights(active_pose)
        self.status_label.text = f"Loss: {current_loss:.4f} | Кадров в памяти: {len(self.gan.dataset)//2} | Обучение..."
        
        if self.gan.t % 5 == 0:
            self.generate_art(None)

if __name__ == '__main__':
    GANPoseApp().run()
