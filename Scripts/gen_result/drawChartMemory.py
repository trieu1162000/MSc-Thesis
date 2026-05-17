import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu cho biểu đồ
models = [
    "Mô hình gốc", "Mô hình gốc kết hợp lượng tử hóa"
]
ram_usage = [2544.544, 1287.012]
stm32_rom_limit = 512  # Giới hạn ROM STM32H743 (kB)

# Tạo biểu đồ
fig, ax = plt.subplots(figsize=(10, 6))
x_positions = np.arange(len(models))
bars = ax.bar(x_positions, ram_usage, color='skyblue', edgecolor='black')

# Thêm đường giới hạn ROM
ax.axhline(y=stm32_rom_limit, color='red', linestyle='--', label='STM32H743 RAM D1')

# Đặt nhãn và tiêu đề
ax.set_xticks(x_positions)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylabel("kBytes")
ax.set_title("Mức tiêu thụ RAM")
ax.legend()

# Hiển thị giá trị trên mỗi cột
for bar, value in zip(bars, ram_usage):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f'{value:.2f}', ha='center', va='bottom', fontsize=9)

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()

