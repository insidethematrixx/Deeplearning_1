import os

base_dir = "val"

source_folder_name = "bees_image"
target_folder_name = "bees_label"

source_path = os.path.join(base_dir, source_folder_name)
target_path = os.path.join(base_dir, target_folder_name)

label_content = source_folder_name.split('_')[0]

os.makedirs(target_path, exist_ok=True)

try:
    image_filenames = sorted([f for f in os.listdir(source_path) if not f.startswith('.')])

    for img_filename in image_filenames:
        file_name_base = os.path.splitext(img_filename)[0]

        txt_filename = f"{file_name_base}.txt"

        output_filepath = os.path.join(target_path, txt_filename)

        with open(output_filepath, 'w') as f:
            f.write(label_content)

    print(f"成功在 '{target_path}' 中创建/更新了 {len(image_filenames)} 个标签文件。")

except FileNotFoundError:
    print(f"【错误】: 找不到源文件夹 '{source_path}'。")
    print("请检查：")
    print("1. base_dir 和 source_folder_name 变量是否正确配置。")
    print("2. 文件结构是否与脚本假设的一致。")