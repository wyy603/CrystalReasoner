import json
from pathlib import Path
from crysreas import Config

def count_json_list_lengths(folder_path):
    # 将字符串路径转换为 Path 对象
    base_path = Path(folder_path)
    
    # 查找文件夹下所有 .json 文件
    json_files = list(base_path.glob("*.json"))
    
    if not json_files:
        print(f"在 {folder_path} 中未找到 json 文件。")
        return

    print(f"{'文件名':<30} | {'test':<10} | {'train':<10} | {'val':<10}")
    print("-" * 70)

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 获取各个 key 的列表长度，如果 key 不存在则返回 0
                test_len = len(data.get("test", []))
                train_len = len(data.get("train", []))
                val_len = len(data.get("val", []))
                
                print(f"{file_path.name:<30} | {test_len:<10} | {train_len:<10} | {val_len:<10}")
        
        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {e}")

if __name__ == "__main__":
    # 替换为你实际的文件夹路径，例如 "./data"
    target_folder = Config.DATA_PATH
    count_json_list_lengths(target_folder)