import json

def split_privacy_train_data(input_path, output_path, start, end):
    """
    从输入文件中提取指定范围的条目保存到输出文件
    
    Args:
        input_path (str): 输入文件路径 (train.jsonl)
        output_path (str): 输出文件路径 (privacy_train.jsonl)
        start (int): 从哪一条数据开始提取
        end (int): 到哪一条数据结束提取
        0-30000是对话数据集
        30000-40000是单选题数据集
        40000-44000是名词解释数据集
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        count = 0
        for line in infile:
            if count < start:
                count += 1
                continue  # 跳过前 `start` 条数据
            
            if count > end:
                break  # 超过结束条目，停止提取
            
            try:
                # 验证JSON格式是否正确
                entry = json.loads(line)
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                print(f"跳过第{count+1}行: 无效的JSON格式")
                continue
            count += 1

if __name__ == "__main__":
    # 定义文件路径
    INPUT_FILE = "train.jsonl"
    OUTPUT_FILE = "privacy_train.jsonl"
    
    # 直接指定起始和结束条目的数字
    start = 0  # 设置起始条目
    end = 30000    # 设置结束条目
    
    # 确保起始条目小于结束条目
    if start > end:
        print("起始条目不能大于结束条目，请重新设置。")
        exit(1)
    
    # 提取指定范围的数据
    split_privacy_train_data(INPUT_FILE, OUTPUT_FILE, start, end)
    print(f"已成功提取从第{start + 1}条到第{end + 1}条的数据到{OUTPUT_FILE}文件中")
