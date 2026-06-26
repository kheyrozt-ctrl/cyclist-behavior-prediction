import os
import re
import random
from collections import defaultdict, Counter

# 定义文件夹路径
folder_path = "GestureDataset/depth_camera_videos_croped"

# 定义六种前缀
prefixes = ['1a', '1b', '1c', '1d', '1e', '1g']

# 用于存储每种前缀下的所有字符集合
char_sets = {prefix: set() for prefix in prefixes}

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 检查文件是否以.mp4结尾并且以1开头
    if file_name.endswith('.mp4'):
        # 遍历所有前缀
        for prefix in prefixes:
            # 检查文件名是否以当前前缀开头
            if file_name.startswith(prefix):
                # 使用正则表达式提取前缀后，第一个下划线之后和下一个下划线之间的字符
                match = re.search(rf'{prefix}_([^_]+)_', file_name)
                if match:
                    # 将找到的字符加入对应前缀的集合中
                    char_sets[prefix].add(match.group(1))

# 找到所有集合都包含的字符
common_chars = set.intersection(*char_sets.values())

print(f"六个文件夹都包含的participant ID是: {','.join(sorted(common_chars))}")

# 初始的训练集中需要固定包含的participant ID
fixed_train_ids = common_chars

# 用于存储每种前缀的数据
data = defaultdict(list)

# 创建一个映射，将前缀映射到类别
prefix_to_category = {
    '1a': 'left',
    '1b': 'left',
    '1c': 'right',
    '1d': 'crossing',
    '1e': 'crossing',
    '1g': 'left'
}

# 遍历文件夹中的所有文件
all_files = []
for file_name in os.listdir(folder_path):
    # 检查文件是否以.mp4结尾并且以1开头
    if file_name.endswith('.mp4') and file_name.startswith('1'):
        all_files.append(file_name)
        # 找到文件前缀
        prefix = file_name[:2]
        if prefix in prefixes:
            # 提取前缀后，第一个下划线之后和下一个下划线之间的字符（参与者ID）
            match = re.search(rf'{prefix}_([^_]+)_', file_name)
            if match:
                participant_id = match.group(1)
                # 将数据按类别和参与者ID存储 (包含类别信息)
                data[participant_id].append((file_name, prefix_to_category[prefix]))
print(f"以.mp4结尾并且以1开头的文件总数量：{len(all_files)}")


# 函数：检查当前分配情况是否符合要求
def check_distribution(train, val, test):
    # 计算每个数据集中的类别数量
    train_counts = Counter([category for _, category in train])
    val_counts = Counter([category for _, category in val])
    test_counts = Counter([category for _, category in test])

    train_count = len(train)
    val_count = len(val)
    test_count = len(test)
    
    # 动态设置的类别限制
    left_train_min, left_train_max = int(train_count * 0.25), int(train_count * 0.75)
    crossing_train_min, crossing_train_max = int(train_count * 0.05), int(train_count * 0.55)
    right_train_min, right_train_max = int(train_count * 0.05), int(train_count * 0.4)
    
    left_val_min, left_val_max = int(val_count * 0.25), int(val_count * 0.75)
    crossing_val_min, crossing_val_max = int(val_count * 0.05), int(val_count * 0.55)
    right_val_min, right_val_max = int(val_count * 0.05), int(val_count * 0.4)
    
    left_test_min, left_test_max = int(test_count * 0.3), int(test_count * 0.7)
    crossing_test_min, crossing_test_max = int(test_count * 0.1), int(test_count * 0.5)
    right_test_min, right_test_max = int(test_count * 0.05), int(test_count * 0.4)


    # 检查数量和类别比例条件
    indicator = (
        78 <= train_count <= 92 and 25 <= val_count <= 31 and 25 <= test_count <= 31 and
        left_train_min <= train_counts['left'] <= left_train_max and 
        crossing_train_min <= train_counts['crossing'] <= crossing_train_max and
        right_train_min <= train_counts['right'] <= right_train_max and
        left_val_min <= val_counts['left'] <= left_val_max and 
        crossing_val_min <= val_counts['crossing'] <= crossing_val_max and
        right_val_min <= val_counts['right'] <= right_val_max and
        left_test_min <= test_counts['left'] <= left_test_max and 
        crossing_test_min <= test_counts['crossing'] <= crossing_test_max and
        right_test_min <= test_counts['right'] <= right_test_max
    )
    # indicator = (
    #     78 <= train_count <= 110 and 21 <= val_count <= 36 and 25 <= test_count <= 31
    # )
    
    # if indicator:
    #     print(f"train_counts['left']{train_counts['left']},train_counts['crossing']{train_counts['crossing']},train_counts['right']{train_counts['right']}")
    #     print(f"val_counts['left']{val_counts['left']},val_counts['crossing']{val_counts['crossing']},val_counts['right']{val_counts['right']}")
    #     print(f"test_counts['left']{test_counts['left']},test_counts['crossing']{test_counts['crossing']},test_counts['right']{test_counts['right']},")
    #     print(" ")
    
    
    return indicator

# 初始放入固定的participant ID的视频文件到训练集中
train, val, test = [], [], []

# 将初始固定的participant ID的视频文件放入训练集
for pid in fixed_train_ids:
    if pid in data:
        train.extend(data[pid])
# print(f"将初始固定的participant ID的视频文件放入训练集后训练集的文件数量：{len(train)}")

# 获取剩余的participant IDs
remaining_ids = set(data.keys()) - fixed_train_ids
# print(remaining_ids)

# 分配剩余的participant IDs的文件到验证集和测试集，直到满足条件
def assign_to_val_test(val, test, initial_train, remaining_ids):
    # 多次尝试不同的a和b，直到满足条件
    for a in range(1, len(remaining_ids) + 1):
        for b in range(1, len(remaining_ids) - a + 1):
            # 将剩余的participant IDs随机打乱
            remaining_ids_list = list(remaining_ids)
            random.shuffle(remaining_ids_list)

            # # 分配participant IDs
            # val_ids = set(remaining_ids_list[:a])
            # test_ids = set(remaining_ids_list[a:a + b])
            # train_ids = set(remaining_ids_list[a + b:])


            # 抽取val_ids和test_ids
            val_ids = set(random.sample(remaining_ids_list, a))
            remaining_after_val = list(set(remaining_ids_list) - val_ids)
            test_ids = set(random.sample(remaining_after_val, b))
            train_ids = set(remaining_after_val) - test_ids

            # 清空val和test数据集
            val.clear()
            test.clear()
            train = initial_train.copy()
            
            

            # 分配文件
            for pid in val_ids:
                val.extend(data[pid])
            for pid in test_ids:
                test.extend(data[pid])
            for pid in train_ids:
                train.extend(data[pid])

            # 检查分配结果是否符合条件
            if check_distribution(train, val, test):
                # print(len(initial_train))
                return val, test, train
            


    return val, test, train



# 分配剩余的participant IDs的文件到验证集和测试集，直到满足条件
def assign_to_val(val, test, initial_train, remaining_ids, existing_distributions):
   
    # 多次尝试不同的a和b，直到满足条件
    for a in range(1, len(remaining_ids) + 1):
        for b in range(1, 100):
            # 将剩余的participant IDs随机打乱
            remaining_ids_list = list(remaining_ids)
            random.shuffle(remaining_ids_list)

            # # 分配participant IDs
            # val_ids = set(remaining_ids_list[:a])
            # test_ids = set(remaining_ids_list[a:a + b])
            # train_ids = set(remaining_ids_list[a + b:])

            # 抽取val_ids
            val_ids = set(random.sample(remaining_ids_list, a))
            
            train_ids = set(remaining_ids_list) - val_ids
            # print(len(val_ids),len(train_ids))
            # print(len(train_ids))

            # 清空val数据集
            val.clear()
            train = initial_train.copy()
          
            
            # 分配文件
            for pid in val_ids:
                val.extend(data[pid])
            for pid in train_ids:
                train.extend(data[pid])

            train_counts = Counter([category for _, category in train])
            val_counts = Counter([category for _, category in val])
            test_counts = Counter([category for _, category in test])
            # 当前分布情况
            current_distribution = (train_counts['left'], train_counts['crossing'], train_counts['right'],
                            val_counts['left'], val_counts['crossing'], val_counts['right'])
            

            # 检查分配结果是否符合条件
            if check_distribution(train, val, test) and current_distribution not in existing_distributions:
                existing_distributions.add(current_distribution)
                print(f"满足条件的分配:")
                print(f"train_counts: {train_counts}, val_counts: {val_counts}, test_counts: {test_counts}")
                
                # print(val)
                
                return val, train, existing_distributions
                
            

                    
    return [],  initial_train, existing_distributions

# 分配剩余的participant IDs的文件
val_temp, test, train_temp = assign_to_val_test(val, test, train, remaining_ids)

# 输出最终的文件分配结果
train_temp_files = [f_name for f_name, _ in train_temp]
val_temp_files = [f_name for f_name, _ in val_temp]
test_files = [f_name for f_name, _ in test]

# print(f"训练数据集文件: {train_files}")
# print(f"验证数据集文件: {val_files}")
# print(f"测试数据集文件: {test_files}")

print(f"训练数据集总数: {len(train_temp)}, 验证数据集总数: {len(val_temp)}, 测试数据集总数: {len(test)}")

# 将文件名结果写入文件中
with open('Gesture2Manuever_Prediction/VideoCrop/train_files.txt', 'w') as f:
    f.write("\n".join(train_temp_files))

with open('Gesture2Manuever_Prediction/VideoCrop/val_files.txt', 'w') as f:
    f.write("\n".join(val_temp_files))

with open('Gesture2Manuever_Prediction/VideoCrop/test_files.txt', 'w') as f:
    f.write("\n".join(test_files))

print("文件名结果已保存到文件中：train_files.txt, val_files.txt, test_files.txt")


# 记录已有的分布情况
existing_distributions = set()

# 锁定test数据集中的participant IDs
locked_test_ids = set()

# 遍历测试数据集，提取其中的participant IDs
for file_name, _ in test:
    # 从文件名中提取participant ID
    match = re.search(r'1[a-g]_([^_]+)_', file_name)
    if match:
        participant_id = match.group(1)
        locked_test_ids.add(participant_id)

print(f"锁定的测试集participant IDs: {locked_test_ids}")
print(" ")

# 准备5组train和val数据集
folds = []
remaining_ids_for_folds = remaining_ids - locked_test_ids
print(remaining_ids_for_folds)
for i in range(5):

    
    fold_val, fold_train, existing_distributions = assign_to_val(val, test, train, remaining_ids_for_folds, existing_distributions)

    train_files = [f_name for f_name, _ in fold_train]
    val_files = [f_name for f_name, _ in fold_val]
    test_files = [f_name for f_name, _ in test]
    
    with open(f'Gesture2Manuever_Prediction/VideoCrop/fold_{i+1}_train_files.txt', 'w') as f:
        f.write("\n".join(train_files))

    with open(f'Gesture2Manuever_Prediction/VideoCrop/fold_{i+1}_val_files.txt', 'w') as f:
        f.write("\n".join(val_files))

    with open(f'Gesture2Manuever_Prediction/VideoCrop/fold_{i+1}_test_files.txt', 'w') as f:
        f.write("\n".join(test_files))

    print(f"Fold {i+1} 文件名结果已保存到文件中：fold_{i+1}_train_files.txt, fold_{i+1}_val_files.txt, fold_{i+1}_test_files.txt")
    print(f"训练数据集总数: {len(fold_train)}, 验证数据集总数: {len(fold_val)}, 测试数据集总数: {len(test)}")
    print(" ")

