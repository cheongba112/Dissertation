# 数据集要求
- 多人物、多年龄
- 每个人物至少两张图片
- 每个图片带有年龄标签
  
# 数据导入逻辑
### get_dataset.py
首先函数将文件夹中文件名使用dict（其实就是hashmap）进行存储，提取出age标签  
再根据dict将两个文件名和一个age label存入list  
Dataset类中的getitem函数不再加载图片，留到training或test中加载，减少Dataset内存  
  


