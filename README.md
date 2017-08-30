# ChatBot
聊天机器人--从原理到实现

###  运行环境
#### python2.7 + tensorflow0.12
#  Install pip and virtualenv
    sudo apt-get install python-pip python-dev python-virtualenv
#  Create a virtualenv environment
    virtualenv --system-site-packages(targetDirectory), just like:virtualenv ~/tensorflow0.12
#  Activate the virtualenv environment
    source ~/tensorflow0.12/bin/activate
#  Install TensorFlow
    sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl

### 1. 数据获取 
    影视对白训练语料库
    wget https://raw.githubusercontent.com/rustch3n/dgk_lost_conv/master/dgk_shooter_min.conv.zip
    unzip dgk_shooter_min.conv.zip
### 2. 数据预处理
    一般来说，我们拿到的基础语料库可能是一些电影台词对话，或者是UBUNTU对话语料库(Ubuntu Dialog Corpus)，但基本上我们都要完成以下几大步骤
    1. 分词(tokenized)
    2. 将语料分成 Question和Answer部分
    3. 训练集和验证集划分
    4. 创建词汇表，对话转为向量形式
### 3. 训练模型
    训练模型时，注销main函数中的test()
    python chatbot.py
### 4. 测试
    启用test函数测试
