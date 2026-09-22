# RAG 学习与开发项目

这是一个从简单基线开始、逐步加入来源追踪和评测的资料问答助手学习项目。
当前支持文档读取、带来源的切分、关键词检索，以及 NumPy 向量索引和离线检索对比。
向量来源目前是哈希测试替身，不调用真实模型，也不需要密钥。

## 环境

- Windows PowerShell
- Python 3.12+
- Git

## 从干净环境开始

在项目根目录 `E:\rag` 执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

如果 PowerShell 不允许激活虚拟环境，直接使用上面的 Python 路径即可；不需要修改系统执行策略。

## 运行演示

```powershell
.\.venv\Scripts\rag-demo.exe
.\.venv\Scripts\rag-demo.exe --question "这个项目现在能做什么？"
```

演示会明确显示当前尚未接入模型；使用下面的子命令运行实际检索。

## 文档检索命令

```powershell
.\.venv\Scripts\rag-demo.exe docs              # 列出已加载的示例文档
.\.venv\Scripts\rag-demo.exe chunks            # 查看切分结果（片段 id、原文位置）
.\.venv\Scripts\rag-demo.exe chunks --doc doc_003 --limit 10   # 只看某份文档的片段
.\.venv\Scripts\rag-demo.exe search "浇水"      # 关键词检索，返回 Top-3 片段和来源
.\.venv\Scripts\rag-demo.exe search "徒步 装备" -k 5   # 指定返回前几个片段
.\.venv\Scripts\rag-demo.exe vector-index             # 建立离线 NumPy 向量索引
.\.venv\Scripts\rag-demo.exe vector-search "登山杖"   # 使用余弦相似度检索
.\.venv\Scripts\rag-demo.exe compare                  # 比较 10 个固定问题
```

`search` 是**简单关键词基线**：按字面匹配计分（大小写不敏感），同义词查不到；
片段保留原文字符区间（char_start–char_end），可用 `content[start:end]` 定位回原文。
切分参数在 `src/rag_assistant/config.py` 中配置，当前为 `chunk_size=500, overlap=50`。
500 是基础块的字符上限，补重叠后最多 550 字符。字符数不是模型的 token 数。

## 步骤 4：向量检索基线

`vector-index` 会读取当前资料、切分片段、生成 NumPy 向量，并在 `data/index/` 保存
`index.npz`（内含向量矩阵、JSON 元数据和来源映射）。一次替换整个文件，避免写入中断时
向量和片段错配。元数据包含模型标识、维度、输入配置指纹、切分参数/算法版本和资料版本。
资料新增、修改、删除，或模型/维度/输入配置/切分参数变化时，
`vector-search` 会拒绝使用旧索引并要求重新运行 `vector-index`。
小数据集采用全量重建并替换，不向旧索引追加；重复建立不会积累重复片段。
开发中旧的 `vectors.npy`、`metadata.json` 不再使用，当前只读取 `index.npz`。

当前默认 provider 是 `offline-hash-char-ngram-v2`。它只是确定性的离线测试替身，
用于验证批量向量化、余弦相似度、持久化和兼容性检查，不理解真正的语义或同义词。
真实服务商尚未选择，因此没有猜测模型名称，也没有真实 API 效果可报告。
真实接入前需要在本地配置 `RAG_EMBEDDING_PROVIDER`、官方文档确认的模型名、
API key 和服务地址，并实现对应适配器；变量名见 `.env.example`。

模型参数修改位置：本地 `.env`。程序现在会读取该文件，进程环境变量优先。
从干净仓库开始时，可以在 PowerShell 执行以下命令创建配置（已有文件时不会覆盖）：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

当前有效参数是 `RAG_EMBEDDING_PROVIDER=offline`、`RAG_OFFLINE_DIMENSION=128`、
`RAG_OFFLINE_NGRAM_MIN=2` 和 `RAG_OFFLINE_NGRAM_MAX=3`。修改后三条中的任一条都需重建索引。
预留的真实模型参数尚不能用于 API 调用；配置了真实服务商时会明确报错，不会暗中使用替身。
新增依赖仅有 NumPy（矩阵与余弦运算）和 python-dotenv（解析本地配置）。

运行 `compare` 可对 `data/eval/retrieval_questions.json` 中的 10 个问题比较关键词
和离线向量结果。两种检索使用同一批片段、原始查询和 Top-k。
有答案题要求命中片段完整覆盖标注证据的原文范围；仅命中文档名不算。
无答案题单独统计空结果，这不是回答正确率。标签标为待人工核对，真实语义效果未测。
逐题结果、来源范围、分数和各方法耗时保存到 `reports/retrieval-comparison.json`，被 Git 忽略。
当前向量 Top-k 不实现拒答阈值，即使全部片段都不相关也可能返回结果。
相似度只是向量方向的相近程度，不能直接解释为答案正确概率。

更详细的执行路径、验收结果及局限见 [步骤 4 学习记录](docs/vector-retrieval.md)。

## 运行测试

```powershell
.\.venv\Scripts\python.exe -m pytest
```

测试是离线的，不会访问网络，也不会读取真实密钥。

## 项目结构

```text
src/rag_assistant/  可安装的 Python 包
tests/              自动化测试
data/sample/        可公开提交的示例资料
docs/               学习笔记和实验记录
```

本地配置放在未提交的 `.env` 中，变量名参考 `.env.example`。私人资料和本地索引应放在被 `.gitignore` 排除的位置。

## Git

查看工作区：

```powershell
git status
git diff
```

本步只准备本地仓库，不推送远程 GitHub。提交前请再次检查暂存区中没有 `.env`、私人资料或本地索引。
