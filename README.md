# RAG 学习与开发项目

这是一个从简单基线开始、逐步加入来源追踪和评测的资料问答助手学习项目。
当前版本只包含可运行的 Python 包和离线演示，不调用模型服务，也不需要密钥。

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

演示会明确显示当前尚未接入模型。后续步骤会在此入口之外增加文档导入、检索和回答模块。

## 文档检索命令

```powershell
.\.venv\Scripts\rag-demo.exe docs              # 列出已加载的示例文档
.\.venv\Scripts\rag-demo.exe chunks            # 查看切分结果（片段 id、原文位置）
.\.venv\Scripts\rag-demo.exe chunks --doc doc_003 --limit 10   # 只看某份文档的片段
.\.venv\Scripts\rag-demo.exe search "浇水"      # 关键词检索，返回 Top-3 片段和来源
.\.venv\Scripts\rag-demo.exe search "徒步 装备" -k 5   # 指定返回前几个片段
```

当前检索是**简单关键词基线**：按字面匹配计分（大小写不敏感），同义词查不到；
片段保留原文字符区间（char_start–char_end），可用 `content[start:end]` 定位回原文。
切分参数（chunk_size、overlap）在 `src/rag_assistant/chunking.py` 中可配置。

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
