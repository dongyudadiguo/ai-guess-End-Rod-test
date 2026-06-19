# AI 猜物品智力测试

通过"猜 Minecraft 物品"游戏测试 AI 模型的推理能力。模型同时扮演猜方和答方，通过 是/否 提问逐步缩小范围，最终猜测目标物品。支持多模型对比，生成统一的结构化结果文件。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
set OPENAI_API_KEY=sk-xxx        # Windows
export OPENAI_API_KEY=sk-xxx     # Linux/Mac

# 3. 编辑 config.json，填写你的模型配置

# 4. 运行
python run_test.py
```

## 配置文件 (config.json)

```jsonc
{
    "models": [
        {
            "name": "GPT-4o",                    // 自定义显示名称
            "base_url": "https://api.openai.com/v1",
            "api_key": "env:OPENAI_API_KEY",     // env:变量名 或直接写密钥
            "model": "gpt-4o",                   // API model 参数
            "temperature_guesser": 0.7,           // 猜方创造性 (建议 0.7)
            "temperature_answerer": 0.0,          // 答方确定性 (建议 0)
            "max_tokens": 512,
            "timeout_seconds": 60
        }
        // 添加更多模型...
    ],

    "test": {
        "target_item": "End Rod",     // 目标物品 (英文)
        "target_item_cn": "末地烛",    // 目标物品 (中文)
        "rounds": 10,                  // 每个模型测试轮数
        "max_questions": 32            // 每轮最多提问数
    }
}
```

### 常见服务商配置

| 服务商 | base_url | 示例 model |
|--------|----------|-----------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 硅基流动 | `https://api.siliconflow.cn/v1` | `deepseek-ai/DeepSeek-V3` |
| 阿里云 DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Ollama 本地 | `http://localhost:11434/v1` | `qwen2.5:7b` |

## 运行流程

```
  猜方 (Guesser)                          答方 (Answerer)
  ─────────────                           ─────────────
  系统提示: 猜一个Minecraft物品             已知目标: End Rod (末地烛)
      │                                        │
      │  "Is it a block?"  ──────────────────→ 判断 → "Yes"
      │  "Is it in the End?"  ───────────────→ 判断 → "Yes"
      │  "Does it glow?"  ───────────────────→ 判断 → "Yes"
      │  ... (最多32问)                         │
      │  "I GUESS: End Rod"  ────────────────→ 核对 → 正确/错误
```

- **猜方**根据系统提示发起提问，每次一问，基于历史回答逐步推理
- **答方**被告知目标物品，对每个问题给出 Yes/No 判断
- 同一模型的猜方和答方使用不同温度参数（猜方 0.7，答方 0.0）
- 每轮独立运行，不继承上一轮的上下文

## 输出文件

运行后生成一个 JSON 文件：`results_YYYYMMDD_HHMMSS.json`

### 结构概览

```
results_20260619_143000.json
├── meta                        # 测试元信息
│   ├── test_time_utc           # UTC 时间
│   ├── target_item             # 目标物品
│   ├── rounds_per_model        # 每模型轮数
│   ├── max_questions           # 每轮最多问题数
│   └── prompts                 # 使用的提示词
├── comparison[]                # 模型对比汇总
│   ├── model_name              # 模型名称
│   ├── success_rate_pct        # 成功率
│   ├── avg_questions           # 平均问题数
│   ├── avg_time_seconds        # 平均耗时
│   ├── avg_guesser_tokens      # 猜方平均Token
│   ├── avg_answerer_tokens     # 答方平均Token
│   └── round_summaries[]       # 每轮概况文本
└── results[]                   # 各模型完整数据
    └── <模型>
        ├── model_config        # 模型参数快照
        ├── summary             # 汇总统计
        └── rounds[]            # 每轮详情
            └── qa_sequence[]   # 每条问答记录
                ├── question_number
                ├── role                    # qa | guesser_guess | guesser_giveup | guesser_error
                ├── guesser_raw_response    # 猜方 API 原始返回
                ├── answerer_raw_response   # 答方 API 原始返回
                ├── answerer_clean          # 归一化后 Yes/No
                ├── guess_value             # 猜测值 (如果是猜)
                ├── is_correct              # 是否猜对
                ├── guesser_token_usage     # {prompt, completion, total}
                ├── answerer_token_usage
                ├── guesser_finish_reason   # stop / length / ...
                ├── answerer_finish_reason
                ├── timestamp_start/end     # ISO 8601 UTC
                └── error                   # 错误信息 (如API失败)
```

### 终端输出示例

```
============================================================
  AI 猜物品智力测试 - 多模型对比
  目标物品: End Rod (末地烛)
  测试模型数: 2  |  每模型 10 轮  |  最多 32 问/轮
============================================================

------------------------------------------------------------
  模型: GPT-4o  (gpt-4o)
  目标: End Rod (末地烛)  |  10 轮, 每轮最多 32 问
------------------------------------------------------------
  [Round 1/10] ✓  (6问, 18.2s, 猜1200t+答300t)  → End Rod
  [Round 2/10] ✓  (5问, 15.1s, 猜980t+答250t)  → End Rod
  ...
  结果: 8/10 (80.0%)  |  平均 7.2问  |  平均 20.1s

============================================================
  最终对比结果
============================================================
  模型                 成功   成功率   平均问数   平均耗时   猜Token   答Token
  ------------------ ------ -------- -------- -------- -------- --------
  GPT-4o              8/10   80.0%      7.2     20.1s    1100      280
  DeepSeek-V3         6/10   60.0%     12.5     15.3s     850      320
============================================================

结果已保存: results_20260619_143000.json
```

## 自定义目标物品

修改 `config.json` 中的 `target_item` 和对应的别名匹配函数：

```json
"test": {
    "target_item": "Diamond Pickaxe",
    "target_item_cn": "钻石镐"
}
```

若目标在 `run_test.py` 的 `is_guess_correct` 函数中没有别名映射，需手动添加：

```python
t_aliases = {
    "end rod": [...],
    "diamond pickaxe": ["diamond pickaxe", "diamond_pickaxe", "钻石镐"],
}
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.json` | 模型配置、测试参数、提示词 |
| `run_test.py` | 测试主程序 |
| `requirements.txt` | Python 依赖 (`openai>=1.0.0`) |
