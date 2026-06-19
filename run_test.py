import json
import csv
import time
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')


def resolve_api_key(raw_key):
    if raw_key.startswith("env:"):
        var_name = raw_key[4:]
        val = os.environ.get(var_name, "")
        if not val:
            print(f"  警告: 环境变量 {var_name} 未设置")
        return val
    return raw_key


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_list(config):
    if "models" in config:
        return config["models"]
    api = config.get("api", {})
    if api:
        return [{
            "name": api.get("model", "unknown"),
            "base_url": api.get("base_url", ""),
            "api_key": api.get("api_key", ""),
            "model": api.get("model", ""),
            "temperature_guesser": api.get("temperature_guesser", 0.7),
            "temperature_answerer": api.get("temperature_answerer", 0.0),
            "max_tokens": api.get("max_tokens", 512),
            "timeout_seconds": api.get("timeout_seconds", 60),
        }]
    return []


def create_client(mcfg):
    return OpenAI(
        base_url=mcfg["base_url"],
        api_key=resolve_api_key(mcfg["api_key"]),
        timeout=mcfg.get("timeout_seconds", 60),
    )


def ask_model(client, mcfg, messages, role_temp_key):
    ts_start = datetime.now(timezone.utc).isoformat()
    response = client.chat.completions.create(
        model=mcfg["model"],
        messages=messages,
        temperature=mcfg.get(role_temp_key, 0.7),
    )
    ts_end = datetime.now(timezone.utc).isoformat()

    choice = response.choices[0]
    content = choice.message.content
    if content:
        content = content.strip()

    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    finish_reason = choice.finish_reason if choice.finish_reason else "unknown"

    return {
        "content": content if content else "",
        "finish_reason": finish_reason,
        "usage": usage,
        "timestamp_start": ts_start,
        "timestamp_end": ts_end,
    }


def get_answer(client, mcfg, config, question):
    target = config["test"]["target_item"]
    target_cn = config["test"].get("target_item_cn", target)
    prompts = config["prompts"]

    system_content = prompts["answerer_system"].replace("{target}", target).replace("{target_cn}", target_cn)
    user_content = prompts["answerer_user_template"].format(
        target=target, target_cn=target_cn, question=question
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    result = ask_model(client, mcfg, messages, "temperature_answerer")
    result["_prompt_system"] = system_content
    result["_prompt_user"] = user_content
    return result


def parse_response(content):
    if not content:
        return "error", "(空响应)"
    upper = content.upper().strip()
    guess_markers = [
        "I GUESS:", "MY GUESS IS:", "THE ANSWER IS:", "IT IS THE",
        "IT'S THE", "I THINK IT IS", "I THINK IT'S", "IS IT THE",
        "FINAL ANSWER:", "我的猜测是:", "我猜是:", "答案是:",
    ]
    for marker in guess_markers:
        if marker in upper:
            idx = upper.index(marker) + len(marker)
            guess = content[idx:].strip().lstrip(":").strip()
            guess = guess.rstrip(".!。！?？\"'")
            if guess:
                return "guess", guess
    give_up_markers = ["I GIVE UP", "I SURRENDER", "I DON'T KNOW", "我放弃"]
    for m in give_up_markers:
        if m in upper:
            return "give_up", None
    return "question", content


def is_guess_correct(guess, target):
    g = guess.strip().lower().rstrip("s")
    t = target.strip().lower()
    t_aliases = {
        "end rod": ["end rod", "endrod", "end rods", "end_rod", "末地烛", "end rod (末地烛)"],
    }
    aliases = t_aliases.get(t, [t])
    return g in aliases


def run_single_round(client, mcfg, config, round_num):
    test = config["test"]
    prompts = config["prompts"]
    target = test["target_item"]
    target_cn = test.get("target_item_cn", target)
    max_q = test["max_questions"]

    system_prompt = prompts["guesser_system"].replace("{max_questions}", str(max_q))

    messages = [{"role": "system", "content": system_prompt}]
    qa_pairs = []
    final_guess = None
    success = False
    ended_early = False
    error_info = None

    round_start = time.time()
    round_ts = datetime.now(timezone.utc).isoformat()

    for q_num in range(1, max_q + 1):
        try:
            guesser_result = ask_model(client, mcfg, messages, "temperature_guesser")
        except Exception as e:
            error_info = {"question_number": q_num, "error_type": type(e).__name__, "error_message": str(e)}
            qa_pairs.append({
                "q_num": q_num, "role": "guesser_error",
                "raw_response": None, "parsed_action": "error",
                "content": "", "answer_raw": "", "answer_clean": "",
                "guesser_usage": {}, "answerer_usage": {},
                "guesser_finish": "error", "answerer_finish": "",
                "ts_start": datetime.now(timezone.utc).isoformat(),
                "ts_end": datetime.now(timezone.utc).isoformat(),
                "error": error_info["error_message"],
            })
            break

        content = guesser_result["content"]
        action, parsed_content = parse_response(content)

        if action == "guess":
            final_guess = parsed_content
            success = is_guess_correct(parsed_content, target)
            qa_pairs.append({
                "q_num": q_num, "role": "guesser_guess",
                "raw_response": content,
                "parsed_action": "guess", "content": content,
                "guess_value": parsed_content, "is_correct": success,
                "answer_raw": "(猜测)", "answer_clean": "(猜测)",
                "guesser_usage": guesser_result["usage"],
                "answerer_usage": {},
                "guesser_finish": guesser_result["finish_reason"],
                "answerer_finish": "",
                "ts_start": guesser_result["timestamp_start"],
                "ts_end": guesser_result["timestamp_end"],
                "error": None,
            })
            ended_early = True
            break
        elif action == "give_up":
            final_guess = "(放弃)"
            success = False
            qa_pairs.append({
                "q_num": q_num, "role": "guesser_giveup",
                "raw_response": content,
                "parsed_action": "give_up", "content": content,
                "guess_value": None, "is_correct": False,
                "answer_raw": "(放弃)", "answer_clean": "(放弃)",
                "guesser_usage": guesser_result["usage"],
                "answerer_usage": {},
                "guesser_finish": guesser_result["finish_reason"],
                "answerer_finish": "",
                "ts_start": guesser_result["timestamp_start"],
                "ts_end": guesser_result["timestamp_end"],
                "error": None,
            })
            ended_early = True
            break
        else:
            answer_result = get_answer(client, mcfg, config, content)
            answer_raw = answer_result["content"]
            answer_clean = answer_raw.strip().rstrip(".")
            if answer_clean.upper().startswith("YES"):
                answer_clean = "Yes"
            elif answer_clean.upper().startswith("NO"):
                answer_clean = "No"

            qa_pairs.append({
                "q_num": q_num, "role": "qa",
                "raw_response": content,
                "parsed_action": "question", "content": content,
                "guess_value": None, "is_correct": None,
                "answer_raw": answer_raw, "answer_clean": answer_clean,
                "guesser_usage": guesser_result["usage"],
                "answerer_usage": answer_result["usage"],
                "guesser_finish": guesser_result["finish_reason"],
                "answerer_finish": answer_result["finish_reason"],
                "ts_start": guesser_result["timestamp_start"],
                "ts_end": answer_result["timestamp_end"],
                "error": None,
            })
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": answer_clean})

    elapsed = round(time.time() - round_start, 2)

    if not ended_early and not error_info:
        final_guess = "(超出问题上限)"
        success = False

    total_guesser_tokens = sum(
        (p["guesser_usage"].get("total_tokens", 0) or 0) for p in qa_pairs
    )
    total_answerer_tokens = sum(
        (p["answerer_usage"].get("total_tokens", 0) or 0) for p in qa_pairs
    )

    return {
        "round": round_num,
        "success": success,
        "questions_asked": len(qa_pairs),
        "final_guess": final_guess,
        "target": target,
        "target_cn": target_cn,
        "elapsed_seconds": elapsed,
        "round_timestamp": round_ts,
        "error": error_info,
        "system_prompt_injected": system_prompt,
        "total_guesser_tokens": total_guesser_tokens,
        "total_answerer_tokens": total_answerer_tokens,
        "qa_pairs": qa_pairs,
    }


def run_model_tests(client, mcfg, config, print_lock=None):
    test = config["test"]
    target = test["target_item"]
    target_cn = test.get("target_item_cn", target)
    rounds = test["rounds"]
    max_q = test["max_questions"]
    model_name = mcfg["name"]

    def pp(*args, **kwargs):
        if print_lock:
            with print_lock:
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    pp("-" * 60)
    pp(f"  模型: {model_name}  ({mcfg['model']})")
    pp(f"  目标: {target} ({target_cn})  |  {rounds} 轮, 每轮最多 {max_q} 问")
    pp("-" * 60)

    round_workers = mcfg.get("round_workers", rounds)

    def run_round(r):
        pp(f"  [{model_name}] [Round {r}/{rounds}]", end=" ", flush=True)
        result = run_single_round(client, mcfg, config, r)
        status = "✓" if result["success"] else "✗"
        tokens = f"猜{result['total_guesser_tokens']}t+答{result['total_answerer_tokens']}t"
        final = result['final_guess'] or "(无)"
        pp(f"{status}  ({result['questions_asked']}问, {result['elapsed_seconds']}s, {tokens})  → {final[:40]}")
        return result

    all_results = []
    with ThreadPoolExecutor(max_workers=round_workers) as worker:
        futures = {worker.submit(run_round, r): r for r in range(1, rounds + 1)}
        for future in as_completed(futures):
            all_results.append(future.result())

    all_results.sort(key=lambda x: x["round"])

    successes = sum(1 for r in all_results if r["success"])
    avg_q = sum(r["questions_asked"] for r in all_results) / rounds
    avg_t = sum(r["elapsed_seconds"] for r in all_results) / rounds
    avg_gt = sum(r["total_guesser_tokens"] for r in all_results) / rounds
    avg_at = sum(r["total_answerer_tokens"] for r in all_results) / rounds
    pp(f"  [{model_name}] 结果: {successes}/{rounds} ({successes/rounds*100:.1f}%)  |  平均 {avg_q:.1f}问  |  平均 {avg_t:.1f}s  |  平均猜{avg_gt:.0f}t+答{avg_at:.0f}t")
    pp()

    return {
        "model_name": model_name,
        "model_id": mcfg["model"],
        "base_url": mcfg["base_url"],
        "model_config": {
            "temperature_guesser": mcfg.get("temperature_guesser"),
            "temperature_answerer": mcfg.get("temperature_answerer"),
            "max_tokens": mcfg.get("max_tokens"),
            "timeout_seconds": mcfg.get("timeout_seconds"),
        },
        "results": all_results,
        "successes": successes,
        "total": rounds,
        "success_rate": successes / rounds,
        "avg_questions": avg_q,
        "avg_time": avg_t,
        "avg_guesser_tokens": avg_gt,
        "avg_answerer_tokens": avg_at,
    }


def generate_output(all_model_results, config, model_configs, output_path):
    output = {
        "meta": {
            "test_time_utc": datetime.now(timezone.utc).isoformat(),
            "test_time_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target_item": config["test"]["target_item"],
            "target_item_cn": config["test"].get("target_item_cn", ""),
            "rounds_per_model": config["test"]["rounds"],
            "max_questions": config["test"]["max_questions"],
            "total_models": len(all_model_results),
            "prompts": {
                "guesser_system": config["prompts"]["guesser_system"],
                "answerer_system": config["prompts"]["answerer_system"],
                "answerer_user_template": config["prompts"]["answerer_user_template"],
            },
        },
        "comparison": [],
        "results": [],
    }

    for mr, mcfg in zip(all_model_results, model_configs):
        output["comparison"].append({
            "model_name": mr["model_name"],
            "model_id": mr["model_id"],
            "base_url": mr["base_url"],
            "model_config": mr["model_config"],
            "successes": mr["successes"],
            "total_rounds": mr["total"],
            "success_rate": mr["success_rate"],
            "success_rate_pct": f"{mr['success_rate']*100:.1f}%",
            "avg_questions": mr["avg_questions"],
            "avg_time_seconds": mr["avg_time"],
            "avg_guesser_tokens": mr["avg_guesser_tokens"],
            "avg_answerer_tokens": mr["avg_answerer_tokens"],
            "round_summaries": [
                f"R{r['round']}: {'✓' if r['success'] else '✗'} {r['questions_asked']}问 → {r['final_guess']}"
                for r in mr["results"]
            ],
        })

        model_data = {
            "model_name": mr["model_name"],
            "model_id": mr["model_id"],
            "base_url": mr["base_url"],
            "model_config": mr["model_config"],
            "summary": {
                "successes": mr["successes"],
                "total_rounds": mr["total"],
                "success_rate": mr["success_rate"],
                "avg_questions": mr["avg_questions"],
                "avg_time_seconds": mr["avg_time"],
                "avg_guesser_tokens": mr["avg_guesser_tokens"],
                "avg_answerer_tokens": mr["avg_answerer_tokens"],
            },
            "rounds": [],
        }

        for r in mr["results"]:
            round_data = {
                "round": r["round"],
                "success": r["success"],
                "questions_asked": r["questions_asked"],
                "final_guess": r["final_guess"],
                "target": r["target"],
                "elapsed_seconds": r["elapsed_seconds"],
                "round_timestamp": r["round_timestamp"],
                "error": r["error"],
                "system_prompt_injected": r["system_prompt_injected"],
                "total_guesser_tokens": r["total_guesser_tokens"],
                "total_answerer_tokens": r["total_answerer_tokens"],
                "qa_sequence": [],
            }
            for qa in r["qa_pairs"]:
                round_data["qa_sequence"].append({
                    "question_number": qa["q_num"],
                    "role": qa["role"],
                    "parsed_action": qa["parsed_action"],
                    "guesser_raw_response": qa["raw_response"],
                    "guesser_parsed_content": qa["content"],
                    "guess_value": qa["guess_value"],
                    "is_correct": qa["is_correct"],
                    "answerer_raw_response": qa["answer_raw"],
                    "answerer_clean": qa["answer_clean"],
                    "guesser_finish_reason": qa["guesser_finish"],
                    "answerer_finish_reason": qa["answerer_finish"],
                    "guesser_token_usage": qa["guesser_usage"],
                    "answerer_token_usage": qa["answerer_usage"],
                    "timestamp_start": qa["ts_start"],
                    "timestamp_end": qa["ts_end"],
                    "error": qa.get("error"),
                })
            model_data["rounds"].append(round_data)

        output["results"].append(model_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output_path


def main():
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    models = get_model_list(config)

    if not models:
        print("错误: 配置文件中没有模型配置，请在 config.json 中添加 models 数组或 api 配置")
        sys.exit(1)

    test = config["test"]
    target = test["target_item"]
    target_cn = test.get("target_item_cn", target)

    print("=" * 60)
    print(f"  AI 猜物品智力测试 - 多模型对比")
    print(f"  目标物品: {target} ({target_cn})")
    print(f"  测试模型数: {len(models)}  |  每模型 {test['rounds']} 轮  |  最多 {test['max_questions']} 问/轮")
    print("=" * 60)
    print()

    all_model_results = []
    model_configs = []
    print_lock = threading.Lock()

    def run_one_model(mcfg):
        api_key = resolve_api_key(mcfg["api_key"])
        if not api_key:
            with print_lock:
                print(f"[跳过] {mcfg['name']}: API Key 未配置")
                print()
            return None, None
        try:
            client = create_client(mcfg)
        except Exception as e:
            with print_lock:
                print(f"[跳过] {mcfg['name']}: 创建客户端失败 - {e}")
                print()
            return None, None
        mr = run_model_tests(client, mcfg, config, print_lock)
        return mr, mcfg

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {executor.submit(run_one_model, mcfg): mcfg for mcfg in models}
        for future in as_completed(futures):
            mr, mcfg = future.result()
            if mr is not None:
                all_model_results.append(mr)
                model_configs.append(mcfg)

    model_order = {m["name"]: i for i, m in enumerate(models)}
    paired = sorted(
        zip(all_model_results, model_configs),
        key=lambda x: model_order.get(x[0]["model_name"], 999),
    )
    all_model_results = [p[0] for p in paired]
    model_configs = [p[1] for p in paired]

    if not all_model_results:
        print("错误: 没有成功运行的模型测试")
        sys.exit(1)

    print("=" * 60)
    print(f"  最终对比结果")
    print("=" * 60)
    print(f"  {'模型':<18} {'成功':>6} {'成功率':>8} {'平均问数':>8} {'平均耗时':>8} {'猜Token':>8} {'答Token':>8}")
    print(f"  {'-'*18} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for mr in all_model_results:
        gt = f"{mr['avg_guesser_tokens']:.0f}" if mr.get("avg_guesser_tokens") else "-"
        at = f"{mr['avg_answerer_tokens']:.0f}" if mr.get("avg_answerer_tokens") else "-"
        print(f"  {mr['model_name']:<18} {mr['successes']}/{mr['total']:>4} {mr['success_rate']*100:>7.1f}% {mr['avg_questions']:>7.1f} {mr['avg_time']:>7.1f}s {gt:>7} {at:>7}")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results_{timestamp}.json"
    generate_output(all_model_results, config, model_configs, output_path)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
