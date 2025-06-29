import requests
import numpy as np
import pandas as pd
import time
import os

#使用本地已经下载好的缓存在C:\Users\Charles\.cache\huggingface\datasets的数据集
os.environ["HF_DATASETS_OFFLINE"] = "1"


class OllamaEmbeddingModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/v1/embeddings"
        self.revision = "main"
        self.model_card_data = {}  # 兼容MTEB

    def encode(self, sentences, batch_size=8, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            print(f"正在处理第 {i} ~ {i+len(batch)-1} 条文本")
            response = requests.post(
                self.api_url,
                json={"model": self.model_name, "input": batch}
            )
            if response.status_code != 200:
                print(f"Error: {response.text}")
                raise Exception(f"Ollama API error: {response.text}")
            data = response.json()
            batch_embeds = [item["embedding"] for item in data["data"]]
            embeddings.extend(batch_embeds)
            time.sleep(0.05)
        return np.array(embeddings)

model_names = [

    "dengcao/Qwen3-Embedding-4B:Q8_0",
    "dengcao/Qwen3-Embedding-0.6B:Q8_0",
    "quentinz/bge-large-zh-v1.5",
    "bge-m3",

]

# 推荐的 MTEB 中文/英文任务（可根据 available_tasks() 调整）
mteb_tasks = ["AFQMC", "BQ", "LCQMC", "PAWSX", "TNews", "STSBenchmark"]

def run_mteb_eval():
    from mteb import MTEB
    results = []
    for model_name in model_names:
        print(f"评测 MTEB: {model_name}")
        model = OllamaEmbeddingModel(model_name)
        evaluation = MTEB(tasks=mteb_tasks)
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        result = evaluation.run(model,
                                output_folder=f"results/mteb/{safe_model_name}",
                                limit=20, # 只评测每个任务前20条
                                verbosity=2  # 让MTEB自动打印详细进度

                                )
        # 兼容 list/dict
        if isinstance(result, dict):
            for task, metrics in result.items():
                for metric, score in metrics.items():
                    results.append({
                        "model": model_name,
                        "task": task,
                        "metric": metric,
                        "score": score
                    })
        elif isinstance(result, list):
            for item in result:
                # 兼容 pydantic 对象和 dict
                if hasattr(item, "model_dump"):  # Pydantic v2
                    item_dict = item.model_dump()
                elif hasattr(item, "dict"):      # Pydantic v1
                    item_dict = item.dict()
                elif hasattr(item, "__dict__"):
                    item_dict = vars(item)
                elif isinstance(item, dict):
                    item_dict = item
                else:
                    print("未知的result item类型，请手动检查！")
                    continue

                # 处理 scores（新版MTEB结构）
                if "scores" in item_dict and item_dict["scores"]:
                    for split, split_results in item_dict["scores"].items():
                        for res in split_results:
                            for metric_name, metric_score in res.items():
                                # 只保留数值型指标
                                if isinstance(metric_score, (int, float)):
                                    results.append({
                                        "model": model_name,
                                        "task": item_dict.get("task_name", ""),
                                        "split": split,
                                        "metric": metric_name,
                                        "score": metric_score
                                    })
                else:
                    print("未发现 scores 字段，item_dict内容如下：")
                    print(item_dict)
                    results.append({
                        "model": model_name,
                        "task": item_dict.get("task_name", ""),
                        "split": "",
                        "metric": "",
                        "score": ""
                    })
        else:
            print("未知的result类型，请手动检查！")
    df = pd.DataFrame(results)
    df.to_csv("mteb-eval-charles.csv", index=False)
    print("MTEB 评测完成，结果已保存到 mteb-eval-charles.csv")

    # 读取评测结果
    df = pd.read_csv("mteb-eval-charles.csv")

    # 只保留有分数的行
    df = df[df["score"].notnull() & (df["score"] != "")]

    # 分组排序并写入txt
    with open("mteb-dimensions-sorted.txt", "w", encoding="utf-8") as f:
        for (task, metric), group in df.groupby(["task", "metric"]):
            f.write(f"【Task: {task} | Metric: {metric}】\n")
            # 按分数降序排列
            group_sorted = group.sort_values("score", ascending=False)
            for idx, row in group_sorted.iterrows():
                f.write(f"  {row['model']} : {row['score']}\n")
            f.write("\n")
    print("按照各项Task，倒序排列各种embedding的对应能力已完成，内容见mteb-dimensions-sorted.txt")




if __name__ == "__main__":
    run_mteb_eval()