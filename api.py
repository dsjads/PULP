import requests
import time
import json

API_BASE = "http://10.246.3.17:8000"

# 1. 生成测试
response = requests.post(
    f"{API_BASE}/generate",
    json={
        "project_path": "./Time-workspace/Time1-buggy",
        "class_fqn": "org.joda.time.IllegalInstantException",
        "mode": "api",
    }
)

data = response.json()
run_id = data["data"]["run_id"]
print(f"Run ID: {run_id}")

# 2. 轮询进度
while True:
    progress = requests.get(f"{API_BASE}/progress/{run_id}").json()
    status = progress.get("status")
    percent = progress.get("progress_percent", 0)

    print(f"状态: {status}, 进度: {percent}%")

    if status == "completed":
        print("生成完成！")
        break

    time.sleep(5)

# 3. 获取测试文件
# result = requests.get(f"{API_BASE}/results/{run_id}/files/sorted").json()
# test_content = result["content"]
#
# with open("test.java", "w") as f:
#     f.write(test_content)
#
# print("测试文件已保存到 test.java")