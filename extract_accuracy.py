"""
extract_lear_accuracy.py

Trich xuat toan bo ket qua Accuracy (Class-IL / Task-IL) tu file log training
cua model LEAR (Multi-Domain Continual Learning) ra file JSON.

Cach hoat dong:
- File log dung ky tu '\r' de cap nhat progress bar (tqdm) tren cung 1 dong,
  nen buoc dau tien la tach log thanh cac dong "logic" bang re.split theo
  ca '\r' lan '\n'.
- Voi moi scenario (hoan vi), log co dong:
      "begin task {t}, dataset:{ten_dataset}"
  cho biet dataset nao duoc train tai buoc t. Khi t quay ve 1, nghia la
  mot scenario moi bat dau.
- Sau khi train xong task t, model duoc danh gia (evaluate) tren tat ca
  cac task i <= t, va log in ra 2 dong:
      "Accuracy for {t} task(s): \t [Class-IL]: xx.xx % \t [Task-IL]: yy.yy %"
      "\tRaw accuracy values: Class-IL [a1, a2, ..., at] | Task-IL [b1, ..., bt]"
  Dong dau la accuracy trung binh (theo Class-IL / Task-IL), dong sau la
  accuracy chi tiet tren tung task i (1..t) tinh den thoi diem hien tai.

Output JSON co cau truc:
{
  "scenario_1": {
    "task_order": ["seq-cifar10", "seq-cropdisease", ...],   # dataset theo thu tu train
    "results": [
      {
        "after_task": 1,                  # vua train xong task 1
        "dataset_trained": "seq-cifar10",
        "class_il_avg": 97.35,
        "task_il_avg": 0.0,
        "class_il_per_task": [97.3457532051282],
        "task_il_per_task": [0]
      },
      {
        "after_task": 2,
        "dataset_trained": "seq-cropdisease",
        "class_il_avg": 97.66,
        "task_il_avg": 0.0,
        "class_il_per_task": [96.62459935897436, 98.69011976047905],
        "task_il_per_task": [0, 0]
      },
      ...
    ]
  },
  "scenario_2": {...},
  "scenario_3": {...}
}
"""

import re
import json
import argparse


BEGIN_TASK_RE = re.compile(r"^begin task (\d+), dataset:(\S+)")
ACC_SUMMARY_RE = re.compile(
    r"^Accuracy for (\d+) task\(s\):\s*"
    r"\[Class-IL\]:\s*([\-\d.]+)\s*%\s*"
    r"\[Task-IL\]:\s*([\-\d.]+)\s*%"
)
RAW_ACC_RE = re.compile(
    r"^Raw accuracy values:\s*Class-IL\s*\[([^\]]*)\]\s*\|\s*Task-IL\s*\[([^\]]*)\]"
)


def parse_number_list(s: str):
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",")]


def extract_accuracy(log_path: str):
    with open(log_path, "r", errors="ignore") as f:
        content = f.read()

    # Progress bars overwrite the line using '\r'; split on both '\r' and '\n'
    # to get one "logical" printed line per element, then strip tab/space.
    raw_lines = re.split(r"[\r\n]+", content)
    lines = [l.strip() for l in raw_lines]

    scenarios = []
    current_scenario = None
    current_dataset_for_task = {}  # task_index -> dataset name, within current scenario

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        m_begin = BEGIN_TASK_RE.match(line)
        if m_begin:
            task_idx = int(m_begin.group(1))
            dataset_name = m_begin.group(2)
            if task_idx == 1:
                # A new scenario (permutation) starts
                current_scenario = {
                    "task_order": [],
                    "results": [],
                }
                scenarios.append(current_scenario)
                current_dataset_for_task = {}
            current_dataset_for_task[task_idx] = dataset_name
            # keep task_order in sequence
            if current_scenario is not None:
                current_scenario["task_order"].append(dataset_name)
            i += 1
            continue

        m_acc = ACC_SUMMARY_RE.match(line)
        if m_acc:
            n_tasks = int(m_acc.group(1))
            class_il_avg = float(m_acc.group(2))
            task_il_avg = float(m_acc.group(3))

            # The "Raw accuracy values" line should be the next logical line
            class_il_per_task = []
            task_il_per_task = []
            if i + 1 < n:
                m_raw = RAW_ACC_RE.match(lines[i + 1])
                if m_raw:
                    class_il_per_task = parse_number_list(m_raw.group(1))
                    task_il_per_task = parse_number_list(m_raw.group(2))

            if current_scenario is not None:
                current_scenario["results"].append({
                    "after_task": n_tasks,
                    "dataset_trained": current_dataset_for_task.get(n_tasks),
                    "class_il_avg": class_il_avg,
                    "task_il_avg": task_il_avg,
                    "class_il_per_task": class_il_per_task,
                    "task_il_per_task": task_il_per_task,
                })
            i += 2  # skip the raw-values line we just consumed
            continue

        i += 1

    result = {f"scenario_{idx + 1}": sc for idx, sc in enumerate(scenarios)}
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract LEAR accuracy log to JSON")
    parser.add_argument("log_path", help="Path to the training log .txt file")
    parser.add_argument("-o", "--output", default="lear_accuracy_results.json",
                         help="Output JSON file path")
    args = parser.parse_args()

    data = extract_accuracy(args.log_path)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    n_scenarios = len(data)
    total_points = sum(len(v["results"]) for v in data.values())
    print(f"Da trich xuat {n_scenarios} scenario(s), tong cong {total_points} diem accuracy.")
    print(f"Ket qua da luu vao: {args.output}")


if __name__ == "__main__":
    main()