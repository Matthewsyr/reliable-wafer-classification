"""Make a compact, denominator-checked readout from four aggregate tables of the historical hosted run."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

HASHES = {
    "protocol.json": "9e8411c52e0e11357547353a21621d956c049439daaafbf4a1b6daf3896753af",
    "summary.csv": "d21bbb9e3b42bd5b258ed2cb82e0355324cb0168eb5a36c7d534388834c169ee",
    "paired_models.csv": "737f63921faf76ba902be248b8178f7cb5236ba6c43c48fd0692e4a0bfaaebad",
    "slot_swaps.csv": "86eb4604648b3a8fa0d2e9cc62a52d1839f0fa252355d65e10545667e68edaec",
}
MODELS = ("RF71", "RF77", "CNN")
SEEDS = ("42", "123", "2026")
SCENARIOS = {"equal":"各缺陷相同", "scratch_5x":"Scratch 五倍", "near_full_5x":"Near-full 五倍"}
POLICIES = ("random_fixed", "any_defect_fixed", "benefit_fixed", "benefit_positive_cap")
POOL = "simulated_flagged_candidates"


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def read_aggregates(source):
    result = {}
    for name, expected in HASHES.items():
        raw = (source/name).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==expected, f"Unexpected archive member {name}")
        result[name] = json.loads(raw) if name.endswith(".json") else list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    return result


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def triple(values):
    display = [f"{float(v):.3f}" for v in values]
    return display[0] if len(set(display)) == 1 else " / ".join(display)


def generate(source, destination):
    data = read_aggregates(source)
    info = data["protocol.json"]
    arrivals, candidates = info["gate"]["rows_total"], info["gate"]["rows_selected"]
    require((arrivals, candidates) == (11591,3642) and info["model_seeds"] == [42,123,2026], "Unexpected cohort")
    cols = ("model_seed","model","scenario","policy","rows","lots","reviewed","capacity","true_normal_reviewed",
            "true_defect_reviewed","true_defect_not_reviewed","total_cost","cost_per_1000",
            "loss_from_excluded_candidates","cost_per_1000_simulated_arrivals")
    matrix = [{k:r[k] for k in cols} for r in data["summary.csv"]
              if r["pool_kind"]==POOL and r["policy"] in POLICIES]
    ref = {(r["model"],r["scenario"],r["model_seed"],r["policy"]):r for r in matrix}
    require(len(matrix)==len(ref)==108, "Incomplete coverage")
    for r in matrix:
        cost = float(r["total_cost"])
        require(int(r["rows"])==candidates and int(r["lots"])==903 and int(r["reviewed"])==int(r["capacity"])==728,
                "Unexpected common pool or utilization")
        require(math.isclose(cost*1000/candidates,float(r["cost_per_1000"]),abs_tol=1e-8) and
                math.isclose((cost+float(r["loss_from_excluded_candidates"]))*1000/arrivals,
                             float(r["cost_per_1000_simulated_arrivals"]),abs_tol=1e-8), "Denominator mismatch")
    destination.mkdir(parents=True,exist_ok=True)
    write_rows(destination/"relative_policy_matrix.csv",matrix)
    swaps = {}
    for r in data["slot_swaps.csv"]:
        if r["pool_kind"]==POOL and r["new_model"]=="RF77" and r["baseline_model"]=="RF71" and r["policy"] in POLICIES[2:]:
            swaps.setdefault((r["model_seed"],r["scenario"],r["policy"]),[]).append(r)
    pairs = []
    for r in data["paired_models.csv"]:
        if (r["pool_kind"],r["new_model"],r["base_model"])!=(POOL,"RF77","RF71") or r["policy"] not in POLICIES[2:]:
            continue
        seed, case, pol = r["model_seed"], r["scenario"], r["policy"]
        labels = swaps[seed,case,pol]
        by_label = {x["label"]:x for x in labels}
        require(len(by_label)==len(labels)==9 and sum(int(x["entered"]) for x in labels)==int(r["entered"])
                and sum(int(x["displaced"]) for x in labels)==int(r["displaced"]), "Unreconciled membership changes")
        a,b = ref["RF71",case,seed,pol], ref["RF77",case,seed,pol]
        savings = float(r["paired_savings_per_1000"])
        arrival_savings = float(a["cost_per_1000_simulated_arrivals"])-float(b["cost_per_1000_simulated_arrivals"])
        require(abs(float(a["cost_per_1000"])-float(b["cost_per_1000"])-savings)<1e-8 and
                abs(savings*candidates/arrivals-arrival_savings)<1e-8, "Unreconciled paired cost")
        pairs.append(dict(model_seed=seed,scenario=case,policy=pol,entered=r["entered"],displaced=r["displaced"],
                          scratch_entered=by_label["Scratch"]["entered"],scratch_displaced=by_label["Scratch"]["displaced"],
                          normal_entered=by_label["none"]["entered"],normal_displaced=by_label["none"]["displaced"],
                          savings_per_1000_candidates=savings,ci_low=r["ci_low"],ci_high=r["ci_high"],
                          savings_per_1000_all_arrivals=arrival_savings))
    require(len(pairs)==18,"Missing paired panel")
    write_rows(destination/"relative_geometry_swaps.csv",pairs)
    lines = [
        "# 固定报警池中的复检排序",
        "",
        "回测使用历史开发审计集中原数据记录的缺陷或正常标签。训练和调参使用了与审计集分开的批次和同图连接组。该审计集已被查看，真实工厂复检结果和新批次独立验证尚缺。",
        "",
        "1000 张指模拟上游报警片。RF71 种子 42 在调参集校准后，在 11,591 张来片中按缺陷概率 ≥ 0.5 筛出 3,642 张，来自 903 个批次。三个完整队列各有 200 个名额，642 张的尾队有 128 个，共 728 个。三个模型在同一组晶圆、相同队列和名额上比较。",
        "",
        "普通缺陷漏损取 1、正常为 0。另分别将 Scratch 或 Near-full 漏损提高到 5。每次复检收费 0.1，假设可避免该缺陷漏损的 90%。不复检某张晶圆计其真实类别漏损 L，复检则计 0.1 + 0.1L。排序用校准概率估计 0.9 × 各类别概率加权漏损 − 0.1；最后按真实类别结算。这些数值是相对单位。",
        "",
        "## 每千张报警候选的模拟总成本",
        "",
        "每格按训练种子 42、123、2026 排列，相同的三位小数合并显示。成本越低越好。随机复用固定的同一份抽样名单。",
        "",
        "| 情景 | 模型 | 随机检满 | 按任意缺陷概率检满 | 按净收益检满 | 只检正净收益、最多 200 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case,title in SCENARIOS.items():
        for model in MODELS:
            cells=[triple(ref[model,case,s,pol]["cost_per_1000"] for s in SEEDS) for pol in POLICIES]
            lines.append("| "+" | ".join([title,model,*cells])+" |")
    lines += ["", "本次各个队列都有足够多的正净收益候选。有限额的正净收益策略实际用完 728 个名额，与净收益检满策略成本相同。108 条完整记录还保留实际复检数、正常片复检数、未获复检的真实缺陷数、候选池外漏损和每千张全来片成本。详见 `relative_policy_matrix.csv`。", "",
              "## RF77 换入和换出的晶圆", "",
              "下表固定 Scratch 五倍损失及满额净收益策略。计入未报警晶圆时，两种模型都承担相同的上游漏损，节省的分母从 3,642 扩至 11,591。区间为给定模型、队列和已选名单下的配对批次重抽样区间。",
              "", "| 种子 | 换入 / 换出 | 换入 / 换出划痕 | 换入 / 换出正常片 | 节省 / 千张候选（95% 区间） | 节省 / 千张全来片 |",
              "|---:|---:|---:|---:|---:|---:|"]
    for r in pairs:
        if r["scenario"]=="scratch_5x" and r["policy"]=="benefit_fixed":
            lines.append(f"| {r['model_seed']} | {r['entered']} / {r['displaced']} | {r['scratch_entered']} / {r['scratch_displaced']} | {r['normal_entered']} / {r['normal_displaced']} | {r['savings_per_1000_candidates']:.3f} [{float(r['ci_low']):.3f}, {float(r['ci_high']):.3f}] | {r['savings_per_1000_all_arrivals']:.3f} |")
    lines+=["", "在等损失与 Near-full 五倍损失情景下，RF77 相对 RF71 的成本差均为数值精度下的零。逐种子、逐情景的换入换出数及区间见 `relative_geometry_swaps.csv`。三个种子复用同一批晶圆，不能算作三个新的验证队列。", ""]
    (destination/"relative_readout.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    generate(args.cost,args.out)
