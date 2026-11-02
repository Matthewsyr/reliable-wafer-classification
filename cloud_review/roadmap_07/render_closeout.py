"""Render the Chinese closeout from checksum-verified, already aggregated cloud outputs."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path


POPS = {"original_full":"原测试全集", "no_direct_dev_duplicate":"剔除开发重复图",
        "no_transitive_dev_component":"剔除全部开发相连组件"}
MAIN = ("original_full", "no_transitive_dev_component")
MODELS = ("RF71", "RF77", "CNN")
SEEDS = ("42", "123", "2026")
COST = "scratch_5x"
RISK = "benefit_positive_cap"
POLICIES = {"random_fixed":"固定随机", "any_defect_fixed":"任意缺陷概率", RISK:"正净收益"}


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def text_table(payload):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))


def index(rows, *fields):
    lookup = {tuple(r[k] for k in fields):r for r in rows}
    require(len(lookup)==len(rows), "Duplicate aggregate keys")
    return lookup


def integer(value):
    return f"{int(value):,}"


def decimal(value):
    x=float(value)
    return f"{0 if abs(x)<.0005 else x:,.3f}"


def ordered(values, fmt=integer):
    return " / ".join(fmt(v) for v in values)


def percent(n, d):
    return f"{100*n/d:.1f}%" if d else "无名额"


def read_run(root):
    receipt=json.loads((root/"run_receipt.json").read_bytes())
    require(receipt.get("status")=="complete", "Replay has not completed")
    result={}
    for name in ("populations","queues","policy_costs","target_occupancy","swaps_by_class","model_pairs"):
        member=name+".csv"
        path=root/member
        data=path.read_bytes() if path.exists() else gzip.decompress((root/(member+".gz")).read_bytes())
        require(hashlib.sha256(data).hexdigest()==receipt["sha256"][member], f"Changed aggregate: {member}")
        result[name]=text_table(data)
    return result,receipt


def report(run, support, repo):
    tables, receipt=read_run(run)
    pop={r["population"]:r for r in tables["populations"]}
    require(set(pop)==set(POPS) and len(pop)==len(tables["populations"]), "Unexpected populations")
    active=[p for p in MAIN if int(pop[p]["candidates"])>0]
    k=("population","seed","model","scenario","policy")
    metrics=index(tables["policy_costs"],*k)
    target=index(tables["target_occupancy"],*k,"subgroup")
    paired=index(tables["model_pairs"],"population","seed","scenario","policy")
    swaps=index(tables["swaps_by_class"],"population","seed","scenario","policy","label")
    require(len(metrics)==receipt["panels"]==45*3*sum(int(r["candidates"])>0 for r in pop.values()),
            "Incomplete policy table")
    require(int(pop[MAIN[0]]["target_normals"])==78158 and int(pop[MAIN[0]]["arrivals"])==118595,
            "Original exposure differs from historical report")
    for policy_key,r in metrics.items():
        p=pop[r["population"]]
        cap,chosen=int(p["capacity"]),int(r["reviewed"])
        n,m=int(p["arrivals"]),int(p["candidates"])
        require(0<=chosen<=cap and m<=n, "Capacity or denominator invalid")
        require(math.isclose(float(r["candidate_cost"])+float(r["upstream_missed_loss"]),float(r["total_cost"]),
                             rel_tol=0,abs_tol=1e-5), "Source settlement incomplete")
        require(math.isclose(float(r["total_cost"])*1000/n,float(r["cost_per_1000_arrivals"]),rel_tol=0,abs_tol=1e-5)
                and math.isclose(float(r["candidate_cost"])*1000/m,float(r["cost_per_1000_candidates"]),rel_tol=0,abs_tol=1e-5),
                "Cost normalization mismatch")
        g=target[policy_key+("target_normal",)]
        require(int(g["reviewed"])<=int(r["normal_reviewed"]) and int(g["all_capacity"])==cap
                and int(g["all_used_slots"])==chosen and int(g["arrivals"])==int(p["target_normals"]),
                "Target dimensions/denominators mismatch")
    def rows(p,model,table,policy=RISK,subgroup=None):
        keys=[(p,s,model,COST,policy) for s in SEEDS]
        return [table[key+((subgroup,) if subgroup else ())] for key in keys]
    lines=["# 复检名额用在了哪些晶圆上", ""]
    if MAIN[0] in active:
        pop0=pop[MAIN[0]]
        t=rows(MAIN[0],"RF77",target,subgroup="target_normal")[0]
        c=rows(MAIN[0],"RF77",metrics)[0]
        share=percent(int(t["reviewed"]),int(t["all_capacity"]))
        lines += [f"原测试中，110,701 张正常片有 78,158 张（70.60%）的原始高宽未见于开发正常片。将这些精确尺寸记为 **T**。在划痕漏损五倍、RF77 种子 42 的正净收益策略下，**T 内正常片占用 {integer(t['reviewed'])} / {integer(t['all_capacity'])} 个可用复检名额（{share}）**。该策略实际使用 {integer(c['reviewed'])} 个名额。70.60% 描述正常来片的尺寸分布，名额占比的分母是本次队列的全部可用名额。", ""]
    lines += ["### 目标尺寸、划痕命中与实际标签结算", "",
              "每格按种子 42 / 123 / 2026 顺序列值。策略在每个报警候选队列内按预计可避免的损失排序，只检查正净收益晶圆。真实划痕列统计所有尺寸中获检的真实划痕。两个成本列分别使用全部来片和全部报警候选作分母，单位均为相对损失，越低越好。", "",
              "| 人群 | 排序模型 | T 内正常片获检数 / 全部可用名额 | 获检真实划痕数 | 成本／千张全来片 | 成本／千张报警候选 |",
              "|---|---|---|---|---|---|"]
    for p in active:
        for model in MODELS:
            m=rows(p,model,metrics);t=rows(p,model,target,subgroup="target_normal")
            capacity=int(pop[p]["capacity"])
            normal="、".join(integer(r["reviewed"]) for r in t)+f"（各 / {integer(capacity)}）"
            lines.append(f"| {POPS[p]} | {model} | {normal} | {ordered(r['scratch_reviewed'] for r in m)} | {ordered((r['cost_per_1000_arrivals'] for r in m),decimal)} | {ordered((r['cost_per_1000_candidates'] for r in m),decimal)} |")
    for p in MAIN:
        if p not in active:
            lines.append(f"\n{POPS[p]}没有报警候选，无法估计候选内的排名和名额占比。")
    lines+=["", "## 固定池与容量", "",
            "1000 张指上游已报警的候选。三个排序模型共用种子 42 的 RF71 筛查结果，使用只在开发调参集上冻结的温度与缺陷概率 ≥ 0.5 的门槛。按固定批次顺序构造虚拟 1000 片队列，每个队列最多复检 200 张，尾队按 20% 向下取整。每个敏感性人群重新组队。原测试的标签已在过去的诊断中查看，本轮固定九份权重与校准参数。", "",
            "| 人群 | 有标签来片 | lot | 报警候选 | 可用名额 | 真实缺陷在上游漏掉 | T 内正常来片 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for p,title in POPS.items():
        r=pop[p]
        lines.append("| "+" | ".join([title]+[integer(r[k]) for k in ("arrivals","lots","candidates","capacity","upstream_fn","target_normals")])+" |")
    lines += ["", "对未复检的晶圆按其真实标签计漏损 L，复检则计 0.1 + 0.1L。普通缺陷漏损为 1，正常为 0。分别回放所有缺陷等损失、Scratch 五倍、Near-full 五倍三种情景，均假定检查收费 0.1 且处置避免 90% 的缺陷漏损。预测概率只用于决策。全来片成本包括未能入池的缺陷。", "",
              "### 相同候选和名额下的三种策略", "",
              "下表固定 RF77 和划痕五倍损失。数值仍按三个种子排列。随机为预先固定的散列抽样，按缺陷概率排序取 1 − P(正常)。净收益排序按各类别漏损加权并计入检查费用。", "",
              "| 人群 | 排序 | 实际使用名额 | 获检真实划痕数 | 成本／千张全来片 |",
              "|---|---|---|---|---|"]
    for p in active:
        for pol,title in POLICIES.items():
            rr=rows(p,"RF77",metrics,policy=pol)
            lines.append(f"| {POPS[p]} | {title} | {ordered(r['reviewed'] for r in rr)} | {ordered(r['scratch_reviewed'] for r in rr)} | {ordered((r['cost_per_1000_arrivals'] for r in rr),decimal)} |")
    if len(active)==2:
        # Hold RF77 and the loss schedule fixed. A paired model comparison answers a different question.
        gaps = []
        for p in MAIN:
            prob, risk = rows(p,"RF77",metrics,policy="any_defect_fixed"), rows(p,"RF77",metrics)
            gaps.append([float(b["cost_per_1000_arrivals"])-float(a["cost_per_1000_arrivals"]) for a,b in zip(prob,risk)])
        if all(d>1e-8 for panel in gaps for d in panel):
            lines += ["",f"**固定 RF77，改换排序策略。** 即使漏掉划痕的损失为普通缺陷的五倍，按任意缺陷概率排序的实际标签结算成本仍在两个人群的三个种子上全部较低。改用预期净收益排序，每千张全来片的成本在原测试全集分别增加 {ordered(gaps[0],decimal)}，在严格隔离人群分别增加 {ordered(gaps[1],decimal)} 相对单位。"]
        a=metrics[MAIN[0],"42","RF77",COST,"any_defect_fixed"]
        b=metrics[MAIN[0],"42","RF77",COST,RISK]
        if a["reviewed"]==b["reviewed"]:
            sa,sb=int(a["scratch_reviewed"]),int(b["scratch_reviewed"])
            na,nb=int(a["normal_reviewed"]),int(b["normal_reviewed"])
            oa=int(a["reviewed"])-sa-na
            ob=int(b["reviewed"])-sb-nb
            # The common gate cancels and equal numbers of reviews imply equal fees.
            extra_cost=0.9*((oa-ob)-5*(sb-sa))
            require(math.isclose(extra_cost,float(b["total_cost"])-float(a["total_cost"]),rel_tol=0,abs_tol=1e-6),
                    "RF77 policy difference does not reconcile with observed class substitutions")
            if sb>sa and nb>na and oa>ob and extra_cost>0:
                lines += ["",f"以全集种子 42 为例，两份名单都复检 {integer(a['reviewed'])} 张。按预期净收益排序，真实划痕从 {integer(sa)} 增至 {integer(sb)}（多 {integer(sb-sa)} 张），正常片从 {integer(na)} 增至 {integer(nb)}（多 {integer(nb-na)} 张），其他真实缺陷从 {integer(oa)} 降至 {integer(ob)}（净少 {integer(oa-ob)} 张）。两份名单的复检费相同。多查到划痕的收益抵不过少查其他真实缺陷的损失，模拟总成本上升 {extra_cost:,.1f} 单位。开发集校准所得分数在这批来片上的排序存在这种代价，本次仍沿用冻结门槛，未借已见测试标签另选阈值。"]
    lines+=["","## RF77 的几何特征是否值得保留", "",
            "固定满额净收益策略，唯一变化是把 RF71 换成 RF77。下表给出每千张全来片节省的相对单位，正值表示 RF77 成本较低。结论逐情景比较；只有全集和严格组件隔离人群均在三个种子上显示正节省，才称该情景的点估计方向稳定。", "",
            "| 情景 | 全集节省（42 / 123 / 2026） | 严格隔离节省（42 / 123 / 2026） | 两人群、三个种子均降低成本 |",
            "|---|---|---|---|"]
    for case,title in (("equal","等损失"),(COST,"Scratch 五倍"),("near_full_5x","Near-full 五倍")):
        vv=[[float(paired[p,s,case,"benefit_fixed"]["per_1000_all_arrivals"]) for s in SEEDS] if p in active else [] for p in MAIN]
        consistent=all(len(v)==3 and all(x>1e-8 for x in v) for v in vv)
        row=[title,*[ordered(v,decimal) if v else "无候选" for v in vv],"是" if consistent else "否"]
        lines.append("| "+" | ".join(row)+" |")
    lines+=["", "Scratch 五倍情景的名单交换如下。换入和换出分别计数，正的划痕净数代表额外获检的真实划痕。区间是按 lot 配对重抽样的每千张报警候选节省区间，固定已训练模型与队列。", "",
            "| 人群 | 种子 | RF77 换入 / 换出 | 真实划痕净数 | 正常片净数 | 节省／千张候选 [95% 区间] |",
            "|---|---:|---:|---:|---:|---:|"]
    for p in active:
        for s in SEEDS:
            r=paired[p,s,COST,"benefit_fixed"]
            def net(lab):
                v=swaps[p,s,COST,"benefit_fixed",lab]
                return int(v["entered"])-int(v["displaced"])
            lines.append(f"| {POPS[p]} | {s} | {integer(r['entered'])} / {integer(r['displaced'])} | {net('Scratch'):+d} | {net('none'):+d} | {decimal(r['paired_savings_per_1000'])} [{decimal(r['ci_low'])}, {decimal(r['ci_high'])}] |")
    dev=text_table((support/"relative_geometry_swaps.csv").read_bytes())
    samples=sorted([r for r in dev if r["scenario"]==COST and r["policy"]=="benefit_fixed"],key=lambda r:SEEDS.index(r["model_seed"]))
    require([r["model_seed"] for r in samples]==list(SEEDS), "Missing 04 development evidence")
    pool=json.loads((repo/"cloud_review/results/shift/pool_summary.json").read_bytes())
    pilot=json.loads((repo/"cloud_review/results/ssl_paired/pilot_summary.json").read_bytes())["primary_audit"]
    require((support/"shift_and_ssl.md").is_file(), "Missing unlabeled audit account")
    lines += ["", "## 开发审计、未标注池与自监督试验", "",
              f"在此前的开发审计中，相同的 RF77 换入策略在划痕损失五倍时每千张全来片节省 {ordered((r['savings_per_1000_all_arrivals'] for r in samples),decimal)} 单位。开发审计只有 11,591 张来片，不能将其成本或命中数直接套用到原测试人群。三种训练种子复用同一批晶圆。", "",
              f"638,507 张未标注晶圆只用于核对分布、批次和重复图，未参与收益结算。排除全部与有标签批次或同图连通的组件后，获得 {integer(pool['unique_candidate_maps'])} 张唯一候选图。50,000 张隔离未标注图的单种子配对试验中，监督训练预算相同，预训练模型额外使用一轮未标注计算。开发审计上，真实划痕命中由 {pilot['control_scratch_tp']} 增至 {pilot['pilot_scratch_tp']}，正常总误报由 {pilot['control_normal_any_fp']} 增至 {pilot['pilot_normal_any_fp']}，未通过联合约束。未标注尺寸覆盖无法推出哪一张原测试片能被模型救回。[未标注集的重叠、尺寸和失效 die 分布](shift_and_ssl.md)", "",
              "本轮能验证已见数据上的固定策略和重叠敏感性。仍待验证的是从未查看的独立批次、真实上游报警与每日到片节奏、实际人工检出率和漏损。这些条件会改变最佳复检规则。", ""]
    return "\n".join(lines)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",type=Path,required=True,help="07 aggregate directory with run_receipt.json")
    ap.add_argument("--support",type=Path,required=True,help="Directory with aggregate 04 swap CSV and shift_and_ssl.md")
    ap.add_argument("--repo",type=Path,required=True,help="Portfolio repository root")
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    text=report(args.run,args.support,args.repo)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(text,encoding="utf-8")
