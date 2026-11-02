"""Create a standalone Chinese HTML page from a verified aggregate decision evidence bundle."""
import argparse
import base64
import csv
import gzip
import html
import io
import json
from pathlib import Path
from zipfile import ZipFile

from render_evidence import AUDIT, FIGURES, MODELS, SEEDS, avg, match, pct, require, sha


STYLE = """
:root{font-family:system-ui,'Microsoft YaHei','Noto Sans CJK SC',sans-serif;color:#152b35;background:#edf3f4;line-height:1.72}
*{box-sizing:border-box}body{max-width:1090px;margin:auto;padding:1.5rem}header,section,nav,footer{padding:1.4rem 2rem;margin-bottom:1rem;background:white;border:1px solid #d7e1e4;border-radius:10px}
header{background:#153845;color:#f4f9f8;border:0}header small{letter-spacing:.08em;color:#b7e8d8}h1{font-size:clamp(1.65rem,5vw,2.7rem);line-height:1.18;margin:.65rem 0}
h2{margin:0 0 1rem;color:#104d55;font-size:1.42rem}h3{font-size:1.05rem;color:#104d55;margin:1.3rem 0 .25rem}p{margin:.75rem 0}
nav{display:flex;gap:.4rem 1.4rem;flex-wrap:wrap;padding-block:.72rem}a{color:#155d70;text-underline-offset:.17rem}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1.3rem}.kpis div{border-left:3px solid #82ceb0;padding-left:.8rem}.kpis b{display:block;font-size:1.45rem}.kpis span{color:#d3e2e4;font-size:.85rem}
.table-scroll{overflow:auto;max-width:100%}table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.89rem;width:100%}th,td{border-bottom:1px solid #dce3e5;padding:.45rem .55rem;text-align:right;white-space:nowrap}th{background:#e9f2f2;color:#193e45}th:first-child,td:first-child{text-align:left}tr:nth-child(even){background:#f7f9f9}
figure{margin:1.3rem 0}img{max-width:100%;height:auto}figcaption,footer{font-size:.85rem;color:#395562}code{overflow-wrap:anywhere}footer p{margin:.25rem 0}
details{margin:1rem 0}summary{cursor:pointer;font-weight:650;color:#155a6a}summary:hover{color:#0a323c}
.finding{background:#eaf4f1;border-left:4px solid #177565;padding:.65rem 1rem;margin-top:.75rem}
@media(max-width:640px){body{padding:.45rem}header,section,nav,footer{padding:1rem}.kpis{gap:.3rem}.kpis b{font-size:1.08rem}}
@media print{body{background:white;padding:0;max-width:none}nav{display:none}header{color:#152b35;background:white}header small,.kpis span{color:#29434b}section,header,footer{padding:.6rem;margin:.4rem 0;border:0}.table-scroll{overflow:visible}table{font-size:8pt}figure,tr{break-inside:avoid}}
"""


def esc(value):
    return html.escape(str(value), quote=True)


def paragraph(value):
    return "<p>" + esc(value) + "</p>"


def table(headers, rows):
    head = "".join(f'<th scope="col">{esc(v)}</th>' for v in headers)
    body = "".join("<tr>"+"".join(f"<td>{esc(v)}</td>" for v in row)+"</tr>" for row in rows)
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def section(anchor, heading, content):
    return f'<section id="{anchor}"><h2>{esc(heading)}</h2>{content}</section>'


def verified(read, hashes, name):
    b = read(name)
    require(sha(b)==hashes[name], f"Checksum mismatch: {name}")
    return b


def records(raw, zipped=False):
    b = gzip.decompress(raw) if zipped else raw
    reader = csv.DictReader(io.StringIO(b.decode("utf-8-sig"), newline=""))
    require(not {"source_row","lot","component","map_sha256"}.intersection(reader.fieldnames or []),
            "Expected aggregate table")
    return list(reader)


def prepared_sections(summary, figs):
    # The publisher owns all means, units, table columns and wording. Render its
    # restricted Markdown rather than implement a second version of those metrics.
    sections, group = {}, "intro"
    for block in summary.strip().split("\n\n"):
        if block.startswith("# "):
            continue
        if block.startswith("## "):
            group=block[3:].strip()
            continue
        if block.startswith("| "):
            rows=[[v.strip() for v in ln.strip().strip("|").split("|")] for ln in block.splitlines()]
            rendered=table(rows[0],rows[2:])
        elif block.startswith("!["):
            name=block.split("](",1)[1].rstrip(")")
            require(name in FIGURES, "Unexpected figure")
            title={"threshold_precision_recall_cost.png":"策略选择分区 · 缺陷概率阈值、精确率、召回率与每千片模拟成本",
                   "calibration_reliability.png":"开发审计分区 · 不同种子在温度校准前后的可靠性曲线"}[name]
            require(figs[name].startswith(b"\x89PNG\r\n\x1a\n"), "Expected PNG image")
            payload=base64.b64encode(figs[name]).decode("ascii")
            rendered=f'<figure><img alt="{esc(title)}" src="data:image/png;base64,{payload}"><figcaption>{esc(title)}</figcaption></figure>'
        else:
            rendered=paragraph(block)
        sections[group]=sections.get(group,"")+rendered
    return sections


def shift_section(root):
    if not root:
        return ""
    contents = []
    for stage, name in (("cost","audit_cost_strata.csv"),("pool","pool_summary.json")):
        folder=root/stage
        receipt_path=folder/"receipt.json"
        if not receipt_path.exists():
            folder=root
            receipt_path=folder/(stage+"_receipt.json")
        if not receipt_path.exists():
            continue
        receipt=json.loads(receipt_path.read_bytes())
        require(receipt["stage"]==stage, "Shift receipt stage mismatch")
        payload=verified(lambda n:(folder/n).read_bytes(),receipt["outputs"],name)
        if stage=="pool":
            p=json.loads(payload)
            contents.append(paragraph(f"无标签池沿批次和相同图像做传递隔离，再按图像去重。最终可用 {p['unique_candidate_maps']:,} 张独特图，来自 {p['candidate_components']:,} 个组件。多跳隔离比直接过滤又排除了 {p['unique_maps_removed_by_multihop']:,} 张图。池大小说明了可用的预训练数据量，识别效果需要与监督对照配对评估。"))
        else:
            rr=match(records(payload),model="RF77",seed="42",grouping="coverage")
            require(bool(rr),"No RF77 seed-42 coverage summary")
            contents.append(paragraph("尺寸覆盖按训练集里该尺寸正常片的数量分层。下表来自早先 RF77、种子 42 的固定复检队列。其成本单位为假设的相对损失，人工检查有效率 90%、划痕损失为普通缺陷的 5 倍。此表应单独解读。"))
            rows=[[r["coverage"],r["rows"],r["normal_flagged"],r["normal_reviewed"],
                   f'{float(r["upstream_missed_loss"]):,.1f}',f'{float(r["capacity_missed_loss"]):,.1f}'] for r in rr]
            contents.append(table(["训练正常片覆盖","审计片","正常片被初筛标记","正常片被复检","初筛漏检损失","名额不足损失"],rows))
            zero=next((r for r in rr if r["coverage"]=="0"),None)
            if zero:
                contents.append(paragraph(f"训练时未见同尺寸正常片的组有 {int(zero['rows']):,} 张审计片，占本次模拟剩余损失的 {float(zero['share_all_residual_loss']):.2%}。这些尺寸组内混合了正常片和缺陷片，损失也包含名额不足造成的漏检，不能直接当作尺寸的因果效应。"))
    require(contents,"No completed cost or pool receipt in --shift")
    return section("sizes","尺寸覆盖与未标注晶圆","".join(contents))


def guarded_section(root, read, get, note, freeze):
    from guarded_selection import ARMS, SELECT, TABLES, compare, stats
    summary=json.loads((root/"guarded_summary.json").read_bytes())
    selected=json.loads((root/"selection_freeze.json").read_bytes())
    require(all(summary[k]==v for k,v in selected.items()) and selected["main_policy_sha256"]==note["source_policy_sha256"]
            and selected["source_receipt_sha256"] in (sha(read("manifest.json")),note["source_receipt_sha256"]),
            "Guarded selection lineage mismatch")
    data={k:gzip.decompress(get(k+".csv.gz")) for k in TABLES}
    require({k:sha(v) for k,v in data.items()}==selected["source_table_sha256"],"Guarded aggregate input mismatch")
    sr,eligible,means=compare(SELECT,stats(data,SELECT))
    ar,audit,_=compare(AUDIT,stats(data,AUDIT))
    candidates=[m for m in ARMS if eligible[m]]
    best=max(candidates,key=means.get)
    require(candidates==selected["eligible_candidates"] and best==selected["frozen_arm"]
            and audit==summary["audit_all_three_all_seeds"] and freeze["downstream_model"]==selected["main_unconstrained_choice"],
            "Guarded summary does not match source aggregates")
    content=paragraph("在每个种子上与同种子的监督交叉熵 CNN 比较。只要真实划痕命中数下降、正常总误报数增加或正常→划痕数增加，就失去候选资格。先在选择分区检验三个种子全部合格，再按平均 Macro-F1 选择，审计差值只用于复查。")
    rows=[]
    for m in ARMS:
        counts=[sum(r["arm"]==m and r["all_three_hold"] for r in rr) for rr in (sr,ar)]
        rows.append([MODELS[m], f"{float(means[m]):.2%}",f"{counts[0]}/3",f"{counts[1]}/3","入选" if m==best else ""])
    content+=table(["匹配训练臂","验证 Macro-F1","验证合格种子","审计合格种子","冻结选择"],rows)
    content+=paragraph(f"验证合格名单为 {', '.join(MODELS[m] for m in candidates)}。"+(
        "与自身比较的三项差值都为零。四种训练改动均未达到替代监督基线的条件。" if candidates==["cnn_ce"] else "此处报告冻结后的审计表现。"))
    return section("guarded","训练改动是否守住三个约束",content)


def ssl_section(root, pack, main, note, classification):
    status=json.loads((root/"run_status.json").read_bytes())
    require(status["status"]=="complete" and status["seed"]==42,"Incomplete SSL evaluation")
    get=lambda n:verified(lambda p:(root/p).read_bytes(),status["sha256"],n)
    summary=json.loads(get("pilot_summary.json"))
    sf=json.loads(get("pilot_validation_freeze.json"))
    require(sf["component_split_salt"]==main["component_salt"] and sf["archive_manifest_sha256"]==main["archive_manifest_sha256"]
            and sf["decision_source_sha256"]==main["source_sha256"] and sf["primary_rule"]==summary["primary_rule"]=="argmax",
            "SSL/main protocol mismatch")
    intro="从与有标签组件隔离的未标注晶圆学习重建被遮挡的区域，再按监督 CNN 的相同预算微调。"
    if pack:
        b=pack.read_bytes()
        require(sha(b)==sf["pack_receipt_sha256"],"SSL pack mismatch")
        receipt=json.loads(b)
        require(receipt["status"]=="complete" and receipt["pool_receipt_sha256"]==sf["pool_receipt_sha256"], "SSL pool mismatch")
        intro+=f"本次抽取 {receipt['maps']:,} 张图，预训练 {receipt['policy']['ssl_epochs']} 轮，完成 {receipt['policy']['ssl_steps']} 次更新。"
    intro+="固定种子 42，主指标使用直接九分类，另作冻结划痕阈值的补充对照。"
    source=records(get("paired_scratch_guard.csv"))
    rows=[]
    for k,name in (("primary_validation","选择集 · 直接分类"),("primary_audit","审计 · 直接分类"),("secondary_audit","审计 · 划痕优先")):
        r=summary[k]
        original=next(x for x in source if (x["phase"],x["rule"])==(r["phase"],r["rule"]))
        compare_phase="selection" if k=="primary_validation" else AUDIT
        base=match(classification,phase=compare_phase,seed="42",model="cnn_ce",rule=r["rule"])[0]
        cols=["scratch_tp","normal_any_fp","normal_scratch_fp"]
        require(all(int(original[prefix+j])==r[prefix+j] for prefix in ("control_","pilot_") for j in cols)
                and all(int(base[j])==r["control_"+j] for j in cols),"Pilot matched-CE metrics disagree")
        rows.append([name]+[f"{r['control_'+j]} → {r['pilot_'+j]} ({r[j+'_delta']:+d})" for j in cols]
                    +["通过" if r["all_three_hold"] else "未通过"])
    content=paragraph(intro)+table(["监督 CE → 自监督微调","真划痕命中","正常总误报","正常→划痕","三项联合"],rows)
    r=summary["primary_audit"]
    content+=paragraph(f"审计共有 {r['scratch_n']} 张划痕和 {r['normal_n']:,} 张正常片。直接分类的划痕命中改变量为 {r['scratch_tp_delta']:+d}，正常总误报改变量为 {r['normal_any_fp_delta']:+d}。"+
                       ("该单种子试验通过了验证与审计的主约束。" if summary["held_primary_in_both"] else "该单种子试验没有同时守住主约束，保留监督 CE 对照。"))
    c=summary["paired_capacity_cost"]
    content+=paragraph(f"在共同候选池、相同名额和划痕 $3,000 的条件下，自监督相对 CE 的节省为每千片 {c['savings_per_1000_arrivals']:,.2f} 美元（正数表示节省）。按 {c['components']:,} 个组件配对重采样的 95% 区间为 [{c['lo_95']:,.2f}, {c['hi_95']:,.2f}]。这份区间针对固定种子和队列。")
    if c["lo_95"]<=0<=c["hi_95"]:
        content+=paragraph("区间跨过零，无法据此认定自监督降低了复检成本。")
    return section("ssl","单种子自监督是否值得替换监督模型",content)


def make(read, shift=None, guarded=None, ssl=None, pack=None):
    note=json.loads(read("manifest.json"))
    get=lambda n:verified(read,note["files"],n)
    summ=prepared_sections(get("SUMMARY.zh-CN.md").decode("utf8"),{n:get(n) for n in FIGURES})
    freeze=json.loads(get("frozen_validation_policy.json"))
    model=freeze["downstream_model"]
    cname=MODELS[model]
    totals=match(records(get("split_support.csv.gz"),True),phase="audit")
    require(len(totals)==9 and len({int(r["total"]) for r in totals})==1, "Inconsistent audit class totals")
    n=int(totals[0]["total"])
    require(n==sum(int(r["rows"]) for r in totals),"Audit support does not reconcile")
    normal=int(match(totals,label="none")[0]["rows"])
    scratch=int(match(totals,label="Scratch")[0]["rows"])
    cards="".join(f"<div><b>{number:,}</b><span>{name}</span></div>" for number,name in
                  ((n,"已见开发审计片"),(normal,"真实正常片"),(scratch,"真实划痕片")))
    head=f'<header><small>WM-811K · 开发实验</small><h1>划痕识别与有限名额复检</h1><p>从同一批晶圆出发，分别核对缺陷类别、正常片误报和送检成本。</p><div class="kpis">{cards}</div></header>'
    historical=paragraph("历史 RF、CNN 在批次验证上的 Macro-F1 为 85.28%、84.73%，在原始冻结测试上分别降到 44.87%、45.16%。之后已查看测试错误并据此设计修复，当前开发结果不能证明最终测试表现恢复。泛化结论仍需新的独立、有标签批次。")
    historical+=paragraph("原始 811,457 张图中有 172,950 张具备标签。历史开发集 54,355 张，正常占 67.57%；原始冻结测试 118,595 张，正常占 93.34%。此前的“约 93% 正常”对应原始测试。真实划痕在两个分区分别占 0.92% 和 0.58%。")
    historical+=paragraph("在原始测试正常片中，70.60% 的精确尺寸从未出现在开发集的正常片中。CNN 将正常片认成划痕的错误里，97.31% 来自开发集中该尺寸正常片只有 0–24 张的组。一次调整 BatchNorm 统计量的诊断虽抑制了部分误报，真实划痕召回却从 53.97% 降至 12.12%。尺寸覆盖有明确关联，尚不能单独解释整个 Macro-F1 降幅。")
    historical+='<p><a href="https://github.com/Matthewsyr/reliable-wafer-classification/blob/main/reports/Project_Report.md">历史实验与尺寸诊断</a> · <a href="https://github.com/Matthewsyr/reliable-wafer-classification/blob/main/docs/TECHNICAL_SUMMARY.zh-CN.md">中文技术总结</a></p>'
    split=paragraph("同一生产批次和图像哈希完全相同的晶圆归入同一连接组。训练片更新参数，tune 的两个组件子集分别拟合温度、选择策略；冻结后只在原开发 audit 上复查。")
    split+=summ["样本与划分"]+paragraph(f"三个种子反复使用这 {scratch:,} 张真实划痕。种子之间的变化反映重训稳定性，独立划痕样本数仍为 {scratch:,}。")
    policy_model=MODELS[freeze["lowest_queue_cost_model"]]
    criterion=(f"在五个匹配 CNN 臂中，按策略选择分区上三种子平均 Macro-F1 选中了 {cname}。"
               if note["training_extension"] else f"此次回放直接使用预先指定的 {cname} 作为下游策略模型。")
    choice=paragraph(criterion+f"若改为按固定候选池、相同复检名额和划痕 $3,000 的模拟成本比较全部候选模型，验证分区上成本最低的是 {policy_model}。选择目标和候选范围决定这个结果。")
    classification=records(get("classification.csv.gz"),True)
    pred=match(classification,phase=AUDIT,model=model)
    lookup={(r["rule"],int(r["seed"])):r for r in pred}
    rows=[]
    for seed in sorted(SEEDS):
        a,b=lookup["argmax",seed],lookup["scratch_joint",seed]
        require(int(a["scratch_n"])==int(b["scratch_n"])==scratch and int(a["normal_n"])==int(b["normal_n"])==normal,
                "Scratch or normal denominator changed between rules")
        rows.append([seed]+[a[k]+" → "+b[k] for k in ("scratch_tp","normal_scratch_fp","normal_any_fp")])
    guard=paragraph(f"对 {cname} 的同一组概率，比较直接取最高分类概率与冻结的划痕优先规则。若正常→划痕减少而正常总误报没有减少，应检查是否有正常片转成了其他缺陷；真实划痕命中数也要同步核对。")
    guard+=table(["种子","真划痕命中数","正常→划痕","正常→任意缺陷"],rows)
    guard+=paragraph("每格左侧为直接分类，右侧为冻结的联合约束策略。九分类正确命中划痕要求标签认对；送检策略只要求晶圆被检出，人工可在复检时纠正缺陷类别。")
    cost=paragraph("排序分数为各缺陷概率乘漏检损失后求和，再减去正常概率乘 $5。在同一批候选里，划痕风险较高的晶圆即使总缺陷概率稍低，也可能获得较高优先级。")
    if note["training_extension"]:
        ce,sampled=[match(classification,phase=AUDIT,model=m,rule="argmax") for m in ("cnn_ce","cnn_sampled")]
        choice+=f'<p class="finding">加权采样使审计划痕召回从 {pct(ce,"scratch_recall")} 变为 {pct(sampled,"scratch_recall")}，正常总误报率也从 {pct(ce,"normal_any_fpr")} 变为 {pct(sampled,"normal_any_fpr")}。两项的分母分别是 {scratch} 张真实划痕和 {normal:,} 张正常片。</p>'
    classes=match(records(get("per_class.csv.gz"),True),phase=AUDIT,model=model,rule="argmax")
    by_label=[]
    for label in dict.fromkeys(r["label"] for r in classes):
        group=match(classes,label=label)
        by_label.append([label,group[0]["support"]]+[pct(group,c) for c in ("precision","recall","f1","average_precision")])
    per_class="<details><summary>查看入选模型的九类审计指标</summary>"+table(["类别","样本数","精确率","召回率","F1","平均精度 AP"],by_label)+"</details>"
    routing=match(records(get("routing.csv.gz"),True),phase=AUDIT,model=model,scenario="scratch_3000")
    prob,net,unlimited=[match(routing,policy=m) for m in ("queue_probability","queue_expected_value","min_cost")]
    cost+=paragraph(f"在候选池、名额、模型都固定时，按净收益排序和按任意缺陷概率排序平均分别复检到 {avg(net,'scratch_captured')} 和 {avg(prob,'scratch_captured')} 张真实划痕。按净收益排序也复检了 {avg(net,'fp')} 张正常片。无名额上限的经验成本阈值要检 {avg(unlimited,'reviewed')} 张，两种方案承担的复检工作量不同。")
    parts=[section("scope","哪些结果可以比较",summ["intro"]+historical),
           section("splits","数据划分与实际样本数",split),
           section("models","模型训练与九分类预测",choice+summ["训练方法与九分类性能"]+per_class),
           section("scratch","真实划痕与正常片误报",guard),
           section("triage","复检阈值、能力与模拟损失",cost+summ["阈值、检验名额与成本"])]
    extra=shift_section(shift)
    nav=[("scope","证据边界"),("splits","划分"),("models","模型"),("scratch","划痕与误报"),("triage","复检与成本")]
    if extra:
        parts.append(extra)
        nav.append(("sizes","尺寸覆盖"))
    if guarded:
        parts.insert(3,guarded_section(guarded,read,get,note,freeze))
        nav.insert(3,("guarded","逐种子选模"))
    if ssl:
        parts.append(ssl_section(ssl,pack,freeze,note,classification))
        nav.append(("ssl","自监督试验"))
    links="".join(f'<a href="#{key}">{label}</a>' for key,label in nav)
    foot=f'<footer><p>数据来源为已核验的公开聚合包 · {int(note["model_seed_pairs"])} 组模型与种子组合。</p><p>原云端完成收据 SHA-256　<code>{esc(note["source_receipt_sha256"])}</code></p></footer>'
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WM-811K · 划痕识别与复检</title><style>'+STYLE+'</style><body>'+head+'<nav aria-label="章节">'+links+'</nav><main>'+"".join(parts)+'</main>'+foot+'</body></html>'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence",required=True,type=Path,help="render_evidence output directory or aggregate_evidence.zip")
    p.add_argument("--shift",type=Path,help="Optional parent of verified cost/ or pool/ from shift_followup.py")
    p.add_argument("--guarded",type=Path,help="Optional complete guarded_selection output")
    p.add_argument("--ssl",type=Path,help="Optional complete ssl_evaluate output")
    p.add_argument("--ssl-pack",type=Path,help="Optional 50k pack receipt for linkage to completed SSL evaluation")
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if a.evidence.is_dir():
        page=make(lambda n:(a.evidence/n).read_bytes(),a.shift,a.guarded,a.ssl,a.ssl_pack)
    else:
        with ZipFile(a.evidence) as archive:
            page=make(archive.read,a.shift,a.guarded,a.ssl,a.ssl_pack)
    with a.output.open("x",encoding="utf8",newline="\n") as f:
        f.write(page)
    print(f"Standalone HTML: {a.output} ({a.output.stat().st_size:,} bytes).")


if __name__=="__main__":
    main()
