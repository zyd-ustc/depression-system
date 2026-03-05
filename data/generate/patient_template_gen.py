import os
import json
import time
import re
from openai import OpenAI
from tqdm import tqdm
import random
# ==================== 配置区域 ====================


API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.deepseek.com" # 示例：DeepSeek API
MODEL_NAME = "deepseek-chat"

RARE_IDENTITIES = [
    "被AI绘画取代后转行做美甲的前原画师",
    "每天只负责给AI写代码纠错的被降薪资深程序员",
    "声音被开源模型克隆后失业的有声书配音员",
    "专门负责训练大模型'鉴黄'的数据标注员",
    "靠ChatGPT洗稿做自媒体矩阵但流量归零的操盘手",
    "只会写Prompt但不懂原理的'提示词工程师'（面临被优化）",
    "翻译腔严重的兼职字幕组成员（发现机翻比自己快）",
    "无人驾驶普及后无单可接的网约车老司机",
    "被数字人直播抢了饭碗的带货小主播",
    "在Deepfake色情视频中看到自己脸的受害者",
    "困在算法里、为了不超时哪怕逆行也要冲的外卖单王",
    "专门在电商平台帮人'刷恶评'以此勒索商家的职业差评师",
    "24小时待命、帮人抢演唱会门票的黄牛党底层",
    "在二手交易平台专门捡漏倒卖电子垃圾的'数码尸检官'",
    "陪诊师（看尽了医院人生百态自己却不敢体检）",
    "上门代厨/代收纳（常被雇主嫌弃手脚不干净的阿姨）",
    "在付费自习室住了三年、假装还在考研的'全职考生'",
    "专门帮游戏土豪代练、昼夜颠倒的'搬砖党'",
    "负责审核短视频暴力内容的平台审核员（PTSD患者）",
    "在大厂外包岗负责贴发票的30岁'螺丝钉'",
    "拿着父母退休金生活的35岁'全职儿女'（愧疚与理所应当并存）",
    "为了逃避催婚、过年哪怕花钱也要在这个城市住酒店的'恐归族'",
    "深陷杀猪盘、至今仍觉得对方爱自己的单身离异者",
    "和'纸片人'（乙女游戏角色）谈恋爱并为此负债的梦女/梦男",
    "隐瞒已婚事实、在交友软件上寻找心理慰藉的'假性单身'",
    "为了融入富二代圈子、拼单租爱马仕的'拼单名媛'",
    "长期遭受网络暴力但舍不得注销账号的小网红",
    "被迫要在朋友圈给老板每条动态点赞的'00后职场演员'",
    "断崖式分手后、每天视奸前任微博访客记录的'电子偷窥狂'",
    "在家族群里被亲戚嘲讽不如隔壁二狗的留守青年",
    "医美整容失败、戴着口罩不敢见人的修复期患者",
    "炒币/炒鞋爆仓、每天被催收电话轰炸的伪中产",
    "盲盒/谷圈（动漫周边）成瘾、家里堆满垃圾的囤积症患者",
    "为了维持精致露营/滑雪人设而刷爆信用卡的月光族",
    "买了烂尾楼、一边租房一边还贷的维权业主",
    "深信'灵修/身心灵'课程、花光积蓄寻求开悟的狂热信徒",
    "被健身房跑路卷走私教费的减肥焦虑者",
    "沉迷直播打赏、幻想主播能看自己一眼的'榜一大哥'（其实是工薪族）",
    "为了买绝版汉服/Lo裙而吃泡面的'吃土少女'",
    "在闲鱼上靠卖前任送的礼物维持生活的断舍离失败者",
    "在剧本杀店当DM（主持人）、每天被迫演戏的社恐",
    "专门写毁三观短剧剧本的枪手编剧",
    "深夜情感电台的主播（接收了太多负能量无法排解）",
    "在鹤岗（低房价城市）隐居、切断所有社交的自由职业者",
    "负责清理凶宅/孤独死现场的特殊清洁工",
    "在相亲角帮孩子举牌子、被无数人白眼的焦虑父母",
    "试图靠买彩票翻身的长期失业者",
    "被导师压榨、还要帮导师接送孩子的延毕博士生",
    "在直播间当'水军'、专门带节奏的键盘侠",
    "拥有高学历但找不到工作、只能去送快递的'孔乙己'"
    "读了8年才毕业的冷门专业博士（发现出道即失业，且年龄超35）",
    "卖房创业失败、背负连带责任债款的中年前老板",
    "被家里安排接班家族企业、但毫无经营兴趣与能力的'傀儡二代'",
    "从小练体育/艺术但因伤退役、无一技之长的退役特长生",
    "为了考编在家里脱产蹲了5年的'范进式'考生",
    "被教培行业整顿后、至今没转行成功的30岁前金牌讲师",
    "误入传销/资金盘、拉黑了所有亲戚朋友的'社会性死亡'者",
    "举全家之力留学回来、月薪只有4000的海归",
    "为了对象放弃一线城市工作回老家、结果被分手的'恋爱脑'牺牲者",
    "买在高点、首付亏光且房子卖不出去的'高位站岗'接盘侠",
    "三甲医院规培生（每天工作14小时，拿着微薄补贴，还要被病人家属骂）",
    "负责裁员谈话的HR（每天接收负能量，感觉自己是刽子手）",
    "大厂里的'背锅位'项目经理（专门负责推进必死项目的工具人）",
    "特殊教育学校老师（长期面对无法沟通的孩子，职业耗竭感极强）",
    "全职微商/带货宝妈（囤了一屋子货卖不出去，还得在朋友圈演成功）",
    "处理网络暴力的公关（每天看着成千上万条诅咒私信）",
    "私人银行理财经理（给客户亏了钱，每天活在被投诉的恐惧中）",
    "甚至没有名字的'影子写手'（帮大V写爆款书，自己一无所有）",
    "幼儿园男老师（长期处于性别刻板印象和家长不信任的视线中）",
    "投诉热线接线员（每天也是在挨骂中度过8小时）",
    "重男轻女家庭里的'扶弟魔'姐姐（工资全上交，感觉自己是耗材）",
    "被催婚逼到甚至想在大街上随便拉个人结婚的'大龄剩女/男'",
    "单亲妈妈（既要工作养家又要带娃，处于崩溃边缘）",
    "被迫出柜后被家族断绝关系的性少数群体",
    "长期照顾失能/阿尔兹海默症老人的独生子女（久病床前无孝子，全是绝望）",
    "被控制狂父母监控手机和社交圈的25岁乖乖女",
    "入赘女婿（在家毫无地位，长期压抑自尊）",
    "丧偶式育儿的家庭主妇（老公活着像死了，自己活得像保姆）",
    "产后抑郁且不被家人理解（被认为是矫情）的新手妈妈",
    "寄养在亲戚家、从小看脸色的留守儿童长大后的'讨好型人格'者",
    "在一线城市打拼10年最终决定回县城的'返乡青年'（无法适应人情社会）",
    "因为征信黑名单而无法找正经工作的老赖",
    "曾是大厂P8、现在开滴滴不敢告诉老婆的失业中年男",
    "被学霸光环绑架、无法接受自己平庸的'小镇做题家'",
    "鹤岗隐居失败、钱花光了又不敢回家的流浪者",
    "被算法困住、为了冲单量不敢上厕所的网约车司机",
    "在城中村蜗居、做着日结工作的'三和大神'精神继承者",
    "因为网赌输掉彩礼钱的准新郎",
    "被P2P爆雷卷走养老金的退休老人（虽然你主要要年轻人，但这个群体也会上网）",
    "因为学历造假/简历注水每天提心吊胆怕被背调的入职者",
    "严重的暴食/厌食症患者（在暴食和催吐的羞耻中循环）",
    "重度社恐（出门买东西都要做半天心理建设的'家里蹲'）",
    "因为长相/身材自卑而疯狂整容的'容貌焦虑重症'患者",
    "总是爱上渣男/渣女的'情感吸渣体质'（依恋损伤）",
    "曾是校园霸凌受害者、至今不敢直视别人眼睛的职场新人",
    "患有成人ADHD（多动症）导致工作频繁出错、被误解为懒惰的人",
    "性别认知障碍者（Thinking gender dysphoria）",
    "微笑抑郁症患者（白天是开心果，晚上自残）",
    "因为口吃/残疾而长期遭受歧视的边缘人",
    "过度迷信塔罗/算命、把人生寄托在玄学上的'精神流浪者'"
]




# 生成配置
TOTAL_PROFILES = len(RARE_IDENTITIES)  # 你想生成的总人数
OUTPUT_FILE = "cards.jsonl"

# ==================== 核心逻辑 ====================

def get_client():
    if not API_KEY:
        raise RuntimeError(
            "未检测到环境变量 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。请先设置后再运行。"
        )
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

ALLOWED_BIASES = [
    "妄下结论",
    "自我实现预言",
    "偏见强化效应",
    "以偏概全",
    "一刀切思维",
    "内外归因偏差",
    "情绪推理",
    "默认效应",
    "群体极化",
]


def _strip_control_chars(s: str) -> str:
    """
    去掉字符串中的不可见控制字符，避免写入/读取 jsonl 时触发 Invalid control character。
    保留常见的换行/回车/制表符（json.dumps 会自动转义它们）。
    """
    if not isinstance(s, str):
        s = str(s)
    return "".join(ch for ch in s if (ch >= " " or ch in "\n\r\t"))


def _normalize_bias(raw: str) -> str | None:
    if not raw:
        return None
    s = _strip_control_chars(str(raw)).strip()
    # 常见变体归一
    s = s.replace("“", "").replace("”", "").replace('"', "").replace("'", "")
    s = s.replace("一刀切", "一刀切思维")
    s = s.replace("二分思维", "一刀切思维")
    s = s.replace("偏见强化", "偏见强化效应")
    # 精确命中
    if s in ALLOWED_BIASES:
        return s
    # 容错：包含关系
    for b in ALLOWED_BIASES:
        if b in s:
            return b
    return None


def _llm_json_object(client, system_prompt: str, user_prompt: str, *, temperature: float = 1.0) -> dict | None:
    """
    让模型输出一个小 JSON object（越小越稳），失败则返回 None。
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "")
        # 这里 json.loads 只解析“短 JSON”，失败概率显著降低
        return json.loads(content)
    except Exception as e:
        print(f"[Error] JSON片段生成失败: {e}")
        return None


def _llm_text(client, system_prompt: str, user_prompt: str, *, temperature: float = 1.2) -> str | None:
    """
    让模型输出纯文本（不要求 JSON），失败则返回 None。
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        content = content.replace("```", "")
        return _strip_control_chars(content).strip()
    except Exception as e:
        print(f"[Error] 文本生成失败: {e}")
        return None


def generate_basic_info(client, seed_scenario: str) -> dict | None:
    system_prompt = (
        "你是一个擅长“非虚构写作”的当代作家。你要根据用户给出的核心场景，补全人物的基本信息。"
        "输出必须是合法JSON对象。"
    )
    user_prompt = f"""
【核心场景/人设种子】
{seed_scenario}

请只输出一个 JSON 对象，字段固定为：
{{
  "age": 18-45之间的整数,
  "occupation": "职业/状态（口语化但真实）",
  "mbti": "MBTI四字母（如INTJ/ENFP），不确定也要给一个最像的"
}}
不要输出其他字段，不要解释。
""".strip()
    obj = _llm_json_object(client, system_prompt, user_prompt, temperature=1.0)
    if not obj:
        return None
    # 轻量清洗/校验
    age = obj.get("age")
    try:
        age_int = int(str(age).strip())
    except Exception:
        age_int = None
    if age_int is None or age_int < 16 or age_int > 70:
        # 宽松一点，别卡死
        age_int = 28
    occupation = _strip_control_chars(str(obj.get("occupation", "")).strip())[:60]
    mbti = _strip_control_chars(str(obj.get("mbti", "")).strip().upper())[:8]
    if not mbti:
        mbti = "INFP"
    return {"age": age_int, "occupation": occupation, "mbti": mbti}


def generate_cognitive_bias(client, seed_scenario: str, basic_info: dict) -> str | None:
    system_prompt = (
        "你是一个熟悉认知偏差的写作者。你只需要在给定候选列表里选一个最贴合人物的。"
        "输出必须是合法JSON对象。"
    )
    bias_list = "，".join(ALLOWED_BIASES)
    user_prompt = f"""
【核心场景/人设种子】
{seed_scenario}

【人物基本信息】
{basic_info}

请从下列候选中选择“最贴合此人物长期思维模式”的一个，并只输出 JSON：
{{"cognitive_bias": "候选之一"}}

候选：{bias_list}
""".strip()
    obj = _llm_json_object(client, system_prompt, user_prompt, temperature=0.8)
    if not obj:
        return None
    bias = _normalize_bias(obj.get("cognitive_bias", ""))
    if not bias:
        # 兜底：随机给一个，保证结构完整
        bias = random.choice(ALLOWED_BIASES)
    return bias


def generate_inner_monologue(client, seed_scenario: str, basic_info: dict, cognitive_bias: str) -> str | None:
    system_prompt = """
你是一个擅长“非虚构写作”的当代作家。你需要写一个极具真实感的深夜树洞独白。

【核心指令】
1. 去AI味：严禁使用“首先、其次、总之、表现为”等结构；严禁教科书式心理学定义。
2. 口语化与时代感(2025)：可以自然提及具体的APP/功能/平台生态，但不要硬植入。
3. 目标：这是一个“抑郁相关困扰”的人，不要写成鸡汤或治疗宣传。
""".strip()
    user_prompt = f"""
【核心场景/人设种子】
{seed_scenario}

【人物基本信息】
{basic_info}

【认知偏差】
{cognitive_bias}

请写一段 180-260 字左右的中文“内心独白”，像他在深夜给树洞投稿/吐槽，不要像写日记。
要求：
- 体现这个身份背后的结构性矛盾与长期压力
- 语言里要“自然体现”你给的认知偏差（不要直接点名偏差词）
- 语气可以混乱、跳跃、带防御性或自嘲，但不要堆砌极端词
只输出独白文本，不要Markdown，不要标题。
""".strip()
    txt = _llm_text(client, system_prompt, user_prompt, temperature=1.4)
    if not txt:
        return None
    # 避免过长
    return txt[:800]


def generate_full_profile(client, seed_scenario):
    """
    第二步：基于具体的种子，扩展成完整的 JSON 档案。
    """
    # 改为“分段生成 -> 本地组装”，显著降低一次性大JSON解析失败概率
    basic_info = generate_basic_info(client, seed_scenario)
    if not basic_info:
        print("[Error] basic_info 生成失败")
        return None

    cognitive_bias = generate_cognitive_bias(client, seed_scenario, basic_info)
    if not cognitive_bias:
        print("[Error] cognitive_bias 生成失败")
        return None

    inner_monologue = generate_inner_monologue(client, seed_scenario, basic_info, cognitive_bias)
    if not inner_monologue:
        print("[Error] inner_monologue 生成失败")
        return None

    # 手动组装最终结构（写 jsonl 时由 json.dumps 负责转义）
    return {
        "basic_info": basic_info,
        "cognitive_bias": cognitive_bias,
        "inner_monologue": inner_monologue,
    }

def save_to_jsonl(data, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

# ==================== 主程序入口 ====================

def main():
    client = get_client()
    
    # 检查文件是否存在，不存在则创建
    if not os.path.exists(OUTPUT_FILE):
        open(OUTPUT_FILE, 'w', encoding='utf-8').close()

    print(f"=== 开始任务：生成 {TOTAL_PROFILES} 个 2025年抑郁症患者档案 ===")

    pbar = tqdm(total=TOTAL_PROFILES)
    generated_count = 0

    for i in range(TOTAL_PROFILES):
        seed = RARE_IDENTITIES[i]
        profile = generate_full_profile(client, seed)
            
        if profile:
            # 补充一些元数据
            profile['metadata'] = {
                'seed_scenario': seed,
                'timestamp': '2025-12-19', # 模拟的时间点
                'model': MODEL_NAME
            }
                
            save_to_jsonl(profile, OUTPUT_FILE)
            generated_count += 1
            pbar.update(1)
        else:
            print(f"跳过无效生成: {seed[:10]}...")
            
        # 稍微停顿，避免触发API频率限制
        time.sleep(0.5)

    pbar.close()
    print(f"\n✅ 全部完成！数据已保存至 {OUTPUT_FILE}")
    
    # === 打印一个样例供检查 ===
    print("\n=== 样例展示 (Inner Monologue) ===")
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                sample = json.loads(first_line)
                print(f"【人设】: {sample['metadata']['seed_scenario']}")
                print(f"【独白】: {sample['inner_monologue']}")
    except:
        pass

if __name__ == "__main__":
    main()
