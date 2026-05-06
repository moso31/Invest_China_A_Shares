---
name: estimate
description: 对单只 A 股计算估值带与策略适配分。策略基线为「长线股息持有(>1年免红利税) + 小幅做T」。当需要判断个股是否便宜、是否适合入仓、给出买入/做T参考价位时使用本技能。
---

# A 股估值技能

本文件是 `Strategy.md` 的实现细则。统一口径为：机器排序只使用 `D/V/T` 三个连续分；`Q` 保留为企业质量复核项，能连续度量的现金流、ROE 稳定性、负债与成长退化吸收到 `D/V`；`R` 只输出风险 flags 和加仓门控，不作为线性扣分项进入总分。

## 0. 何时使用

调用本技能当且仅当满足:
- 对象是单只 A 股(沪/深/北),且已有申万三级行业归属
- 已具备 TTM 财务、5 年以上日频历史 PE/PB、近 3 年分红数据
- 需要输出: 估值带 / 估值分位 / 策略适配分 / 风险标记

不适用: 港股、美股 ADR、刚 IPO < 1 年的次新股(历史不足)、ST/退市风险股。

## 1. 输入契约

```python
@dataclass
class StockInput:
    # 标识
    symbol: str                    # "600519"
    industry: str                  # 申万三级名称,如"白酒Ⅱ"
    price: float                   # 当前收盘价
    market_cap: float              # 总市值(元)

    # 财务 TTM
    net_profit_ttm: float
    revenue_ttm: float
    equity: float                  # 归母净资产
    ocf_ttm: float                 # 经营现金流
    fcf_ttm: float                 # 自由现金流(可选)
    fcf_per_share: float | None    # 每股自由现金流(可选; 当前 AKShare 缓存口径更稳定)
    total_debt: float              # 有息负债
    cash: float

    # 质量
    roe_ttm: float                 # %
    roe_3y_avg: float              # %
    roe_5y_std: float              # ROE 5 年标准差,衡量盈利稳定性
    gross_margin_ttm: float        # %
    debt_to_asset: float           # %
    accounts_receivable: float | None  # 应收账款(可选; R 使用)
    inventory: float | None            # 存货(可选; R 使用)

    # 分红
    dps_history: list[float]       # 近 5 年每股现金分红,旧→新; 缺失/上市前年份填 0
    payout_ratio_3y_avg: float     # 近 3 年分红率均值

    # 历史时序(5 年,日频)
    pe_ttm_series: "pd.Series"     # index=date
    pb_series: "pd.Series"
    close_series: "pd.Series"
    volume_series: "pd.Series"     # 成交额(元)

    # 行情衍生
    beta_60d: float                # vs 沪深300
    vol_60d_annual: float          # 60 日年化波动率,小数(0.30 = 30%)

    # 可选: 一致预期(若不可用置 None)
    eps_forecast_y1: float | None  # 下一年度 EPS 预测
    growth_3y_cagr: float | None   # 历史 3 年净利复合增速,%
```

## 2. 输出契约

```python
@dataclass
class ValuationResult:
    bucket: str                    # 8 个 bucket 之一
    method: str                    # 主估值方法
    fair_low: float                # 每股
    fair_mid: float                # 每股
    fair_high: float               # 每股
    margin_of_safety: float        # (fair_mid - price) / price

    pe_percentile_5y: float        # 0-1, 越低越便宜
    pb_percentile_5y: float

    dividend_score: float          # 0-100
    value_score: float             # 0-100
    t_score: float                 # 0-100
    strategy_fit: float            # 0-100, 综合

    t_band_low: float              # 做 T 下沿(建议补仓)
    t_band_high: float             # 做 T 上沿(建议减仓)

    flags: list[str]               # 红旗,见 §7
    notes: list[str]               # 解释性说明
```

## 3. 主流程

```
def evaluate(stock: StockInput) -> ValuationResult:
    bucket = MAP_INDUSTRY[stock.industry]                    # §4
    method = BUCKET_RULES[bucket]["method"]                  # §5

    fair_low, fair_mid, fair_high = compute_fair_value(      # §5
        bucket, stock
    )

    pe_pct = percentile(stock.pe_ttm_series, current=stock.pe_ttm_series.iloc[-1])
    pb_pct = percentile(stock.pb_series, current=stock.pb_series.iloc[-1])

    div = dividend_score(stock)                              # §6.1
    val = value_score(stock, bucket, pe_pct, pb_pct)         # §6.2
    t   = t_score(stock)                                     # §6.3

    fit = 0.45*div + 0.35*val + 0.20*t                       # 策略权重; Q/R 不进总分

    t_low, t_high = t_band(stock, fair_mid)                  # §6.3

    flags = run_red_flags(stock, bucket)                     # §7
    return ValuationResult(...)
```

## 4. 行业 → Bucket 映射

8 个 bucket。**给定一个申万三级名,必须落到唯一 bucket**。

| Bucket | 估值核心 | 收录的申万三级 |
|---|---|---|
| `PE_STABLE` | PE + 历史分位 | 白酒Ⅱ, 非白酒, 饮料乳品, 食品加工, 调味发酵品Ⅱ, 休闲食品, 农产品加工, 个护用品, 化妆品, 一般零售, 旅游零售Ⅱ, 专业连锁Ⅱ, 互联网电商, 贸易Ⅱ, 白色家电, 黑色家电, 厨卫电器, 小家电, 家电零部件Ⅱ, 照明设备Ⅱ, 化学制药, 中药Ⅱ, 生物制品, 医疗服务, 医疗美容, 医疗器械, 医药商业, 动物保健Ⅱ, 服装家纺, 家居用品, 文娱用品, 旅游及景区, 酒店餐饮, 教育, 饰品, 汽车服务, 摩托车及其他, 软件开发, 计算机设备, 通信设备, 元件, 光学光电子, 消费电子, 电子化学品Ⅱ, 其他电子Ⅱ, 出版, 广告营销, 电视广播Ⅱ, 数字媒体, 游戏Ⅱ, 体育Ⅱ, 影视院线, 通用设备, 专用设备, 自动化设备, 轨交设备Ⅱ, 工程咨询服务Ⅱ, 专业服务, 综合Ⅱ, 装修建材, 装修装饰Ⅱ, 包装印刷, 农化制品, 非金属材料Ⅱ, 金属新材料, 燃气Ⅱ, 环保设备Ⅱ, 环境治理 |
| `PB_CYCLICAL` | PB + 周期均值 PE | 钢铁(普钢, 特钢Ⅱ, 冶钢原料), 化学原料, 化学制品, 化学纤维, 橡胶, 塑料, 造纸, 玻璃玻纤, 水泥, 工业金属, 小金属, 工程机械, 商用车, 乘用车, 汽车零部件, 纺织制造, 焦炭Ⅱ, 油服工程, 炼化及贸易, 半导体, 电池, 光伏设备, 风电设备, 电机Ⅱ, 其他电源设备Ⅱ, 能源金属, 房屋建设Ⅱ, 基础建设, 专业工程 |
| `BANK` | PB-ROE 双因子 | 银行Ⅱ |
| `INSURANCE` | P/EV(若有) 否则 PB | 保险Ⅱ |
| `SECURITIES` | PB | 证券Ⅱ |
| `DIV_INCOME` | 股息率 + EV/EBITDA | 电力, 电网设备, 煤炭开采, 油气开采Ⅱ, 铁路公路, 航空机场, 航运港口, 物流, 通信服务, 多元金融 |
| `NAV_RE` | NAV + PB | 房地产开发, 房地产服务 |
| `CYCLICAL_AG` | 头均市值 + PB | 养殖业, 种植业, 渔业, 饲料, 农业综合Ⅱ, 林业Ⅱ |
| `DEFENSE` | PE(高估值容忍) + 订单 | 航空装备Ⅱ, 航天装备Ⅱ, 地面兵装Ⅱ, 航海装备Ⅱ, 军工电子Ⅱ |

> 注:`DEFENSE` 单列因 PE 中枢系统性高于 `PE_STABLE`。
> `PB_CYCLICAL` 涵盖了"传统周期 + 转入周期定价的新能源中后段"——电池/光伏/风电近 2 年已切换为 PB 估值范式。

## 5. 各 Bucket 处理规则

### 5.1 `PE_STABLE`

```
method = "historical_pe_band"
pe_low  = 5y PE 25 分位
pe_mid  = 5y PE 50 分位
pe_high = 5y PE 75 分位
eps_ttm = net_profit_ttm / shares
fair_low  = pe_low  * eps_ttm
fair_mid  = pe_mid  * eps_ttm
fair_high = pe_high * eps_ttm
```

特例:
- 若 `roe_5y_std > 8`(盈利不稳),退化到 `PB_CYCLICAL` 处理
- 若 `growth_3y_cagr > 25%` 且 `eps_forecast_y1` 可用,**额外**计算 `peg = pe_ttm / growth_3y_cagr`,作为 note 报出
- 互联网电商 / 专业连锁 / 一般零售优先看 `调整后 PE`;若 `net_profit_ttm < 0`,换用 `EV/Sales` (mid = 行业近 3 年中位数)

### 5.2 `PB_CYCLICAL`

```
method = "pb_band_with_cycle_pe"
pb_low  = 5y PB 20 分位
pb_mid  = 5y PB 50 分位
pb_high = 5y PB 80 分位
bvps    = equity / shares
fair_low_pb  = pb_low  * bvps
fair_mid_pb  = pb_mid  * bvps
fair_high_pb = pb_high * bvps

# 周期均值 PE 校验(避免顶部低估)
avg_profit_5y = mean(近 5 年净利润)
cycle_pe_mid  = 历史可比公司中枢,见下表
fair_mid_cyc  = cycle_pe_mid * (avg_profit_5y / shares)

fair_mid = min(fair_mid_pb, fair_mid_cyc)   # 取保守值
```

| 子领域 | cycle_pe_mid | PB 中枢参考 |
|---|---|---|
| 钢铁 | 8 | 0.8 |
| 化工大宗 | 12 | 1.5 |
| 水泥/玻璃 | 10 | 1.2 |
| 工业金属 | 12 | 2.0 |
| 造纸 | 10 | 1.0 |
| 工程机械 | 13 | 1.8 |
| 整车(乘用车) | 14 | 1.5 |
| 半导体 | 35 | 3.0 |
| 锂电/光伏/风电 | 18 | 2.0 |
| 焦炭/油服 | 10 | 1.2 |
| 房屋建设/基建 | 7 | 0.7 |

> 红旗:`PB_CYCLICAL` 中若 `pe_ttm < 5` 且 `pb > 5y PB 70 分位`,**强烈疑似周期顶**,触发 flag `cycle_top_risk`。

### 5.3 `BANK`

```
method = "pb_roe"
# 合理 PB ≈ (ROE - g) / (Re - g),取 g=3%, Re=10%
implied_pb = max(0, (roe_3y_avg/100 - 0.03) / (0.10 - 0.03))
bvps = equity / shares
fair_mid = implied_pb * bvps
fair_low  = fair_mid * 0.85
fair_high = fair_mid * 1.15
```

强制 flag:
- `pb < 0.5 and roe < 8`: `state_bank_value_trap`(国有大行陷阱形态,但常态)
- 不良率 > 2% 或 拨备覆盖率 < 150%: `asset_quality_warning`(若数据可得)

### 5.4 `INSURANCE`

```
若 P/EV 数据可得:
    method = "p_ev"
    fair_mid = 0.8 * EV_per_share        # A 股寿险中枢 0.6-1.0
    fair_low = 0.55 * EV_per_share
    fair_high = 1.0 * EV_per_share
否则退化:
    method = "pb"
    沿用 BANK 规则但 g=2%, Re=11%(寿险更高股权成本)
```

### 5.5 `SECURITIES`

```
method = "pb_band"
pb_low  = max(0.9, 5y PB 15 分位)
pb_mid  = 5y PB 50 分位
pb_high = min(2.5, 5y PB 85 分位)
fair_x  = pb_x * bvps
```

> 证券股 PE 失真,严禁用 PE 估值。

### 5.6 `DIV_INCOME` ⭐ 本策略主战场

```
method = "ddm_plus_div_yield"

# 1. 当前股息率(用最近 1 年现金分红)
div_yield = dps_history[-1] / price

# 2. 5 年股息率分位
div_yield_series = dps_history_aligned / close_series   # 需要做时间对齐
dy_pct = percentile(div_yield_series, current=div_yield)

# 3. 简化 DDM(常增长模型)
g_div = min(0.05, mean_growth(dps_history, 3))          # 上限 5%
required_return = 0.08                                   # 长债收益率 + 风险溢价
fair_mid = dps_history[-1] * (1 + g_div) / (required_return - g_div)
fair_low = fair_mid * 0.85
fair_high = fair_mid * 1.20

# 4. 校验: 高分红的"现金牛锚"
target_div_yield_low  = 0.04   # 4% 进入观察
target_div_yield_high = 0.06   # 6% 进入买入
fair_high_check = dps_history[-1] / target_div_yield_low
fair_low_check  = dps_history[-1] / target_div_yield_high
fair_mid = (fair_mid + (fair_high_check + fair_low_check)/2) / 2
```

特殊处理:
- **煤炭/油气**: 当 `pe_ttm > 5y PE 70 分位` 时,商品价格可能见顶,fair_mid * 0.85 风险折价
- **铁路公路/电力(水电)/运营商**: 视为类公用事业,`required_return` 调降至 0.07
- **航空机场**: 周期重,`g_div` 设为 0,且若 `dps_history` 有断档(疫情)用 2019 年值替代当年

### 5.7 `NAV_RE`(房地产开发)

```
method = "nav_pb_floor"
# A 股开发商 PB 长期 0.3-0.8,以 PB 为主轴
央国企标签 = is_central_soe(symbol)   # 由调用方提供
if 央国企标签:
    pb_target_mid = 0.7
    pb_target_low = 0.5
    pb_target_high = 1.0
else:
    pb_target_mid = 0.4
    pb_target_low = 0.25
    pb_target_high = 0.7
fair_x = pb_target_x * bvps
```

强制 flag:
- `debt_to_asset > 75`: `re_high_leverage`
- `cash_short_debt_ratio < 1.0`: `re_liquidity_risk`（见下方计算口径）
- 民营开发商默认追加 flag `re_private_developer_risk`

#### `re_liquidity_risk` 计算口径

数据来源：`ak.stock_financial_em(stock, symbol="资产负债表")`，对字段名做模糊匹配（东财口径存在版本漂移）。

```python
# 字段候选（按优先级）
cash_candidates    = ["货币资金", "现金及现金等价物"]
st_loan_candidates = ["短期借款", "短期债务"]
st_due_candidates  = ["一年内到期的非流动负债", "一年内到期非流动负债"]

# 受限货币资金经验折扣
# 预售资金监管账户的钱只能用于本项目工程款，无法偿债。
# akshare 主表不暴露"受限货币资金"字段，用行业经验近似：
restriction_factor = 0.2 if is_central_soe else 0.5
effective_cash     = 货币资金 × (1 - restriction_factor)

st_debt = 短期借款 + 一年内到期非流动负债   # 最小可用版本，未含有息商票
cash_short_debt_ratio = effective_cash / st_debt
# ratio < 1.0 → 触发 re_liquidity_risk
```

> `is_central_soe` 在当前实现中以公司名称关键词做近似识别（保利/招商/华润/华发/越秀/陆家嘴/金融街/首开）；若 `name` 未传入则默认按民营折扣（0.5），偏保守，央国企可能误报。

### 5.8 `CYCLICAL_AG`

养殖股(尤其生猪)以**头均市值**为主:

```
method = "per_head_value"
# 由调用方提供年出栏量(头数)
per_head_market_cap = market_cap / annual_output_heads
# 生猪行业经验区间(随周期更新):
#   底部估值: 4000-6000 元/头
#   中枢:    6000-9000 元/头
#   顶部:    > 12000 元/头
fair_mid = 7500 * annual_output_heads / shares
fair_low = 5000 * annual_output_heads / shares
fair_high = 11000 * annual_output_heads / shares

# 校验: 周期均值 PE
若 5 年净利平均为正:
    cycle_pe = 15
    fair_check = cycle_pe * mean_profit_5y / shares
    fair_mid = (fair_mid + fair_check) / 2
```

非生猪养殖(白鸡、水产、种植)直接用 `PB_CYCLICAL` 规则,`cycle_pe_mid=15, pb_mid=2.0`。

### 5.9 `DEFENSE`

```
method = "pe_band_high_tolerance"
pe_low  = max(25, 5y PE 25 分位)
pe_mid  = max(35, 5y PE 50 分位)
pe_high = min(60, 5y PE 75 分位)
```

> 军工长期高估值,严禁套用 `PE_STABLE` 的中枢。
> 军工股普遍**不分红或低分红**,在本策略下 `dividend_score` 多在 30 分以下,自动被 `strategy_fit` 降权,无需额外处理。

## 6. 评分函数

### 6.1 `dividend_score` (0-100)

针对**长线股息+免五**核心要求。D 不只看股息率，还要吸收 Q 中能直接度量的现金流质量与盈利稳定性。

```
def dividend_score(s: StockInput) -> float:
    score = 0

    # (a) 5 年加权股息率 X (60 分)
    # 权重按近→远排列; 缺失/上市前年份的 DPS 在输入中直接为 0。
    weights = [0.40, 0.27, 0.18, 0.10, 0.05]
    dps_new_to_old = list(reversed(s.dps_history[-5:]))
    weighted_yield = sum(w * (dps / s.price) for w, dps in zip(weights, dps_new_to_old))
    score += min(60, weighted_yield / 0.04 * 60)

    # (b) 分红率风险过滤 P_filter (10 分)
    pr = s.payout_ratio_3y_avg
    if pr is None:
        score += 5
    elif pr <= 80:
        score += 10

    # (c) 经营现金流质量 C (10 分)
    # OCF / 净利润不依赖分红率; 即使公司暂不分红，也能评价利润现金含量。
    ocf_over_ni = mean(last_3y_ocf / last_3y_net_profit)
    if ocf_over_ni >= 1.0:
        score += 10
    elif ocf_over_ni >= 0.8:
        score += 6
    elif ocf_over_ni >= 0.5:
        score += 3

    # (d) 自由现金流覆盖分红 F (5 分)
    # 使用每股自由现金流 / 每股现金分红; 缺失不补虚假分。
    if s.fcf_per_share is not None and latest_dps > 0:
        fcf_cover = s.fcf_per_share / latest_dps
        if fcf_cover >= 1.2:
            score += 5
        elif fcf_cover >= 1.0:
            score += 3

    # (e) 盈利稳定性 S (15 分)
    # ROE 标准差小,分红可预测性高。
    if   s.roe_5y_std < 3: score += 15
    elif s.roe_5y_std < 6: score += 8

    return min(100, score)
```

### 6.2 `value_score` (0-100)

```
def value_score(s, bucket, pe_pct, pb_pct) -> float:
    # 不同 bucket 看不同分位
    primary_pct = {
        "PE_STABLE":   pe_pct,
        "PB_CYCLICAL": pb_pct,
        "BANK":        pb_pct,
        "INSURANCE":   pb_pct,
        "SECURITIES":  pb_pct,
        "DIV_INCOME":  pe_pct,        # 也看股息率分位,见下
        "NAV_RE":      pb_pct,
        "CYCLICAL_AG": pb_pct,
        "DEFENSE":     pe_pct,
    }[bucket]

    # 基础分: 分位越低越高分
    score = 100 * (1 - primary_pct)

    # 衰减: 极端低分位可能是价值陷阱
    if primary_pct < 0.05:
        # 检查盈利质量
        if s.roe_3y_avg < 5 or s.net_profit_ttm < 0:
            score *= 0.5      # 价值陷阱惩罚
        elif s.debt_to_asset > 70 and bucket != "BANK":
            score *= 0.7

    # 衰减: 基本面退化不能因为估值便宜而被忽略
    if revenue_decline_streak(s, n=2):
        score *= 0.7          # 营收连续下滑，历史估值中枢可能失效

    # DIV_INCOME 加成: 当前股息率 > 5y 中位数,加 10 分
    if bucket == "DIV_INCOME":
        dy_now = s.dps_history[-1] / s.price
        dy_median_5y = median_div_yield_5y(s)
        if dy_now > dy_median_5y * 1.1:
            score = min(100, score + 10)

    return max(0, score)
```

### 6.3 `t_score` 与做 T 区间

针对**小幅做 T**: 需要振幅适中、流动性好、有可识别的均值回归区间。

```
def t_score(s: StockInput) -> float:
    score = 0

    # (a) 波动率甜蜜点: 年化 20%-40% (40 分)
    v = s.vol_60d_annual
    if   0.20 <= v <= 0.40: score += 40
    elif 0.15 <= v <= 0.50: score += 25
    elif 0.10 <= v <= 0.60: score += 10
    # 太低(死水)或太高(妖股)都不适合

    # (b) 流动性 (30 分)
    avg_turnover = mean(s.volume_series.tail(20))    # 元
    if   avg_turnover > 5e8: score += 30
    elif avg_turnover > 1e8: score += 20
    elif avg_turnover > 3e7: score += 10

    # (c) Beta 接近 1 (20 分): 跟随大盘,做 T 信号更有效
    b = s.beta_60d
    if   0.7 <= b <= 1.3: score += 20
    elif 0.5 <= b <= 1.6: score += 10

    # (d) 价格非单边趋势 (10 分)
    # 60 日涨跌幅在 ±15% 内,认为是震荡状态
    ret_60d = s.close_series.iloc[-1] / s.close_series.iloc[-60] - 1
    if abs(ret_60d) < 0.15: score += 10
    elif abs(ret_60d) < 0.25: score += 5

    return min(100, score)


def t_band(s: StockInput, fair_mid: float) -> tuple[float, float]:
    """做 T 的下沿(补仓位)与上沿(减仓位)"""
    # 思路: 取 60 日 Bollinger(20, 2σ),与估值带交叉裁剪
    ma60 = s.close_series.tail(60).mean()
    std60 = s.close_series.tail(60).std()
    boll_low  = ma60 - 1.5 * std60
    boll_high = ma60 + 1.5 * std60

    # 下沿: 不低于 fair_mid * 0.85;上沿: 不高于 fair_mid * 1.15
    t_low  = max(boll_low,  fair_mid * 0.85)
    t_high = min(boll_high, fair_mid * 1.15)
    return t_low, t_high
```

## 7. 红旗检查 (Red Flags)

每条命中追加到 `flags`,**不直接否决**,由调用方/上层 agent 综合判断。

| Flag ID | 触发条件 | 严重度 |
|---|---|---|
| `negative_profit` | `net_profit_ttm < 0` | 高 |
| `loss_streak` | 近 3 年内有 ≥2 年亏损 | 高 |
| `revenue_decline_streak` | 营业收入连续若干期下滑 | 高 |
| `net_profit_decline_streak` | 净利润连续若干期下滑 | 高 |
| `goodwill_heavy` | 商誉 / 净资产 > 30% | 中 |
| `over_leverage` | 非金融股 `debt_to_asset > 70` | 中 |
| `dividend_unstable` | 5 年内有断档/暴跌 | 高(对本策略) |
| `ocf_disparity` | OCF / 净利润 < 0.5,持续 2 年 | 中 |
| `ar_anomaly` | 应收账款年度增幅 > 40% | 中 |
| `inventory_anomaly` | 存货年度增幅 > 40% | 中 |
| `pledge_heavy` | 大股东高比例质押 | 高(当前缓存不可得，需新增数据源后启用) |
| `cycle_top_risk` | `PB_CYCLICAL` 中 `pe<5 & pb>70 分位` | 高 |
| `value_trap` | 估值分位 < 5% 且 ROE 趋势下行 | 高 |
| `low_liquidity` | 20 日均成交 < 3000 万元 | 中(影响做 T) |
| `tiny_cap` | 总市值 < 30 亿 | 低 |
| `state_bank_value_trap` | 见 §5.3 | 低(常态) |
| `re_high_leverage` / `re_liquidity_risk` / `re_private_developer_risk` | 见 §5.7 | 高 |
| `data_insufficient` | 历史 < 1000 个交易日 | 高(本技能不适用) |

## 8. 一个最小 worked example

输入(贵州茅台,简化):
```
symbol = "600519", industry = "白酒Ⅱ"
price = 1500, eps_ttm = 70, bvps = 220
roe_3y = 30%, payout_ratio_3y = 50%
dps_history = [21.7, 25.9, 27.6, 30.9, 36.5]   # 元
pe_5y_25/50/75 = 28 / 35 / 45
vol_60d_annual = 0.22, beta = 0.95
```

执行:
```
bucket = PE_STABLE
fair_low  = 28 * 70 = 1960
fair_mid  = 35 * 70 = 2450
fair_high = 45 * 70 = 3150
margin_of_safety = (2450-1500)/1500 = 63%

5 年加权股息率 ≈ 2.1% → X ≈ 31.6 分
分红率 50% → P_filter 10 分
OCF 覆盖充足 → F 15 分
ROE 极稳 → S 15 分
dividend_score ≈ 71.6

pe 分位假设 0.10 → value_score ≈ 90
t_score: vol 0.22 → 40, 流动性极佳 → 30, beta 0.95 → 20, 60 日震荡 → 10
       = 100

strategy_fit = 0.45*71.6 + 0.35*90 + 0.20*100 = 32.2 + 31.5 + 20 = 83.7
flags = []
t_band = (1480, 1700) 大约
```

## 9. akshare 数据接口速查(供调用方实现 Loader)

```python
# 估值时序(PE/PB/PS/股息率,长江证券口径,日频)
ak.stock_a_indicator_lg(symbol="600519")

# 财务三大表 + 财务指标
ak.stock_financial_em(stock="600519", symbol="资产负债表")
ak.stock_financial_em(stock="600519", symbol="利润表")
ak.stock_financial_em(stock="600519", symbol="现金流量表")
ak.stock_financial_analysis_indicator(symbol="600519")

# 当前实现优先使用的缓存口径
ak.stock_financial_abstract(symbol="600519")                 # OCF、ROE、利润率、每股自由现金流等
ak.stock_balance_sheet_by_yearly_em(symbol="SH600519")       # 应收、存货、商誉、现金、短债等

# 分红
ak.stock_dividend_cninfo(symbol="600519")

# 行情
ak.stock_zh_a_hist(symbol="600519", period="daily", adjust="qfq")

# 主营构成(用于行业归属/SOTP 拆分)
ak.stock_zygc_em(symbol="SH600519")

# 一致预期
ak.stock_profit_forecast_em(symbol="600519")

# 行业分类
ak.sw_index_third_info()                # 申万三级
ak.stock_board_industry_name_em()       # 东财行业(备用)
```

## 10. 实现注意事项

1. **不要 fail-fast**: 任何缺失字段使用 `None`/`NaN`,在该字段相关的子分计 0,继续输出其他维度。
2. **历史分位时序对齐**: 计算 PE 分位时需剔除 `pe_ttm <= 0` 的天(亏损期 PE 无意义)。
3. **复权**: 所有价格/EPS/DPS 计算需用**前复权**口径,保持时序一致。
4. **冷启动**: 上市 < 1 年的股票直接 `flags=["data_insufficient"]` 并跳过分位计算。
5. **bucket 兜底**: 若行业名称不在 §4 表中(罕见,如新增三级),默认归 `PE_STABLE` 并加 flag `bucket_fallback`。
6. **免五策略下的 dividend_score 校准**: 因免税带来 ~20% 真实收益增益,本评分体系的 X 子项**已默认按税前 DPS/股息率算**,无需额外加权。但如果调用方做策略对比时纳入了短线持有(<1年)的 case,需对短线情形将各年 DPS 或 5 年加权股息率乘 0.8 后再评分。
7. **缓存可得性边界**: 当前缓存已覆盖 OCF、每股自由现金流、ROE、毛利率、净利率、资产负债率、应收、存货、商誉、现金和短债；未覆盖大股东质押比例，`pledge_heavy` 不能仅凭现有缓存可靠触发。
8. **做 T 区间的实操**: `t_band` 输出仅是参考价位,实际下单需结合调用方的仓位管理与单笔交易额限制(避免冲击成本吃掉 T 收益)。

---

**版本**: v1.1
**适用市场**: A 股(沪/深/北),不含港美股
**复审建议**: 季度更新各 bucket 的中枢参数(§5 表格);年度复盘 strategy_fit 权重与历史持仓表现的相关性。
