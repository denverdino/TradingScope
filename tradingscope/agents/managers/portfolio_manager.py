"""Portfolio Manager Agent for TradingScope using AgentScope framework."""

from __future__ import annotations

from agentscope import logger

# AgentScope imports
from agentscope.agent import Agent
from agentscope.agent._config import ReActConfig
from agentscope.tool import FunctionTool, Toolkit

# Local imports
from tradingscope.agents.utils.agent_utils import COMPLIANCE_PROMPT
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.tool_response_helper import code_interpreter


def create_portfolio_manager_agent(
    context: AgentContext,
    name: str = "PortfolioManager",
) -> Agent:
    """Create Portfolio Manager Agent that evaluates risk analysis debates and makes final decisions.

    Args:
        context: AgentContext instance containing all necessary context information
        name: Agent name

    Returns:
        Agent: Configured portfolio manager agent
    """
    company_of_interest = context.company_of_interest
    trade_date = context.trade_date
    latest_trading_date = context.latest_trading_date

    system_message = f"""{COMPLIANCE_PROMPT}

作为风险管理委员会主席和辩论主持人，您的目标是评估三位风险分析师——激进、中性和安全/保守——之间的辩论，制定最佳交易行动方案。

决策指导原则：
1. **总结关键论点**：提取每位分析师的最强观点，重点关注与背景的相关性。
2. **提供理由**：用辩论中的直接引用和反驳论点支持您的建议。
3. **完善交易计划**：根据分析师的见解调整交易员的原始交易员操作计划。
4. **从过去的错误中学习**：使用长期记忆中的历史经验教训来改进决策，确保不会重复过去的误判。

交付成果：
- 明确且可操作的建议：买入、卖出或持有。
- 基于辩论和过去反思的详细推理。

专注于可操作的见解和持续改进。建立在过去经验教训的基础上，批判性地评估所有观点，确保每个决策都能带来更好的结果。请用中文撰写所有分析内容和建议。

**重要：您的回复必须以如下格式开头：**

- **股票代码**：{company_of_interest}
- **交易日期**：{trade_date}
- **最新美股交易日期**：{latest_trading_date}
- **交易决策**：【买入/卖出/持有】


然后按以下格式提供详细分析：

### 最终决策
- 交易决策：买入/卖出/持有
- 置信度：0.x（0-1之间）
- 风险评分：0.x（0为低风险，1为高风险）

### 辩论观点总结
- **激进派观点**：关键论点摘要
- **保守派观点**：关键论点摘要
- **中性派观点**：关键论点摘要
- **采纳理由**：解释最终决策的依据

### 优化后的交易计划
- **入场价位**：xxx
- **止损价位**：xxx（含依据）
- **目标价位**：xxx
- **仓位建议**：轻仓/中等仓位/重仓
- **盈亏比**：x:1

### 风险控制措施
- 列出2-3条具体的风险控制措施

### 失效条件
- 明确说明交易计划失效的条件

# 可用资源：

{context.generate_risk_evaluation_context_md()}"""

    toolkit = Toolkit(tools=[FunctionTool(code_interpreter)])

    # 创建 Agent
    agent = Agent(
        name=name,
        system_prompt=system_message,
        model=context.model,
        toolkit=toolkit,
        react_config=ReActConfig(max_iters=8),
    )

    logger.debug("📊 [DEBUG] ===== 投资组合经理 Agent 创建完成 =====")
    return agent
