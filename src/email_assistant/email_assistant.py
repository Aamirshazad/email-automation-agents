from typing import Literal

from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

from email_assistant.tools import get_tools, get_tools_by_name
from email_assistant.tools.default.prompt_templates import HITL_TOOLS_PROMPT
from email_assistant.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt_hitl, default_background, default_triage_instructions, default_response_preferences, default_cal_preferences
from email_assistant.schemas import State, RouterSchema, StateInput
from email_assistant.utils import parse_email, format_for_display, format_email_markdown



llm = init_chat_model("openai:gpt-4")
llm_router = llm.with_structured_output(RouterSchema)

# router node
def triage_router(state: State) -> Command[Literal["notify_agent", "email_agent", "__end__"]]:
    """Handle interrupts from the triage step"""

    email_markdown = format_email_markdown(state["email_input"])

    system_prompt = triage_system_prompt.format(
        backgraound=default_background,
        triage_instruction = default_triage_instructions,
    )

    result = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": email_markdown}
        ]
    )
    # Decision
    classification = result.classifiation

    if classification == "respond":
        goto = "response_agent"

        update = {
            "classificatation_decision": result.classification,
            "messages": [{
                "role": "user",
                "contnt": f "Respond to this email {email_markdown}"
            }]
        }

    elif classification == "notify":
        goto = "notify"
        update = {
            "classification_desion": result.classification,
            "messages": [{
                "role": "user",
                "content": f"this is notify from user{}"
            }]
        }

    return Command(goto=goto, update=update)


from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

system_prompt = agent_system_prompt_hitl.format(
                    tools_prompt = HITL_TOOLS_PROMPT,
                    background = default_background,
                    default_response_preferences = default_response_preferences,
                    cal_preferences = default_cal_preferences
                
                )

tools = get_tools(["write_email", "schedule_meeting", "check_calendar_availability", "Question", "Done"])


email_agent = create_agent(
        model = llm,
        tools = ["write_email", "schedule_meeting", "check_calendar_availability", "Question", "Done"],
        promt = system_prompt,
        middleware= [
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_messages": True,
                    "schdule_metting": True,
                    "Question": {"allow_respond":True},
                    "check_calender": True,
                },
                description_prefix="Plase answer given question",
                                
                
            ),
        ],
        checkpointer =InMemoryScaver(),

    ) 




notify_agent = create_agent(
    model = llm,
    tools = ["notify"]
    prompt = "",
    HumanInTheLoopMiddleware = [
        interrup_on = {
            "notify": {"all_response": True}
        }
    ]
    checkpointer=InMemorySaver()
)


workflow = StateGraph(State)
workflow.add_node("triage_router",triage_router)
workflow.add_node("email_agent", email_agent)
workflow.add_node("notify_agent", notify_agent)

workflow.add_edge(START, "triage_router")

email_assistant = workflow.compile()
