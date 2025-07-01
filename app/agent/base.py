from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import ROLE_TYPE, AgentState, Memory, Message


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.
    用于管理代理状态和执行的抽象基类。

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    提供状态转换、内存管理和基于步骤的执行循环的基础功能。子类必须实现 `step` 方法。
    """

    # Core attributes
    # 核心属性
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts
    # 提示
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    # 依赖项
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )

    # Execution control
    # 执行控制
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    duplicate_threshold: int = 2

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility in subclasses
        # 允许额外字段以增加子类的灵活性

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        """如果未提供，则使用默认设置初始化代理。"""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe agent state transitions.
        用于安全代理状态转换的上下文管理器。

        Args:
            new_state: The state to transition to during the context.
                       在上下文中要转换到的状态。

        Yields:
            None: Allows execution within the new state.
                  允许在新状态下执行。

        Raises:
            ValueError: If the new_state is invalid.
                        如果 new_state 无效。
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            # 失败时转换到 ERROR 状态
            raise e
        finally:
            self.state = previous_state  # Revert to previous state
            # 恢复到先前的状态

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.
        向代理的内存中添加一条消息。

        Args:
            role: The role of the message sender (user, system, assistant, tool).
                  消息发送者的角色（用户、系统、助手、工具）。
            content: The message content.
                     消息内容。
            base64_image: Optional base64 encoded image.
                          可选的 base64 编码图像。
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).
                      其他参数（例如，工具消息的 tool_call_id）。

        Raises:
            ValueError: If the role is unsupported.
                        如果角色不受支持。
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        # Create message with appropriate parameters based on role
        # 根据角色使用适当的参数创建消息
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.
        异步执行代理的主循环。

        Args:
            request: Optional initial user request to process.
                     可选的要处理的初始用户请求。

        Returns:
            A string summarizing the execution results.
            一个总结执行结果的字符串。

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
                          如果代理在启动时状态不是 IDLE。
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                step_result = await self.step()

                # Check for stuck state
                # 检查是否陷入停滞状态
                if self.is_stuck():
                    self.handle_stuck_state()

                results.append(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
        await SANDBOX_CLIENT.cleanup()
        return "\n".join(results) if results else "No steps executed"

    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow.
        在代理的工作流程中执行单个步骤。

        Must be implemented by subclasses to define specific behavior.
        必须由子类实现以定义特定行为。
        """

    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        """通过添加提示来改变策略来处理卡住状态"""
        stuck_prompt = "\
        Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        """通过检测重复内容来检查代理是否陷入循环"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # Count identical content occurrences
        # 计算相同内容的出现次数
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        """从代理的内存中检索消息列表。"""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        """设置代理内存中的消息列表。"""
        self.memory.messages = value
